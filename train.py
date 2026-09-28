# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from contextlib import nullcontext
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
from diffusers import StableDiffusion3Pipeline
import numpy as np
import flow_grpo.rewards
from flow_grpo.pareto_balancer import (
    balance_reward_gradients,
    capture_grad_vector,
    clear_parameter_grads,
    restore_grad_vector,
    sanitize_grad_vector,
    sync_grad_vector,
    sync_grad_vectors,
)
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.mixed_dataset import MixedPromptDataset
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/nft.py:sd3_multi_reward_pareto_5r_alpha_ema", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


class StratifiedKRepeatSampler(Sampler):
    """Samples equal number of prompts from each dataset source per iteration.

    For a mixed dataset with sources [pickscore, ocr, geneval] and m=3 total
    prompts per iteration, this samples 1 prompt from each source, ensuring
    balanced reward signal across all sources.
    """

    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0, source_weights=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0
        self.m = self.total_samples // self.k  # total unique prompts per iteration

        # Build per-source index lists
        self.source_indices = {}
        for i, meta in enumerate(dataset.metadatas):
            source = meta.get("source", "unknown")
            if source not in self.source_indices:
                self.source_indices[source] = []
            self.source_indices[source].append(i)
        self.sources = sorted(self.source_indices.keys())

        # Allocate prompts per source. Without source_weights this is uniform
        # (with leftover spread across the first `remainder` sources, matching
        # the original behaviour). With source_weights it is proportional to
        # the weights, with rounding remainder going to sources with the
        # largest fractional part — so {pickscore:2, ocr:1, geneval:2} on m=30
        # yields 12 / 6 / 12.
        if source_weights is None:
            base = self.m // len(self.sources)
            self.prompts_per_source_dict = {s: base for s in self.sources}
            for i in range(self.m - base * len(self.sources)):
                self.prompts_per_source_dict[self.sources[i]] += 1
        else:
            total_w = sum(source_weights.get(s, 1.0) for s in self.sources)
            raw = {s: self.m * source_weights.get(s, 1.0) / total_w for s in self.sources}
            floored = {s: int(raw[s]) for s in self.sources}
            remainder = self.m - sum(floored.values())
            for s in sorted(self.sources, key=lambda x: -(raw[x] - floored[x]))[:remainder]:
                floored[s] += 1
            self.prompts_per_source_dict = floored
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            # Sample prompts from each source according to allocated counts
            indices = []
            for source in self.sources:
                src_ids = self.source_indices[source]
                n = self.prompts_per_source_dict[source]
                perm = torch.randperm(len(src_ids), generator=g)
                for j in range(n):
                    indices.append(src_ids[perm[j % len(src_ids)].item()])

            # Repeat each prompt k times
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            # Shuffle and distribute
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def permute_batch_tensors(batch_dict, perm):
    output = {}
    for key, value in batch_dict.items():
        if isinstance(value, dict):
            output[key] = {sub_key: sub_value[perm] for sub_key, sub_value in value.items()}
        else:
            output[key] = value[perm]
    return output


def slice_batch_tensors(batch_dict, start, end):
    output = {}
    for key, value in batch_dict.items():
        if isinstance(value, dict):
            output[key] = {sub_key: sub_value[start:end] for sub_key, sub_value in value.items()}
        else:
            output[key] = value[start:end]
    return output


def compute_policy_and_kl_terms(
    x0,
    xt,
    t_expanded,
    forward_prediction,
    old_prediction,
    ref_forward_prediction,
    advantages_t,
    config,
    loss_mask=None,
):
    loss_terms = {}
    advantages_clip = torch.clamp(
        advantages_t,
        -config.train.adv_clip_max,
        config.train.adv_clip_max,
    )
    if hasattr(config.train, "adv_mode"):
        if config.train.adv_mode == "positive_only":
            advantages_clip = torch.clamp(advantages_clip, 0, config.train.adv_clip_max)
        elif config.train.adv_mode == "negative_only":
            advantages_clip = torch.clamp(advantages_clip, -config.train.adv_clip_max, 0)
        elif config.train.adv_mode == "one_only":
            advantages_clip = torch.where(
                advantages_clip > 0,
                torch.ones_like(advantages_clip),
                torch.zeros_like(advantages_clip),
            )
        elif config.train.adv_mode == "binary":
            advantages_clip = torch.sign(advantages_clip)

    normalized_advantages_clip = (advantages_clip / config.train.adv_clip_max) / 2.0 + 0.5
    r = torch.clamp(normalized_advantages_clip, 0, 1)

    # Normalize r via z-score then shift to mean 0.5
    if getattr(config, "r_normalize_mean", False):
        r = (r - r.mean()) / (r.std() + 1e-4) + 0.5
        r = torch.clamp(r, 0, 1)

    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
    loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach()
    loss_terms["old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2).detach()

    positive_prediction = config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach()
    implicit_negative_prediction = (1.0 + config.beta) * old_prediction.detach() - config.beta * forward_prediction

    x0_prediction = xt - t_expanded * positive_prediction
    with torch.no_grad():
        weight_factor = (
            torch.abs(x0_prediction.double() - x0.double())
            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
            .clip(min=0.00001)
        )
    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))

    negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
    with torch.no_grad():
        negative_weight_factor = (
            torch.abs(negative_x0_prediction.double() - x0.double())
            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
            .clip(min=0.00001)
        )
    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
        dim=tuple(range(1, x0.ndim))
    )

    ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
    if loss_mask is not None:
        ori_policy_loss = ori_policy_loss * loss_mask
        n_valid = loss_mask.sum().clamp(min=1)
        policy_loss = (ori_policy_loss * config.train.adv_clip_max).sum() / n_valid
    else:
        policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()

    kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(
        dim=tuple(range(1, x0.ndim))
    )
    kl_div_loss = torch.mean(kl_div_loss)

    loss_terms["policy_loss"] = policy_loss.detach()
    loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach()
    loss_terms["kl_div_loss"] = kl_div_loss.detach()
    loss_terms["kl_div"] = torch.mean(
        ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
    ).detach()
    loss_terms["old_kl_div"] = torch.mean(
        ((old_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
    ).detach()

    return policy_loss, kl_div_loss, loss_terms


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def return_decay(step, decay_type, uphold_override=None, uprate_override=None):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.95
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if uphold_override is not None:
        uphold = uphold_override
    if uprate_override is not None:
        uprate = uprate_override

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def eval_fn(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.transformer.eval()

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,  # This is per-GPU batch size
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not is_main_process(rank),
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
        )
        current_batch_size = len(prompt_embeds)
        if current_batch_size < len(sample_neg_prompt_embeds):  # Handle last batch
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds[:current_batch_size]
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:current_batch_size]
        else:
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds

        with torch_autocast(enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=current_sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=current_sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver="flow",
                    model_type="sd3",
                )

        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        # Only gather a fixed set of keys so all ranks call gather the same
        # number of times.  Geneval group keys (e.g. color_attr_strict_accuracy)
        # vary across ranks and would cause a deadlock in gather_tensor_to_all.
        eval_gather_keys = list(config.reward_fn.keys()) + ["avg", "accuracy", "strict_accuracy"]
        current_batch_size = len(prompts)
        for key in eval_gather_keys:
            if key in rewards:
                rewards_tensor = torch.as_tensor(rewards[key], device=device).float()
            else:
                rewards_tensor = torch.full((current_batch_size,), float("nan"), device=device)
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

    if is_main_process(rank):
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}

        images_to_log = images.cpu()
        prompts_to_log = prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(images_to_log))
            for idx in range(num_samples_to_log):
                image = images_to_log[idx].float()
                pil = Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(num_samples_to_log)]

            wandb_log_safe(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts_log, sampled_rewards_log))
                    ],
                    **{
                        f"eval_reward_{key}": float(np.nanmean(value[(value != -10) & ~np.isnan(value)]))
                        if np.any((value != -10) & ~np.isnan(value))
                        else 0.0
                        for key, value in final_rewards.items()
                    },
                },
                step=global_step,
            )

    # Broadcast eval scores to all ranks for best-checkpoint tracking
    eval_scores = {}
    if is_main_process(rank):
        for key, value in final_rewards.items():
            valid = value[(value != -10) & ~np.isnan(value)]
            eval_scores[key] = float(np.nanmean(valid)) if len(valid) > 0 else 0.0

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier()

    return eval_scores


def wandb_log_safe(data, step=None):
    """Log to wandb."""
    wandb.log(data, step=step)


def save_ckpt(
    save_dir, transformer_ddp, global_step, epoch, rank, ema, transformer_trainable_parameters, config, optimizer, scaler,
    cached_pareto_alphas=None, last_full_mgda_step=-1, reward_ema_means=None, lr_scheduler=None,
):
    if is_main_process(rank):
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)

        model_to_save = transformer_ddp.module

        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        model_to_save.save_pretrained(save_root_lora)  # For LoRA/PEFT models

        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))
        training_state = {"global_step": global_step, "epoch": epoch}
        if reward_ema_means:
            training_state["reward_ema_means"] = reward_ema_means
        if lr_scheduler is not None:
            training_state["lr_scheduler"] = lr_scheduler.state_dict()
        torch.save(training_state, os.path.join(save_root, "training_state.pt"))
        if cached_pareto_alphas is not None:
            torch.save(
                {"cached_pareto_alphas": cached_pareto_alphas, "last_full_mgda_step": last_full_mgda_step},
                os.path.join(save_root, "amortized_mgda_state.pt"),
            )

        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)
        logger.info(f"Saved checkpoint to {save_root}")


def main(_):
    config = FLAGS.config

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    # --- Auto-resume: find latest checkpoint ---
    if config.auto_resume and not config.resume_from:
        # Search both {save_dir}/checkpoints/ (new) and {save_dir}/ (old layout)
        candidate_dirs = [
            os.path.join(config.save_dir, "checkpoints"),
            config.save_dir,
        ]
        best_step, best_path = -1, None
        for cdir in candidate_dirs:
            if not os.path.isdir(cdir):
                continue
            for d in os.listdir(cdir):
                if d.startswith("checkpoint-"):
                    try:
                        step = int(d.split("-")[-1])
                        if step > best_step:
                            best_step = step
                            best_path = os.path.join(cdir, d)
                    except ValueError:
                        continue
        if best_path:
            config.resume_from = best_path
            logger.info(f"Auto-resume: found {config.resume_from}")

    # --- Run name (always append timestamp, even on resume — new wandb run each time) ---
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # --- WandB Init (only on main process, always new run) ---
    if is_main_process(rank):
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(project="flow-grpo", name=config.run_name, config=config.to_dict(), dir=log_dir)
    logger.info(f"\n{config}")

    set_seed(config.seed, rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    # --- Load pipeline and models ---
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    transformer = pipeline.transformer.to(device)

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)
        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")
    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("default")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,  # Use params from original model for optimizer
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- LR Scheduler ---
    lr_scheduler = None
    if getattr(config.train, "lr_min_ratio", 0.0) > 0:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        lr_total_steps = getattr(config.train, "lr_total_steps", 0) or config.num_epochs
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=lr_total_steps,
            eta_min=config.train.learning_rate * config.train.lr_min_ratio,
        )
        if is_main_process(rank):
            logger.info(f"LR scheduler: cosine decay over {lr_total_steps} steps, min_lr={config.train.learning_rate * config.train.lr_min_ratio:.2e}")

    # --- Datasets and Dataloaders ---
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "mixed":
        max_per_source = getattr(config, "max_samples_per_source", None)
        train_dataset = MixedPromptDataset(config.dataset, "train", max_samples_per_source=max_per_source)
        test_dataset = MixedPromptDataset(config.dataset, "test", max_samples_per_source=max_per_source)
        if is_main_process(rank):
            logger.info(f"Mixed dataset: {len(train_dataset)} train, {len(test_dataset)} test prompts")
    else:
        raise NotImplementedError("Prompt function not supported with dataset")

    use_stratified = getattr(config, "stratified_sampling", False) and config.prompt_fn == "mixed"
    sampler_cls = StratifiedKRepeatSampler if use_stratified else DistributedKRepeatSampler
    stratified_source_weights = getattr(config, "stratified_source_weights", None)
    if use_stratified and is_main_process(rank):
        source_counts = {}
        for meta in train_dataset.metadatas:
            s = meta.get("source", "unknown")
            source_counts[s] = source_counts.get(s, 0) + 1
        logger.info(f"Stratified sampling enabled. Source distribution: {source_counts}")
        if stratified_source_weights:
            logger.info(f"Stratified source weights: {dict(stratified_source_weights)}")
    sampler_kwargs = dict(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
    )
    if use_stratified and stratified_source_weights:
        sampler_kwargs["source_weights"] = dict(stratified_source_weights)
    train_sampler = sampler_cls(**sampler_kwargs)
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=train_dataset.collate_fn, pin_memory=True
    )

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,  # Per-GPU
        sampler=test_sampler,  # Use distributed sampler for eval
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # --- Prompt Embeddings ---
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    pareto_enabled = hasattr(config, "pareto") and config.pareto.enabled
    reward_names = list(config.reward_fn.keys())

    # Pre-compute reward grouping / weighting for MGDA
    reward_groups = dict(config.pareto.reward_groups) if pareto_enabled and config.pareto.reward_groups else {}
    reward_weights_cfg = dict(config.pareto.reward_weights) if pareto_enabled and config.pareto.reward_weights else {}
    if reward_groups:
        mgda_group_members = {}  # group_name -> [indices in reward_names]
        grouped_reward_set = set()
        mgda_names = []
        for group_name, members in reward_groups.items():
            member_indices = [i for i, rn in enumerate(reward_names) if rn in members]
            if member_indices:
                mgda_group_members[group_name] = member_indices
                mgda_names.append(group_name)
                grouped_reward_set.update(reward_names[i] for i in member_indices)
        for i, rn in enumerate(reward_names):
            if rn not in grouped_reward_set:
                mgda_names.append(rn)
        # Map reward_name -> (mgda_name, split_factor) for amortized path
        reward_to_mgda = {}
        for group_name, member_indices in mgda_group_members.items():
            for i in member_indices:
                reward_to_mgda[reward_names[i]] = (group_name, len(member_indices))
        for rn in reward_names:
            if rn not in reward_to_mgda:
                reward_to_mgda[rn] = (rn, 1)
        if is_main_process(rank):
            logger.info(f"Reward grouping: {reward_groups} -> MGDA names: {mgda_names}")
    else:
        mgda_group_members = {}
        mgda_names = list(reward_names)
        reward_to_mgda = {rn: (rn, 1) for rn in reward_names}

    mgda_weights = [reward_weights_cfg.get(name, 1.0) for name in mgda_names] if reward_weights_cfg else None
    if mgda_weights and is_main_process(rank):
        logger.info(f"MGDA weights: {dict(zip(mgda_names, mgda_weights))}")

    # Pre-compute reward dropout config
    reward_dropout_targets = (
        set(config.pareto.reward_dropout_targets)
        if pareto_enabled and hasattr(config.pareto, 'reward_dropout_targets') and config.pareto.reward_dropout_targets
        else set()
    )
    reward_dropout_max_prob = (
        config.pareto.reward_dropout_prob
        if pareto_enabled and hasattr(config.pareto, 'reward_dropout_prob')
        else 0.0
    )
    reward_dropout_warmup_steps = (
        config.pareto.reward_dropout_warmup_steps
        if pareto_enabled and hasattr(config.pareto, 'reward_dropout_warmup_steps')
        else 0
    )
    if reward_dropout_targets and is_main_process(rank):
        logger.info(
            f"Reward dropout: targets={reward_dropout_targets}, max_prob={reward_dropout_max_prob}, "
            f"warmup_steps={reward_dropout_warmup_steps}"
        )

    # Pre-compute alpha floor / EMA config
    alpha_floor = config.pareto.alpha_floor if pareto_enabled and hasattr(config.pareto, 'alpha_floor') else 0.0
    alpha_ema_decay = config.pareto.alpha_ema_decay if pareto_enabled and hasattr(config.pareto, 'alpha_ema_decay') else 0.0
    if (alpha_floor > 0 or alpha_ema_decay > 0) and is_main_process(rank):
        logger.info(f"Alpha stabilization: floor={alpha_floor}, ema_decay={alpha_ema_decay}")

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
        per_reward_stat_trackers = (
            {
                reward_name: PerPromptStatTracker(config.sample.global_std)
                for reward_name in reward_names
            }
            if pareto_enabled
            else {}
        )
    else:
        stat_tracker = None
        per_reward_stat_trackers = {}

    executor = futures.ThreadPoolExecutor(max_workers=8)  # Async reward computation

    # Train!
    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)  # Pass device
    eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)  # Pass device

    # Amortized MGDA state
    cached_pareto_alphas = None  # dict: {reward_name: float}
    equal_alpha = getattr(config.pareto, "equal_alpha", False)
    amortized_mgda_enabled = pareto_enabled and (config.pareto.amortized or equal_alpha)
    amortize_every_n = config.pareto.amortize_every_n if amortized_mgda_enabled else 1
    amortize_warmup = config.pareto.amortize_warmup if amortized_mgda_enabled else 0
    last_full_mgda_step = -1
    if equal_alpha and pareto_enabled:
        # Fixed alphas — never run MGDA solve.  If `pareto.fixed_alphas` is
        # set, use those values (must cover all mgda_names and sum to 1);
        # otherwise default to uniform 1/n.
        n_rewards = len(mgda_names)
        fixed_alphas_cfg = getattr(config.pareto, "fixed_alphas", None)
        if fixed_alphas_cfg:
            missing = [n for n in mgda_names if n not in fixed_alphas_cfg]
            if missing:
                raise ValueError(f"fixed_alphas missing entries for: {missing}")
            cached_pareto_alphas = {name: float(fixed_alphas_cfg[name]) for name in mgda_names}
            s = sum(cached_pareto_alphas.values())
            if abs(s - 1.0) > 1e-6:
                raise ValueError(f"fixed_alphas must sum to 1, got {s}: {cached_pareto_alphas}")
        else:
            cached_pareto_alphas = {name: 1.0 / n_rewards for name in mgda_names}
        last_full_mgda_step = float("inf")
        amortize_every_n = float("inf")
        if is_main_process(rank):
            logger.info(f"Equal alpha mode: {cached_pareto_alphas}")

    # EMA reward normalization state
    reward_ema_means = {}

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        # Assuming checkpoint dir contains lora, optimizer.pt, scaler.pt
        lora_path = os.path.join(config.resume_from, "lora")
        if os.path.exists(lora_path):  # Check if it's a PEFT model save
            transformer_ddp.module.load_adapter(lora_path, adapter_name="default", is_trainable=True)
            transformer_ddp.module.load_adapter(lora_path, adapter_name="old", is_trainable=False)
        else:  # Try loading full state dict if it's not a PEFT save structure
            model_ckpt_path = os.path.join(config.resume_from, "transformer_model.pt")  # Or specific name
            if os.path.exists(model_ckpt_path):
                transformer_ddp.module.load_state_dict(torch.load(model_ckpt_path, map_location=device))

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and enable_amp:
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))

        # Restore training state (global_step + epoch)
        state_path = os.path.join(config.resume_from, "training_state.pt")
        if os.path.exists(state_path):
            training_state = torch.load(state_path, map_location="cpu")
            global_step = training_state["global_step"]
            first_epoch = training_state["epoch"]
            if "reward_ema_means" in training_state:
                reward_ema_means = training_state["reward_ema_means"]
            if "lr_scheduler" in training_state and lr_scheduler is not None:
                lr_scheduler.load_state_dict(training_state["lr_scheduler"])
                logger.info(f"Resumed LR scheduler, current lr={optimizer.param_groups[0]['lr']:.2e}")
            logger.info(f"Resumed global_step={global_step}, epoch={first_epoch}")
        else:
            # Fallback for old checkpoints without training_state.pt
            try:
                global_step = int(os.path.basename(config.resume_from).split("-")[-1])
                # Estimate epoch from global_step (1 gradient step per epoch when gradient_step_per_epoch=1)
                first_epoch = global_step
                logger.info(f"Resumed global_step={global_step}, estimated epoch={first_epoch} (no training_state.pt)")
            except ValueError:
                logger.warning(
                    f"Could not parse global_step from checkpoint name: {config.resume_from}. Starting from 0."
                )
                global_step = 0

    # Restore amortized MGDA state if available (skip for equal_alpha — keep uniform alphas)
    if config.resume_from and amortized_mgda_enabled and not equal_alpha:
        amort_path = os.path.join(config.resume_from, "amortized_mgda_state.pt")
        if os.path.exists(amort_path):
            amort_state = torch.load(amort_path, map_location="cpu")
            cached_pareto_alphas = amort_state["cached_pareto_alphas"]
            last_full_mgda_step = amort_state["last_full_mgda_step"]
            # If the ckpt was saved by an equal_alpha run, last_full_mgda_step
            # is float("inf"), which would freeze MGDA solving forever in this
            # non-equal_alpha continuation. Also wipe cached_pareto_alphas so
            # the EMA blend at the first solve does not mix the stale equal-α
            # values (e.g. 1/K) into the fresh MGDA solution; with cached=None,
            # the trainer falls through to the "no-EMA fresh assignment" path
            # at the first full solve.
            if last_full_mgda_step == float("inf") or last_full_mgda_step is None:
                logger.info(
                    f"Resumed last_full_mgda_step={last_full_mgda_step} "
                    f"(likely from an equal_alpha ckpt); resetting to -1 and "
                    f"discarding cached_pareto_alphas={cached_pareto_alphas} "
                    f"to trigger a fresh MGDA solve with no EMA carryover."
                )
                last_full_mgda_step = -1
                cached_pareto_alphas = None
            else:
                logger.info(f"Resumed amortized MGDA: cached_alphas={cached_pareto_alphas}, last_full_step={last_full_mgda_step}")

    ema = None
    if config.train.ema:
        ema_decay = getattr(config.train, "ema_decay", 0.999)
        ema_warmup_rate = getattr(config.train, "ema_warmup_rate", 0.0)
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=ema_decay, update_step_interval=1, warmup_rate=ema_warmup_rate, device=device)
        if is_main_process(rank):
            logger.info(f"EMA enabled with decay={ema_decay}, warmup_rate={ema_warmup_rate}")

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    # Best checkpoint tracking
    best_eval_score = float("-inf")
    best_eval_step = -1

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    for src_param, tgt_param in zip(
        transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
    ):
        tgt_param.data.copy_(src_param.detach().data)
        assert src_param is not tgt_param

    for epoch in range(first_epoch, config.num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # SAMPLING
        pipeline.transformer.eval()
        samples_data_list = []

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            transformer_ddp.module.set_adapter("default")
            if hasattr(train_sampler, "set_epoch") and isinstance(train_sampler, (DistributedKRepeatSampler, StratifiedKRepeatSampler)):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            if i == 0 and epoch % config.eval_freq == 0 and not config.debug:
                eval_scores = eval_fn(
                    pipeline,
                    test_dataloader,
                    text_encoders,
                    tokenizers,
                    config,
                    device,
                    rank,
                    world_size,
                    global_step,
                    eval_reward_fn,
                    executor,
                    mixed_precision_dtype,
                    ema,
                    transformer_trainable_parameters,
                )

                # Best checkpoint selection (rank 0 only has eval_scores)
                if is_main_process(rank) and eval_scores:
                    reward_keys = [k for k in config.reward_fn.keys() if k in eval_scores]
                    if reward_keys:
                        composite = np.mean([eval_scores[k] for k in reward_keys])
                        if composite > best_eval_score:
                            best_eval_score = composite
                            best_eval_step = global_step
                            logger.info(f"New best eval score: {composite:.4f} at step {global_step}")
                            save_ckpt(
                                config.save_dir, transformer_ddp, global_step, epoch, rank,
                                ema, transformer_trainable_parameters, config, optimizer, scaler,
                                cached_pareto_alphas=cached_pareto_alphas,
                                last_full_mgda_step=last_full_mgda_step,
                                reward_ema_means=reward_ema_means,
                                lr_scheduler=lr_scheduler,
                            )
                            # Symlink best checkpoint
                            best_link = os.path.join(config.save_dir, "checkpoints", "best")
                            best_target = f"checkpoint-{global_step}"
                            if os.path.islink(best_link):
                                os.remove(best_link)
                            elif os.path.isdir(best_link):
                                import shutil
                                shutil.rmtree(best_link)
                            os.symlink(best_target, best_link)

            if i == 0 and epoch % config.save_freq == 0 and is_main_process(rank) and not config.debug:
                save_ckpt(
                    config.save_dir,
                    transformer_ddp,
                    global_step,
                    epoch,
                    rank,
                    ema,
                    transformer_trainable_parameters,
                    config,
                    optimizer,
                    scaler,
                    cached_pareto_alphas=cached_pareto_alphas,
                    last_full_mgda_step=last_full_mgda_step,
                    reward_ema_means=reward_ema_means,
                    lr_scheduler=lr_scheduler,
                )

            transformer_ddp.module.set_adapter("old")
            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images, latents, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        deterministic=config.sample.deterministic,
                        solver=config.sample.solver,
                        model_type="sd3",
                    )
            transformer_ddp.module.set_adapter("default")

            latents = torch.stack(latents, dim=1)
            timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device)

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            samples_data_list.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                    "latents_clean": latents[:, -1],
                    "rewards_future": rewards_future,  # Store future
                }
            )

        # Fixed key set for rewards — geneval group keys vary per batch and
        # would cause KeyError during collation across sampling batches.
        stable_reward_keys = reward_names + ["avg", "accuracy", "strict_accuracy"]

        for sample_item in tqdm(
            samples_data_list, desc="Waiting for rewards", disable=not is_main_process(rank), position=0
        ):
            rewards, reward_metadata = sample_item["rewards_future"].result()
            batch_size_local = len(next(iter(rewards.values())))
            sample_item["rewards"] = {}
            for rk in stable_reward_keys:
                if rk in rewards:
                    sample_item["rewards"][rk] = torch.as_tensor(rewards[rk], device=device).float()
                else:
                    sample_item["rewards"][rk] = torch.full((batch_size_local,), float("nan"), device=device)
            del sample_item["rewards_future"]

        # Collate samples
        collated_samples = {
            k: (
                torch.cat([s[k] for s in samples_data_list], dim=0)
                if not isinstance(samples_data_list[0][k], dict)
                else {sk: torch.cat([s[k][sk] for s in samples_data_list], dim=0) for sk in samples_data_list[0][k]}
            )
            for k in samples_data_list[0].keys()
        }

        # Accumulate all sampling-phase log data into single dict to avoid
        # wandb media duplication (multiple wandb.log at same step duplicates media).
        sampling_log = {}

        # Logging images (main process)
        if epoch % 10 == 0 and is_main_process(rank):
            images_to_log = images.cpu()  # from last sampling batch on this rank
            prompts_to_log = prompts  # from last sampling batch on this rank
            rewards_to_log = collated_samples["rewards"]["avg"][-len(images_to_log) :].cpu()
            num_to_log = min(15, len(images_to_log))
            wandb_images = []
            for idx in range(num_to_log):
                img_data = images_to_log[idx]
                pil = Image.fromarray((img_data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                wandb_images.append(
                    wandb.Image(pil, caption=f"{prompts_to_log[idx]:.100} | avg: {rewards_to_log[idx]:.2f}")
                )
            sampling_log["images"] = wandb_images
        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        )
        if pareto_enabled:
            for reward_name in reward_names:
                collated_samples["rewards"][reward_name] = (
                    collated_samples["rewards"][reward_name].unsqueeze(1).repeat(1, num_train_timesteps)
                )

        # Gather rewards across processes — use fixed key set to avoid
        # deadlocks when geneval group keys differ across ranks.
        train_gather_keys = reward_names + ["avg", "accuracy", "strict_accuracy"]
        gathered_rewards_dict = {}
        for key in train_gather_keys:
            if key in collated_samples["rewards"]:
                gathered_rewards_dict[key] = gather_tensor_to_all(
                    collated_samples["rewards"][key], world_size
                ).numpy()
            else:
                dummy = torch.full_like(collated_samples["rewards"]["avg"], float("nan"))
                gathered_rewards_dict[key] = gather_tensor_to_all(dummy, world_size).numpy()

        if is_main_process(rank):  # accumulate reward stats
            sampling_log["epoch"] = epoch
            sampling_log.update({
                f"reward_{k}": float(np.nanmean(v))
                for k, v in gathered_rewards_dict.items()
                if "_strict_accuracy" not in k and "_accuracy" not in k
            })

        # EMA reward normalization: scale each reward to mean ~0.5 for gradient computation
        if config.reward_ema_normalize:
            for rk in reward_names:
                raw = gathered_rewards_dict[rk]
                valid = raw[~np.isnan(raw)]
                if len(valid) > 0:
                    batch_mean = float(np.mean(valid))
                    if rk not in reward_ema_means:
                        reward_ema_means[rk] = batch_mean
                    else:
                        reward_ema_means[rk] = (
                            config.reward_ema_decay * reward_ema_means[rk]
                            + (1 - config.reward_ema_decay) * batch_mean
                        )
            # Normalize individual rewards → mean ~0.5
            for rk in reward_names:
                if rk in reward_ema_means and reward_ema_means[rk] > 1e-8:
                    gathered_rewards_dict[rk] = 0.5 * gathered_rewards_dict[rk] / reward_ema_means[rk]
            # Recompute avg from normalized rewards (NaN-aware weighted mean)
            reward_weights = config.reward_fn
            batch_shape = gathered_rewards_dict["avg"].shape
            new_avg = np.full(batch_shape[0], np.nan)
            for i in range(batch_shape[0]):
                w_sum, w_total = 0.0, 0.0
                for rk, w in reward_weights.items():
                    val = gathered_rewards_dict[rk][i, 0] if gathered_rewards_dict[rk].ndim == 2 else gathered_rewards_dict[rk][i]
                    if not np.isnan(val):
                        w_sum += w * val
                        w_total += w
                new_avg[i] = w_sum / w_total if w_total > 0 else np.nan
            if gathered_rewards_dict["avg"].ndim == 2:
                new_avg = np.broadcast_to(new_avg[:, None], batch_shape).copy()
            gathered_rewards_dict["avg"] = new_avg
            if is_main_process(rank):
                sampling_log.update({f"reward_ema_mean_{rk}": v for rk, v in reward_ema_means.items()})

        if config.per_prompt_stat_tracking:
            prompt_ids_all = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
            prompts_all_decoded = pipeline.tokenizer.batch_decode(
                prompt_ids_all.cpu().numpy(), skip_special_tokens=True
            )
            # Stat tracker update expects numpy arrays for rewards
            advantages = stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"])
            per_reward_advantages = {}
            if pareto_enabled:
                for reward_name, tracker in per_reward_stat_trackers.items():
                    per_reward_advantages[reward_name] = tracker.update(
                        prompts_all_decoded, gathered_rewards_dict[reward_name]
                    )

            if is_main_process(rank):
                group_size, trained_prompt_num = stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all_decoded, gathered_rewards_dict)
                log_payload = {
                    "group_size": group_size,
                    "trained_prompt_num": trained_prompt_num,
                    "zero_std_ratio": zero_std_ratio,
                    "reward_std_mean": reward_std_mean,
                    "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                    "mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                    "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                    "mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                    "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                }
                if pareto_enabled:
                    for reward_name, reward_advantages in per_reward_advantages.items():
                        log_payload[f"advantage_mean_{reward_name}"] = float(np.mean(reward_advantages))
                        log_payload[f"advantage_std_{reward_name}"] = float(np.std(reward_advantages))
                sampling_log.update(log_payload)
            stat_tracker.clear()
            for tracker in per_reward_stat_trackers.values():
                tracker.clear()
        else:
            avg_rewards_all = gathered_rewards_dict["avg"]
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (avg_rewards_all.std() + 1e-4)
            per_reward_advantages = {}
            if pareto_enabled:
                for reward_name in reward_names:
                    reward_values = gathered_rewards_dict[reward_name]
                    per_reward_advantages[reward_name] = (
                        reward_values - reward_values.mean()
                    ) / (reward_values.std() + 1e-4)
        # Flush all sampling-phase logs in a single wandb call
        if is_main_process(rank) and sampling_log:
            wandb_log_safe(sampling_log, step=global_step)

        # Distribute advantages back to processes
        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if advantages.shape[0] == world_size * samples_per_gpu:
            collated_samples["advantages"] = torch.from_numpy(
                advantages.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
        else:
            assert False

        if pareto_enabled:
            collated_samples["per_reward_advantages"] = {}
            for reward_name, reward_advantage in per_reward_advantages.items():
                if reward_advantage.ndim == 1:
                    reward_advantage = reward_advantage[:, None]
                collated_samples["per_reward_advantages"][reward_name] = torch.from_numpy(
                    reward_advantage.reshape(world_size, samples_per_gpu, -1)[rank]
                ).to(device)

        if is_main_process(rank):
            logger.info(f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}")

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]

        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size

        filtered_samples = collated_samples

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape

        # TRAINING
        transformer_ddp.train()  # Sets DDP model and its submodules to train mode.

        # Total number of backward passes before an optimizer step
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps

        current_accumulated_steps = 0  # Counter for backward passes
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = permute_batch_tensors(filtered_samples, perm)

            perms_time = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
            )
            for key in ["timesteps", "next_timesteps"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
                ]

            training_batch_size = total_batch_size_filtered // num_batches

            samples_batched_list = []
            for k_batch in range(num_batches):
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                samples_batched_list.append(slice_batch_tensors(shuffled_filtered_samples, start, end))

            info_accumulated = defaultdict(list)  # For accumulating stats over one grad acc cycle

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                current_micro_batch_size = len(train_sample_batch["prompt_embeds"])

                if config.sample.guidance_scale > 1.0:
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:current_micro_batch_size], train_sample_batch["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [
                            train_neg_pooled_prompt_embeds[:current_micro_batch_size],
                            train_sample_batch["pooled_prompt_embeds"],
                        ]
                    )
                else:
                    embeds = train_sample_batch["prompt_embeds"]
                    pooled_embeds = train_sample_batch["pooled_prompt_embeds"]

                # Loop over timesteps for this micro-batch
                for j_idx, j_timestep_orig_idx in tqdm(
                    enumerate(range(num_train_timesteps)),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not is_main_process(rank),
                ):
                    assert j_idx == j_timestep_orig_idx
                    x0 = train_sample_batch["latents_clean"]

                    t = train_sample_batch["timesteps"][:, j_idx] / 1000.0

                    t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))

                    noise = torch.randn_like(x0.float())

                    xt = (1 - t_expanded) * x0 + t_expanded * noise

                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            # prediction v
                            old_prediction = transformer_ddp(
                                hidden_states=xt,
                                timestep=train_sample_batch["timesteps"][:, j_idx],
                                encoder_hidden_states=embeds,
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                            )[0].detach()
                        transformer_ddp.module.set_adapter("default")

                        # prediction v
                        forward_prediction = transformer_ddp(
                            hidden_states=xt,
                            timestep=train_sample_batch["timesteps"][:, j_idx],
                            encoder_hidden_states=embeds,
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                        )[0]

                        with torch.no_grad():  # Reference model part
                            # For LoRA, disable adapter.
                            if config.use_lora:
                                with transformer_ddp.module.disable_adapter():
                                    ref_forward_prediction = transformer_ddp(
                                        hidden_states=xt,
                                        timestep=train_sample_batch["timesteps"][:, j_idx],
                                        encoder_hidden_states=embeds,
                                        pooled_projections=pooled_embeds,
                                        return_dict=False,
                                    )[0]
                                transformer_ddp.module.set_adapter("default")
                            else:  # Full model - this requires a frozen copy of the model
                                assert False
                    main_advantages_t = train_sample_batch["advantages"][:, j_idx]
                    # In mixed dataset mode, avg advantage may be NaN; clean for loss computation
                    main_advantages_t = torch.nan_to_num(main_advantages_t, nan=0.0)

                    # Determine training mode for this micro-step
                    use_cached_alpha = (
                        amortized_mgda_enabled
                        and cached_pareto_alphas is not None
                        and (
                            equal_alpha  # Fixed weights apply from step 0, including after resume.
                            or (
                                global_step >= amortize_warmup
                                and (global_step - last_full_mgda_step) < amortize_every_n
                            )
                        )
                    )
                    use_full_mgda = pareto_enabled and not use_cached_alpha

                    if use_cached_alpha:
                        # --- Amortized MGDA: single backward with cached-alpha-weighted advantages ---
                        # Decide reward dropout for amortized step
                        amort_dropout_set = set()
                        amort_non_pickscore_mask = None
                        if reward_dropout_targets and reward_dropout_max_prob > 0:
                            effective_dropout_prob = (
                                min(global_step / reward_dropout_warmup_steps, 1.0) * reward_dropout_max_prob
                                if reward_dropout_warmup_steps > 0
                                else reward_dropout_max_prob
                            )
                            amort_dropout_rng = torch.Generator(device='cpu')
                            amort_dropout_rng.manual_seed(global_step * 1000 + j_idx)
                            for rn in reward_names:
                                if rn in reward_dropout_targets:
                                    if torch.rand(1, generator=amort_dropout_rng).item() < effective_dropout_prob:
                                        amort_dropout_set.add(rn)
                            if amort_dropout_set:
                                amort_non_pickscore_mask = torch.zeros(
                                    train_sample_batch["latents_clean"].shape[0], dtype=torch.bool, device=device
                                )
                                for src_rn in reward_names:
                                    if src_rn not in reward_dropout_targets:
                                        src_adv = train_sample_batch["per_reward_advantages"][src_rn][:, j_idx]
                                        amort_non_pickscore_mask = amort_non_pickscore_mask | ~torch.isnan(src_adv)

                        combined_adv = torch.zeros_like(main_advantages_t)
                        for reward_name in reward_names:
                            reward_adv = train_sample_batch["per_reward_advantages"][reward_name][:, j_idx]
                            clean_adv = torch.nan_to_num(reward_adv, nan=0.0)
                            # Apply dropout: mask OCR/GenEval samples for dropped aesthetic rewards
                            if reward_name in amort_dropout_set and amort_non_pickscore_mask is not None:
                                clean_adv = clean_adv * (~amort_non_pickscore_mask).float()
                            mgda_name, split_factor = reward_to_mgda[reward_name]
                            alpha = cached_pareto_alphas[mgda_name] / split_factor
                            combined_adv = combined_adv + alpha * clean_adv

                        policy_loss, kl_div_loss, loss_terms = compute_policy_and_kl_terms(
                            x0=x0,
                            xt=xt,
                            t_expanded=t_expanded,
                            forward_prediction=forward_prediction,
                            old_prediction=old_prediction,
                            ref_forward_prediction=ref_forward_prediction,
                            advantages_t=combined_adv,
                            config=config,
                        )
                        loss = policy_loss + config.train.beta * kl_div_loss
                        loss_terms["total_loss"] = loss.detach()

                        scaled_loss = loss / effective_grad_accum_steps
                        if mixed_precision_dtype == torch.float16:
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        current_accumulated_steps += 1

                        # Log cached alpha info
                        loss_terms["amortized_step"] = torch.tensor(1.0, device=device)
                        loss_terms["alpha_staleness"] = torch.tensor(
                            float(global_step - last_full_mgda_step), device=device
                        )
                        for mgda_name in mgda_names:
                            loss_terms[f"pareto_alpha_{mgda_name}"] = torch.tensor(
                                cached_pareto_alphas[mgda_name], device=device
                            )

                    elif use_full_mgda:
                        # --- Full MGDA: K+1 backward passes + MGDA solver ---
                        policy_loss, kl_div_loss, loss_terms = compute_policy_and_kl_terms(
                            x0=x0,
                            xt=xt,
                            t_expanded=t_expanded,
                            forward_prediction=forward_prediction,
                            old_prediction=old_prediction,
                            ref_forward_prediction=ref_forward_prediction,
                            advantages_t=main_advantages_t,
                            config=config,
                        )

                        scale_value = scaler.get_scale() if mixed_precision_dtype == torch.float16 else 1.0
                        accumulated_grad, accumulated_had_non_finite = sanitize_grad_vector(
                            capture_grad_vector(transformer_trainable_parameters)
                        )
                        clear_parameter_grads(transformer_trainable_parameters)

                        reward_grad_vectors = []
                        reward_loss_terms = {}
                        reward_non_finite_flags = {}

                        # Decide reward dropout for this step (deterministic so all ranks agree)
                        reward_dropout_set = set()
                        if reward_dropout_targets and reward_dropout_max_prob > 0:
                            effective_dropout_prob = (
                                min(global_step / reward_dropout_warmup_steps, 1.0) * reward_dropout_max_prob
                                if reward_dropout_warmup_steps > 0
                                else reward_dropout_max_prob
                            )
                            dropout_rng = torch.Generator(device='cpu')
                            dropout_rng.manual_seed(global_step * 1000 + j_idx)
                            for rn in reward_names:
                                if rn in reward_dropout_targets:
                                    if torch.rand(1, generator=dropout_rng).item() < effective_dropout_prob:
                                        reward_dropout_set.add(rn)

                        # Pre-compute non-pickscore mask (samples from OCR/GenEval sources)
                        non_pickscore_mask = None
                        if reward_dropout_set:
                            non_pickscore_mask = torch.zeros(
                                train_sample_batch["latents_clean"].shape[0], dtype=torch.bool, device=device
                            )
                            for src_rn in reward_names:
                                if src_rn not in reward_dropout_targets:  # ocr, geneval
                                    src_adv = train_sample_batch["per_reward_advantages"][src_rn][:, j_idx]
                                    non_pickscore_mask = non_pickscore_mask | ~torch.isnan(src_adv)

                        sync_context = (
                            transformer_ddp.no_sync()
                            if world_size > 1 and config.pareto.ddp_sync
                            else nullcontext()
                        )

                        with sync_context:
                            for reward_name in reward_names:
                                reward_adv = train_sample_batch["per_reward_advantages"][reward_name][:, j_idx]
                                # Handle NaN advantages from mixed dataset (incompatible samples)
                                reward_valid_mask = ~torch.isnan(reward_adv)
                                clean_adv = torch.nan_to_num(reward_adv, nan=0.0)

                                # Reward dropout: mask out OCR/GenEval samples for dropped aesthetic rewards
                                if reward_name in reward_dropout_set and non_pickscore_mask is not None:
                                    reward_valid_mask = reward_valid_mask & ~non_pickscore_mask
                                    clean_adv = clean_adv * (~non_pickscore_mask).float()

                                reward_loss_mask = reward_valid_mask.float() if not reward_valid_mask.all() else None

                                reward_policy_loss, _, reward_terms = compute_policy_and_kl_terms(
                                    x0=x0,
                                    xt=xt,
                                    t_expanded=t_expanded,
                                    forward_prediction=forward_prediction,
                                    old_prediction=old_prediction,
                                    ref_forward_prediction=ref_forward_prediction,
                                    advantages_t=clean_adv,
                                    config=config,
                                    loss_mask=reward_loss_mask,
                                )
                                scaled_reward_loss = reward_policy_loss / effective_grad_accum_steps
                                if mixed_precision_dtype == torch.float16:
                                    scaler.scale(scaled_reward_loss).backward(retain_graph=True)
                                else:
                                    scaled_reward_loss.backward(retain_graph=True)

                                reward_grad_vector = capture_grad_vector(transformer_trainable_parameters)
                                if mixed_precision_dtype == torch.float16:
                                    reward_grad_vector = reward_grad_vector / scale_value
                                reward_grad_vector, reward_had_non_finite = sanitize_grad_vector(reward_grad_vector)
                                reward_grad_vectors.append(reward_grad_vector)
                                reward_loss_terms[reward_name] = reward_terms
                                reward_non_finite_flags[reward_name] = reward_had_non_finite
                                clear_parameter_grads(transformer_trainable_parameters)

                            scaled_kl_loss = (config.train.beta * kl_div_loss) / effective_grad_accum_steps
                            if mixed_precision_dtype == torch.float16:
                                scaler.scale(scaled_kl_loss).backward()
                            else:
                                scaled_kl_loss.backward()
                            kl_grad_vector = capture_grad_vector(transformer_trainable_parameters)
                            if mixed_precision_dtype == torch.float16:
                                kl_grad_vector = kl_grad_vector / scale_value
                            kl_grad_vector, kl_had_non_finite = sanitize_grad_vector(kl_grad_vector)
                            clear_parameter_grads(transformer_trainable_parameters)

                        if world_size > 1 and config.pareto.ddp_sync:
                            synced_reward_grads = sync_grad_vectors(reward_grad_vectors)
                            synced_kl_grad = sync_grad_vector(kl_grad_vector)
                        else:
                            synced_reward_grads = reward_grad_vectors
                            synced_kl_grad = kl_grad_vector

                        sanitized_reward_grads = []
                        for reward_name, synced_reward_grad in zip(reward_names, synced_reward_grads, strict=True):
                            sanitized_reward_grad, synced_reward_had_non_finite = sanitize_grad_vector(
                                synced_reward_grad
                            )
                            reward_non_finite_flags[reward_name] = (
                                reward_non_finite_flags[reward_name] or synced_reward_had_non_finite
                            )
                            sanitized_reward_grads.append(sanitized_reward_grad)
                        synced_reward_grads = sanitized_reward_grads

                        synced_kl_grad, synced_kl_had_non_finite = sanitize_grad_vector(synced_kl_grad)
                        kl_had_non_finite = kl_had_non_finite or synced_kl_had_non_finite

                        # Apply reward grouping before MGDA
                        if mgda_group_members:
                            mgda_grads = []
                            for mgda_name in mgda_names:
                                if mgda_name in mgda_group_members:
                                    member_indices = mgda_group_members[mgda_name]
                                    group_grad = torch.stack(
                                        [synced_reward_grads[i] for i in member_indices]
                                    ).mean(dim=0)
                                    mgda_grads.append(group_grad)
                                else:
                                    idx = reward_names.index(mgda_name)
                                    mgda_grads.append(synced_reward_grads[idx])
                        else:
                            mgda_grads = synced_reward_grads

                        balance_result = balance_reward_gradients(
                            mgda_grads,
                            normalize=config.pareto.normalize_grads,
                            eps=config.pareto.grad_eps,
                            weights=mgda_weights,
                            alpha_floor=alpha_floor,
                        )

                        # Cache alphas for amortized steps
                        if amortized_mgda_enabled:
                            new_alphas = {
                                name: balance_result["alphas"][idx].item()
                                for idx, name in enumerate(mgda_names)
                            }
                            if alpha_ema_decay > 0 and cached_pareto_alphas is not None:
                                cached_pareto_alphas = {
                                    name: alpha_ema_decay * cached_pareto_alphas[name]
                                    + (1 - alpha_ema_decay) * new_alphas[name]
                                    for name in mgda_names
                                }
                            else:
                                cached_pareto_alphas = new_alphas
                            last_full_mgda_step = global_step

                        current_step_unscaled_grad = balance_result["combined_grad"] + synced_kl_grad
                        current_step_unscaled_grad, current_step_had_non_finite = sanitize_grad_vector(
                            current_step_unscaled_grad
                        )
                        current_step_scaled_grad = (
                            current_step_unscaled_grad * scale_value
                            if mixed_precision_dtype == torch.float16
                            else current_step_unscaled_grad
                        )
                        final_grad = accumulated_grad + current_step_scaled_grad
                        final_grad, final_grad_had_non_finite = sanitize_grad_vector(final_grad)
                        restore_grad_vector(transformer_trainable_parameters, final_grad)
                        current_accumulated_steps += 1

                        loss_terms["total_loss"] = (policy_loss + config.train.beta * kl_div_loss).detach()
                        loss_terms["amortized_step"] = torch.tensor(0.0, device=device)
                        loss_terms["pareto_combined_grad_norm"] = balance_result["combined_grad"].norm().detach()
                        loss_terms["pareto_kl_grad_norm"] = synced_kl_grad.norm().detach()
                        loss_terms["pareto_final_grad_norm"] = (
                            final_grad.norm() / (scale_value if mixed_precision_dtype == torch.float16 else 1.0)
                        ).detach()
                        loss_terms["pareto_fallback_used"] = torch.tensor(
                            0.0 if balance_result["fallback_reason"] == "none" else 1.0,
                            device=device,
                        )
                        loss_terms["pareto_rescale_factor"] = balance_result["rescale_factor"].detach()
                        loss_terms["pareto_nonfinite_accumulated_grad"] = torch.tensor(
                            float(accumulated_had_non_finite),
                            device=device,
                        )
                        loss_terms["pareto_nonfinite_kl_grad"] = torch.tensor(
                            float(kl_had_non_finite),
                            device=device,
                        )
                        loss_terms["pareto_nonfinite_current_step_grad"] = torch.tensor(
                            float(current_step_had_non_finite),
                            device=device,
                        )
                        loss_terms["pareto_nonfinite_final_grad"] = torch.tensor(
                            float(final_grad_had_non_finite),
                            device=device,
                        )
                        # Log MGDA-level metrics (per group or per reward)
                        for mgda_index, mgda_name in enumerate(mgda_names):
                            loss_terms[f"pareto_raw_grad_norm_{mgda_name}"] = (
                                balance_result["raw_norms"][mgda_index]
                            ).detach()
                            loss_terms[f"pareto_alpha_{mgda_name}"] = balance_result["alphas"][mgda_index].detach()
                            loss_terms[f"pareto_valid_{mgda_name}"] = (
                                balance_result["valid_mask"][mgda_index].float().detach()
                            )
                        # Log per-reward metrics (always per individual reward)
                        for reward_name in reward_names:
                            loss_terms[f"pareto_nonfinite_reward_grad_{reward_name}"] = torch.tensor(
                                float(reward_non_finite_flags[reward_name]),
                                device=device,
                            )
                            loss_terms[f"policy_loss_{reward_name}"] = reward_loss_terms[reward_name]["policy_loss"]
                        # Log reward dropout info
                        if reward_dropout_targets and reward_dropout_max_prob > 0:
                            loss_terms["reward_dropout_count"] = torch.tensor(
                                float(len(reward_dropout_set)), device=device
                            )
                            loss_terms["reward_dropout_prob"] = torch.tensor(
                                effective_dropout_prob, device=device
                            )
                    else:
                        # --- Baseline: single backward with avg advantages ---
                        policy_loss, kl_div_loss, loss_terms = compute_policy_and_kl_terms(
                            x0=x0,
                            xt=xt,
                            t_expanded=t_expanded,
                            forward_prediction=forward_prediction,
                            old_prediction=old_prediction,
                            ref_forward_prediction=ref_forward_prediction,
                            advantages_t=main_advantages_t,
                            config=config,
                        )
                        loss = policy_loss + config.train.beta * kl_div_loss
                        loss_terms["total_loss"] = loss.detach()

                        scaled_loss = loss / effective_grad_accum_steps
                        if mixed_precision_dtype == torch.float16:
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        current_accumulated_steps += 1

                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(transformer_ddp.module.parameters(), config.train.max_grad_norm)
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()
                        if lr_scheduler is not None:
                            lr_scheduler.step()

                        log_info = {k: torch.mean(torch.stack(v_list)).item() for k, v_list in info_accumulated.items()}
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced_log_info = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}
                        if is_main_process(rank):
                            lr_log = {"lr": optimizer.param_groups[0]["lr"]} if lr_scheduler is not None else {}
                            ema_log = {"ema_decay": ema.get_current_decay(global_step)} if ema is not None else {}
                            wandb_log_safe(
                                {
                                    "step": global_step,
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                                    **lr_log,
                                    **ema_log,
                                },
                                step=global_step,
                            )

                        global_step += 1  # gradient step
                        info_accumulated = defaultdict(list)  # Reset for next accumulation cycle

                if (
                    config.train.ema
                    and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier()

        with torch.no_grad():
            decay_step_offset = getattr(config, "decay_step_offset", 0)
            decay = return_decay(
                max(global_step - decay_step_offset, 0),
                config.decay_type,
                getattr(config, "decay_eta_max", None),
                getattr(config, "decay_uprate", None),
            )
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
