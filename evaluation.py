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

import argparse
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusion3Pipeline
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model

from flow_grpo.rewards import multi_score

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict
from peft import PeftModel

import logging

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def setup_distributed(rank, world_size):
    """Initializes the distributed process group."""
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Destroys the distributed process group."""
    dist.destroy_process_group()


def is_main_process(rank):
    """Checks if the current process is the main one (rank 0)."""
    return rank == 0


class TextPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test"):
        self.file_path = os.path.join(dataset_path, f"{split}.txt")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}, "original_index": idx}


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test"):
        self.file_path = os.path.join(dataset_path, f"{split}_metadata.jsonl")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx], "original_index": idx}


def collate_fn(examples):
    prompts = [example["prompt"] for example in examples]
    metadatas = [example["metadata"] for example in examples]
    indices = [example["original_index"] for example in examples]
    return prompts, metadatas, indices


def main(args):
    # --- Distributed Setup ---
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if args.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None

    if is_main_process(rank):
        print(f"Running distributed evaluation with {world_size} GPUs.")
        if enable_amp:
            print(f"Using mixed precision: {args.mixed_precision}")
        os.makedirs(args.output_dir, exist_ok=True)
        if args.save_images:
            os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    results_filepath = os.path.join(args.output_dir, "evaluation_results.jsonl")

    if args.model_type != "sd3":
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # --- Load Model and Pipeline (skipped in --skip_generation mode) ---
    pipeline = None
    transformer_lora_config = None
    if args.skip_generation:
        if is_main_process(rank):
            print("--skip_generation: skipping pipeline + LoRA load; "
                  "will read images from disk in args.output_dir/images/.")
    else:
        pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
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

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not args.skip_generation:
        if args.lora_hf_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, args.lora_hf_path)
            pipeline.transformer = pipeline.transformer.merge_and_unload()
        elif args.checkpoint_path:
            lora_path = os.path.join(args.checkpoint_path, "lora")
            if is_main_process(rank):
                print(f"Loading LoRA weights from: {lora_path}")
            if not os.path.exists(lora_path):
                raise FileNotFoundError(
                    f"LoRA directory not found at {lora_path}. Ensure your checkpoint has a 'lora' subdirectory."
                )

            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
            pipeline.transformer.load_adapter(lora_path, adapter_name="default", is_trainable=False)

        pipeline.transformer.eval()
        text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

        pipeline.transformer.to(device, dtype=text_encoder_dtype)
        pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
        pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
        pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
        pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

        pipeline.safety_checker = None
        pipeline.set_progress_bar_config(
            position=1,
            disable=not is_main_process(rank),
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

    # --- Load Dataset with Distributed Sampler ---
    dataset_path = f"dataset/{args.dataset}"
    if is_main_process(rank):
        print(f"Loading dataset from: {dataset_path}")

    if args.dataset == "geneval":
        dataset = GenevalPromptDataset(dataset_path, split="test")
        all_reward_scorers = {"geneval": 1.0}
        eval_batch_size = 14
    elif args.dataset == "ocr":
        dataset = TextPromptDataset(dataset_path, split="test")
        all_reward_scorers = {"ocr": 1.0}
        eval_batch_size = 16
    elif args.dataset == "pickscore":
        dataset = TextPromptDataset(dataset_path, split="test")
        all_reward_scorers = {
            "imagereward": 1.0,
            "pickscore": 1.0,
            "aesthetic": 1.0,
            "clipscore": 1.0,
            "hpsv2": 1.0,
        }
        eval_batch_size = 16
    elif args.dataset == "drawbench":
        dataset = TextPromptDataset(dataset_path, split="test")
        all_reward_scorers = {
            "imagereward": 1.0,
            "pickscore": 1.0,
            "aesthetic": 1.0,
            "clipscore": 1.0,
            "hpsv2": 1.0,
            "unifiedreward": 1.0
        }
        eval_batch_size = 5

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # --- Instantiate Reward Models (skipped in --no_score mode) ---
    scoring_fn = None
    if not args.no_score:
        if is_main_process(rank):
            print("Initializing reward models...")
        scoring_fn = multi_score(device, all_reward_scorers)
    elif is_main_process(rank):
        print("--no_score: skipping reward model loading; will only generate images.")

    # --- Evaluation Loop ---
    results_this_rank = []

    images_dir = os.path.join(args.output_dir, "images")
    for batch in tqdm(dataloader, desc=f"Evaluating (Rank {rank})", disable=not is_main_process(rank)):
        prompts, metadata, indices = batch
        current_batch_size = len(prompts)

        # Copy prompts before passing to pipeline — SD3 pipeline may mutate
        # the input list during tokenization, corrupting prompts for scoring.
        prompts_original = list(prompts)

        if args.skip_generation:
            # Load already-generated images from disk; produce the same
            # (N, C, H, W) float-in-[0,1] tensor the pipeline would have.
            np_imgs = np.stack([
                np.array(
                    Image.open(os.path.join(images_dir, f"{idx:05d}.jpg")).convert("RGB"),
                    dtype=np.uint8,
                )
                for idx in indices
            ])  # (N, H, W, C) uint8
            images = (
                torch.from_numpy(np_imgs).permute(0, 3, 1, 2).float().to(device) / 255.0
            )
        else:
            with torch.cuda.amp.autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images = pipeline(
                        prompts,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        output_type="pt",
                        height=args.resolution,
                        width=args.resolution,
                    )[0]

        if scoring_fn is not None:
            all_scores, _ = scoring_fn(images, prompts_original, metadata, only_strict=False)
        else:
            all_scores = {}

        for i in range(current_batch_size):
            sample_idx = indices[i]
            result_item = {
                "sample_id": sample_idx,
                "prompt": prompts_original[i],
                "metadata": metadata[i] if metadata else {},
                "scores": {},
            }

            if args.save_images and not args.skip_generation:
                image_path = os.path.join(args.output_dir, "images", f"{sample_idx:05d}.jpg")
                pil_image = Image.fromarray((images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil_image.save(image_path)
                result_item["image_path"] = image_path
            elif args.skip_generation:
                # Reference the existing on-disk image without rewriting.
                result_item["image_path"] = os.path.join(images_dir, f"{sample_idx:05d}.jpg")

            for score_name, score_values in all_scores.items():
                if isinstance(score_values, torch.Tensor):
                    result_item["scores"][score_name] = score_values[i].detach().cpu().item()
                else:
                    result_item["scores"][score_name] = float(score_values[i])

            results_this_rank.append(result_item)

        del images, all_scores
        torch.cuda.empty_cache()

    # --- Gather and Save Results ---
    dist.barrier()

    all_gathered_results = [None] * world_size
    dist.all_gather_object(all_gathered_results, results_this_rank)

    if is_main_process(rank):
        flat_results = [item for sublist in all_gathered_results for item in sublist]

        flat_results.sort(key=lambda x: x["sample_id"])

        with open(results_filepath, "w") as f_out:
            for result_item in flat_results:
                f_out.write(json.dumps(result_item) + "\n")

        print(f"\nEvaluation finished. All {len(flat_results)} results saved to {results_filepath}")

        # Save image-to-prompt mapping
        image_prompts = {}
        for result in flat_results:
            image_file = f"{result['sample_id']:05d}.jpg"
            image_prompts[image_file] = {
                "prompt": result["prompt"],
                "scores": result["scores"],
            }
        mapping_filepath = os.path.join(args.output_dir, "image_prompts.json")
        with open(mapping_filepath, "w") as f_map:
            json.dump(image_prompts, f_map, indent=2, ensure_ascii=False)
        print(f"Image-prompt mapping saved to {mapping_filepath}")

        # In --no_score mode there are no scores to aggregate. Skip the
        # average_scores.json write entirely so the file is only ever
        # produced by the actual scorer (avoids the empty-{} sentinel
        # that would falsely trigger downstream skip logic).
        if args.no_score:
            print("\n--no_score: skipping average_scores aggregation. "
                  "Run evaluation.py with --skip_generation and the same --output_dir to compute and save scores.")
        else:
            all_scores_agg = defaultdict(list)

            for result in flat_results:
                for score_name, score_value in result["scores"].items():
                    if isinstance(score_value, (int, float)):
                        all_scores_agg[score_name].append(score_value)

            average_scores = {
                name: np.mean(list(filter(lambda score: score != -10.0, scores))) for name, scores in all_scores_agg.items()
            }

            print("\n--- Average Scores ---")
            if not average_scores:
                print("No scores were found to average.")
            else:
                for name, avg_score in sorted(average_scores.items()):
                    print(f"{name:<20}: {avg_score:.4f}")
            print("----------------------")

            avg_scores_filepath = os.path.join(args.output_dir, "average_scores.json")
            with open(avg_scores_filepath, "w") as f_avg:
                json.dump(average_scores, f_avg, indent=4)
            print(f"Average scores also saved to {avg_scores_filepath}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained diffusion model in a distributed manner.")
    parser.add_argument(
        "--lora_hf_path",
        type=str,
        default="",
        help="Huggingface path for LoRA.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Local path to the LoRA checkpoint directory (e.g., './save/run_name/checkpoints/checkpoint-5000').",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["sd3"],
        help="Type of the base model ('sd3').",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["geneval", "ocr", "pickscore", "drawbench"], help="Dataset type."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_output",
        help="Directory to save evaluation results and generated images.",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=40, help="Number of inference steps for the diffusion pipeline."
    )
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution of the generated images.")
    parser.add_argument(
        "--save_images", action="store_true", help="Include this flag to save generated images to the output directory."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between 'no', 'fp16', or 'bf16'.",
    )
    parser.add_argument(
        "--no_score",
        action="store_true",
        help="Skip reward model loading and scoring; only generate + save images and prompt mapping.",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip pipeline + LoRA load and image generation; instead read pre-generated images from "
             "<output_dir>/images/<sample_id:05d>.jpg and feed them through the existing scoring path. "
             "Pairs with a prior --no_score run.",
    )

    args = parser.parse_args()
    main(args)
