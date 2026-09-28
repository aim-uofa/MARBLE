"""MARBLE release configurations: rho=0.7 and equal alpha, both on 16 GPUs."""

import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "marble_base", os.path.join(os.path.dirname(__file__), "base.py")
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def get_config(name):
    return {
        "sd3_multi_reward_pareto_5r_alpha_ema": sd3_multi_reward_pareto_5r_alpha_ema,
        "sd3_ablation_equal_alpha": sd3_ablation_equal_alpha,
    }[name]()


def sd3_multi_reward_pareto_5r_alpha_ema():
    """Five rewards, MGDA every 10 steps, alpha EMA rho=0.7, 16 GPUs."""
    config = base.get_config()
    config.base_model = "sd3"
    config.dataset = os.path.join(os.getcwd(), "dataset")
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.resolution = 512

    config.sample.num_steps = 25
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1.0
    config.sample.noise_level = 0.7
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    # 48 prompts x 24 images = 16 GPUs x 9 images x 8 batches.
    config.sample.num_image_per_prompt = 24
    config.sample.train_batch_size = 9
    config.sample.num_batches_per_epoch = 8
    config.sample.test_batch_size = 16
    config.train.batch_size = 9
    config.train.gradient_accumulation_steps = 8
    config.train.beta = 0.0001
    config.train.adv_mode = "all"

    config.prompt_fn = "mixed"
    config.stratified_sampling = True
    config.reward_fn = {
        "pickscore": 1.0,
        "hpsv2": 1.0,
        "clipscore": 1.0,
        "ocr": 1.0,
        "geneval": 1.0,
    }
    config.beta = 0.1
    config.decay_type = 1
    config.pareto.enabled = True
    config.pareto.method = "mgda"
    config.pareto.normalize_grads = True
    config.pareto.ddp_sync = True
    config.pareto.grad_eps = 1e-12
    config.pareto.amortized = True
    config.pareto.amortize_every_n = 10
    config.pareto.amortize_warmup = 1
    config.pareto.alpha_ema_decay = 0.7

    config.run_name = "nft_sd3_multi_reward_pareto_5r_alpha_ema_v2"
    config.save_dir = "logs/nft/sd3/multi_reward_pareto_5r_alpha_ema_v2"
    config.auto_resume = True
    return config


def sd3_ablation_equal_alpha():
    """Same 16-GPU recipe, with fixed alpha=0.2 per reward and no MGDA solve."""
    config = sd3_multi_reward_pareto_5r_alpha_ema()
    config.run_name = "ablation_equal_alpha"
    config.save_dir = "logs/nft/sd3/ablation/equal_alpha"
    config.pareto.equal_alpha = True
    return config
