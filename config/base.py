import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = ""
    config.debug = False

    # random seed for reproducibility.
    config.seed = 42
    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those
    # samples.
    config.num_epochs = 100000
    # number of epochs between saving model checkpoints.
    config.save_freq = 30
    config.eval_freq = 10
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision = "fp16"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.resume_from = ""
    # auto-resume: find latest checkpoint in save_dir and resume wandb run
    config.auto_resume = False
    # whether or not to use LoRA.
    config.use_lora = True
    config.dataset = ""
    config.resolution = 768

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = ""
    # revision of the model to load.
    pretrained.revision = ""

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample.num_steps = 40
    sample.eval_num_steps = 40
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 4.5
    # batch size (per GPU!) to use for sampling.
    sample.train_batch_size = 1
    sample.num_image_per_prompt = 1
    sample.test_batch_size = 1
    # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch *
    # batch_size * num_gpus`.
    sample.num_batches_per_epoch = 2
    # Whether use all samples in a batch to compute std
    sample.global_std = True
    # noise level
    sample.noise_level = 1.0

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU!) to use for training.
    train.batch_size = 1
    # learning rate.
    train.learning_rate = 3e-4
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8
    # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
    # gradient_accumulation_steps`.
    train.gradient_accumulation_steps = 1
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0
    # number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one
    # outer epoch's round of sampling.
    train.num_inner_epochs = 1
    # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.adv_clip_max = 5
    # the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the
    # timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.
    train.timestep_fraction = 0.99
    # kl ratio
    train.beta = 0.0001
    # pretrained lora path
    train.lora_path = None
    train.ema = True
    train.ema_decay = 0.9
    # cosine LR decay: minimum LR as a fraction of initial LR (0 = no decay)
    train.lr_min_ratio = 0.0
    # total training steps for cosine schedule (if 0, uses config.num_epochs)
    train.lr_total_steps = 0

    ###### Prompt Function ######
    # prompt function to use. see `prompts.py` for available prompt functions.
    config.prompt_fn = ""
    # kwargs to pass to the prompt function.
    config.prompt_fn_kwargs = {}

    ###### Reward Function ######
    # reward function to use. see `rewards.py` for available reward functions.
    config.reward_fn = ml_collections.ConfigDict()
    ###### Pareto Gradient Balancing ######
    config.pareto = pareto = ml_collections.ConfigDict()
    pareto.enabled = False
    pareto.method = "mgda"
    pareto.normalize_grads = True
    pareto.ddp_sync = True
    pareto.grad_eps = 1e-12
    pareto.amortized = False
    pareto.amortize_every_n = 10
    pareto.amortize_warmup = 1
    pareto.reward_groups = {}  # e.g. {"aesthetic": ["pickscore", "hpsv2", "clipscore"]}
    pareto.reward_weights = {}  # e.g. {"aesthetic": 1.0, "ocr": 2.0, "geneval": 2.0}
    pareto.alpha_floor = 0.0  # minimum alpha for each reward in MGDA QP (0 = no floor)
    pareto.alpha_ema_decay = 0.0  # EMA decay for cached alphas (0 = no smoothing, use raw MGDA solve)
    pareto.reward_dropout_targets = []  # rewards to randomly dropout before MGDA
    pareto.reward_dropout_prob = 0.0  # per-reward independent dropout probability (also serves as max_prob when warmup > 0)
    pareto.reward_dropout_warmup_steps = 0  # linear ramp: 0 = constant prob, >0 = ramp from 0 to dropout_prob over N steps
    config.save_dir = ""

    ###### Per-Prompt Stat Tracking ######
    config.per_prompt_stat_tracking = True
    config.reward_ema_normalize = False
    config.reward_ema_decay = 0.99
    config.r_normalize_mean = False  # Normalize r coefficient to batch mean 0.5
    config.stratified_sampling = False  # Balanced sampling across dataset sources

    return config
