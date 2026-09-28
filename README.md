# MARBLE

Multi-reward fine-tuning of Stable Diffusion 3.5 Medium, built on [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT).

The code has been organized with assistance from Codex and Claude.

## Environment setup

From the repository root, activate a Python 3.10 environment and install the dependencies:

```bash
conda create -n marble python=3.10 -y
conda activate marble
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
```

Prepare the model, reward dependencies, and datasets before launching. The launcher uses the active environment and do not install system packages.

## Reward Preparation

### Checkpoints Downloading

```bash
mkdir reward_ckpts
cd reward_ckpts
# Aesthetic
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/refs/heads/main/sac+logos+ava1-l14-linearMSE.pth
# GenEval
wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth
# ClipScore
wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin
# HPSv2.1
wget https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt
cd ..
```

### Reward Environments

```bash
# GenEval
pip install -U openmim
mim install mmengine
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv; git checkout 1.x
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e . -v
cd ..

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -e . -v
cd ..

pip install open-clip-torch clip-benchmark

# OCR
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein

# HPSv2.1
pip install hpsv2x==1.2.0

# ImageReward
pip install image-reward
pip install git+https://github.com/openai/CLIP.git
```

For `UnifiedReward`, we deploy the reward service using sglang. To avoid conflicts, first create a new environment and install sglang with:

```bash
pip install "sglang[all]"
```

Then launch the service with:

```bash
python -m sglang.launch_server --model-path CodeGoat24/UnifiedReward-7b-v1.5 --api-key flowgrpo --port 17140 --chat-template chatml-llava --enable-p2p-check --mem-fraction-static 0.85
```

Memory usage can be reduced by lowering `--mem-fraction-static`, limiting `--max-running-requests`, and increasing `--data-parallel-size` or `--tensor-parallel-size`.

## Training

`scripts/` contains only `train.sh`. It selects `rho_07` (the default) or `equal_alpha` and starts the root-level `train.py`. The standalone evaluation entry point is `evaluation.py`.

The two nodes must have the same code, dependencies, and dataset paths. Make the SD3.5 Medium weights available on both nodes. Use shared storage for the output directory when resuming across nodes.

Dataset layout:

```text
dataset/
  pickscore/{train,test}.txt
  ocr/{train,test}.txt
  geneval/{train,test}_metadata.jsonl
```

See [mixed dataset details](docs/MIXED_DATASET.md) for reward applicability and masking.

For **ρ = 0.7**, run the following commands on the respective nodes, replacing `HEAD_NODE_IP` with node 0's reachable address:

```bash
# Node 0
MASTER_ADDR=HEAD_NODE_IP NODE_RANK=0 bash scripts/train.sh rho_07

# Node 1
MASTER_ADDR=HEAD_NODE_IP NODE_RANK=1 bash scripts/train.sh rho_07
```

For **equal alpha**, select `equal_alpha` on both nodes:

```bash
# Node 0
MASTER_ADDR=HEAD_NODE_IP NODE_RANK=0 bash scripts/train.sh equal_alpha

# Node 1
MASTER_ADDR=HEAD_NODE_IP NODE_RANK=1 bash scripts/train.sh equal_alpha
```

## Evaluation

We provide an inference script for loading LoRA checkpoints and running evaluation.

```bash
# Hugging Face LoRA checkpoint, w/ CFG
torchrun --nproc_per_node=8 evaluation.py \
    --lora_hf_path "jieliu/SD3.5M-FlowGRPO-GenEval" \
    --model_type sd3 \
    --dataset geneval \
    --guidance_scale 4.5 \
    --mixed_precision fp16 \
    --save_images

# Local LoRA checkpoint, w/o CFG
torchrun --nproc_per_node=8 evaluation.py \
    --checkpoint_path "logs/nft/sd3/multi_reward_pareto_5r_alpha_ema_v2/checkpoints/checkpoint-STEP" \
    --model_type sd3 \
    --dataset geneval \
    --guidance_scale 1.0 \
    --mixed_precision fp16 \
    --save_images
```

The `--dataset` flag supports `geneval`, `ocr`, `pickscore`, and `drawbench`.


## Acknowledgements and license

This implementation builds on DiffusionNFT. Original copyright notices and the [Apache-2.0 license](LICENSE) are retained.

```bibtex
@article{zheng2025diffusionnft,
  title={DiffusionNFT: Online Diffusion Reinforcement with Forward Process},
  author={Zheng, Kaiwen and Chen, Huayu and Ye, Haotian and Wang, Haoxiang and Zhang, Qinsheng and Jiang, Kai and Su, Hang and Ermon, Stefano and Zhu, Jun and Liu, Ming-Yu},
  journal={arXiv preprint arXiv:2509.16117},
  year={2025}
}
```

If you find MARBLE useful in your research, please cite:

```bibtex
@article{zhao2026marblemultiaspectrewardbalance,
  title={MARBLE: Multi-Aspect Reward Balance for Diffusion RL},
  author={Canyu Zhao and Hao Chen and Yunze Tong and Yu Qiao and Jiacheng Li and Chunhua Shen},
  year={2026},
  eprint={2605.06507},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.06507},
}
```
