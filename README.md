<h1 align="center">MARBLE: Multi-Aspect Reward BaLancE for Diffusion RL</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.06507"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white"></a>
  <a href="https://aim-uofa.github.io/MARBLE/"><img src="https://img.shields.io/badge/Project_Page-aim--uofa.github.io-2ea44f?logo=github"></a>
  <a href="#"><img src="https://img.shields.io/badge/Checkpoints-HuggingFace-ffd21e?logo=huggingface&logoColor=black"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</p>

<p align="center">
Canyu Zhao<sup>1</sup>,&nbsp;
Hao Chen<sup>1</sup>,&nbsp;
Yunze Tong<sup>1</sup>,&nbsp;
Yu Qiao<sup>2</sup>,&nbsp;
Jiacheng Li<sup>2</sup>,&nbsp;
Chunhua Shen<sup>1,3,✉</sup>
</p>

<p align="center">
<sup>1</sup>Zhejiang University &nbsp; <sup>2</sup>Hithink &nbsp; <sup>3</sup>Zhejiang University of Technology
</p>

<p align="center">
  <img src="https://aim-uofa.github.io/MARBLE/assets/figures/teaser.png" width="92%" alt="MARBLE teaser"/>
</p>

---

## TL;DR

**MARBLE** is a multi-reward RL fine-tuning framework for diffusion models that **preserves per-reward gradient information** end-to-end and combines them in **gradient space**. Built on [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT), MARBLE **simultaneously** improves different dimensions within a single model, without sequential training.

> **Why scalar reward aggregation fails.** Specialist rollouts — prompts that only carry signal for a subset of rewards (a "cat on sofa" image carries no OCR signal) — get their reward-specific advantage **diluted by averaging**. Weighted-sum and sequential fine-tuning systematically trade specialist rewards for general ones. MARBLE solves this *in gradient space*, not by reward engineering.

---

## Status

> 🚧  **Code release in progress.** Inference + checkpoints first, training code later. ⭐ the repo to get notified.

- [ ] Inference + 🤗 LoRA checkpoints
- [ ] Training code, configs, multi-node scripts
- [ ] MARBLE for Video Model

---

## Highlights

| | |
|---|---|
| 🎯 **One model, all rewards at once.** | No sequential fine-tuning — joint training across all reward dimensions. |
| 🧩 **Specialist rewards survive.** | OCR / GenEval no longer collapse under joint training. |
| ⚡ **~1× single-reward cost.** | Amortized MGDA + EMA-smoothed $\alpha$. |
| 📈 **Composite +1.12.** | Best on every reward axis vs. all baselines (Table 1). |

See the [project page](https://aim-uofa.github.io/MARBLE/) for the full results table, training-dynamics curves, EMA-decay $\rho$ ablation, qualitative comparisons and human-evaluation results.

---

## Citation

If you find MARBLE useful, please cite:

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

---

## Acknowledgement

MARBLE builds on [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT). We thank the authors for their excellent open-source releases.
