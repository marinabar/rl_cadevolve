# RL Fine-Tuning for CADEvolve

<img width="1831" height="384" alt="CADEvolve Gallery_page-0001" src="https://github.com/user-attachments/assets/924881ef-4f7d-4d7c-ac71-01b983d1727d" />



This repository contains the **GRPO-based reinforcement learning fine-tuning** code used in the paper:

> **CADEvolve: Creating Realistic CAD via Program Evolution**  
> Maksim Elistratov, Marina Barannikov, Gregory Ivanov, Valentin Khrulkov, Anton Konushin, Andrey Kuznetsov, Dmitrii Zhemchuzhnikov  
> arXiv:2602.16317 · [arXiv](https://arxiv.org/abs/2602.16317)

The pipeline fine-tunes a **Qwen2-VL** vision-language model to generate executable [CadQuery](https://cadquery.readthedocs.io/) Python programs from multi-view renders of 3-D shapes, using geometry-aware rewards (IoU, Chamfer Distance, normal AUC).

<p align="center">
  <img width="300" alt="CADEvolve-M-1" src="https://github.com/user-attachments/assets/2003687a-c75c-4407-8fba-d8d38fa9b60f" />
</p>

---

## Overview

```
grpo_cadevolve.py      # Main RL training script (dataset prep, reward fn, trainer setup)
grpo_loss.py           # Custom GRPO loss: Top-K advantage selection (CPPO variant)
cadevolve_dataset.py   # Dataset class + multi-view rendering pipeline (STLImageToCode)
utils_async.py         # Async worker pool for CadQuery execution & metric computation
normal_metrics.py      # Normal consistency (AUC) metric for surface quality
config.yaml            # GRPO hyperparameters (learning rate, KL coeff, etc.)
```

---

## Method

The RL loop is built on top of [TRL's GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) with two key modifications:

### 1. Geometry Reward
Each model completion (a CadQuery Python script) is executed in an isolated subprocess pool. The resulting mesh is compared against the ground-truth STL using three metrics:

| Metric | Description |
|--------|-------------|
| **IoU** | Volumetric intersection-over-union |
| **CD** | Chamfer Distance (point cloud, 8192 pts) |
| **Normal AUC** | Surface normal consistency AUC |

The default reward is `r = 10 × IoU`, clipped to `[-10, 10]`. Scripts that fail to execute or produce empty meshes receive a configurable `failure_reward` (default `-10`).

### 2. Top-K Advantage Selection (CPPO)
Instead of updating on all `G` generated completions per prompt, `TopSampleGRPOTrainer` selects only the `top_k=4` completions by **absolute advantage**. This focuses gradient updates on the most informative generations, reducing variance.

The clipped PPO objective follows the standard GRPO formulation:

$$
\mathcal{L}
= -\mathbb{E}\left[
\min\left(
\rho_t A_t,\;
\mathrm{clip}\left(\rho_t, 1-\varepsilon_{\mathrm{low}}, 1+\varepsilon_{\mathrm{high}}\right) A_t
\right)
\right],
$$

where

$$
\rho_t = \exp\left(\log \pi_\theta - \log \pi_{\mathrm{old}}\right)
$$

is the per-token probability ratio.<img width="680" height="506" alt="CADEvolve-M-1" src="https://github.com/user-attachments/assets/a0ab1bce-28cb-45f2-a474-aef453330d9a" />


---

## Requirements

```bash
pip install torch transformers trl datasets
pip install cadquery open3d trimesh qwen-vl-utils
pip install flash-attn --no-build-isolation   # optional but recommended
```

**Python:** 3.10+  
**GPU:** Multi-GPU training via `accelerate` / DeepSpeed is supported.

---

## Data

Training uses `.stl` mesh files organised into a flat directory. The `STLImageToCode` dataset class renders each mesh into a **7-view montage** (front, back, left, right, top, bottom, isometric) using Open3D, and pairs it with a fixed instruction prompt:

> *"Generate CadQuery v2 code for this 3-D shape. Return only Python code that assigns the final solid to variable `result`."*

Set the paths in `grpo_cadevolve.py` (or override via CLI):

```python
STLS_ROOT  = Path("/path/to/your/stl_files")
TRAIN_SPLIT = Path("/path/to/train_list.txt")
VAL_SPLIT   = Path("/path/to/val_list.txt")
```

Split files are plain text with one relative STL path per line.

---

## Training

### 1. Configure hyperparameters

Edit `config.yaml`:

```yaml
# Example config.yaml
output_dir: my_run
num_train_epochs: 3
per_device_train_batch_size: 4
num_generations: 16
learning_rate: 5e-7
kl_coef: 0.01
epsilon_high: 0.2
epsilon_low: 0.2
loss_type: grpo            # or "dr_grpo"
importance_sampling_level: token
```

### 2. Configure reward args (CLI or config)

| Argument | Default | Description |
|----------|---------|-------------|
| `--failure_reward` | `-10.0` | Reward for scripts that fail to execute |
| `--iou_coef` | `10.0` | Multiplier for IoU in reward |
| `--cd_coef` | `0.0` | Multiplier for CD in reward |
| `--auc_coef` | `0.0` | Multiplier for normal AUC in reward |
| `--pool_size` | `16` | Number of async CadQuery worker processes |
| `--gen_sample_steps` | `25` | Log a sample completion every N steps |

### 3. Launch

```bash
# Single node, multi-GPU with accelerate
accelerate launch --config_file accelerate_config.yaml \
    grpo_cadevolve.py \
    --config config.yaml \
    --sft_path /path/to/sft_checkpoint \
    --failure_reward -10.0 \
    --pool_size 16
```

Model checkpoints are saved to `models/<output_dir>/`.

---

## Repository File Guide

### `grpo_cadevolve.py`
Entry point. Handles:
- Rendering the dataset to a HuggingFace `Dataset` with `image` columns (cached to disk)
- Building chat-template prompts for Qwen2-VL
- Defining the reward function (`get_reward_function`)
- Instantiating `TopSampleGRPOTrainer` and launching training

### `grpo_loss.py`
Contains:
- `adv_select_top_samples` — selects `top_k` completions per prompt by absolute advantage
- `cppo_compute_loss` — clipped PPO loss computed only over selected samples, with per-token entropy and clip-ratio logging

### `cadevolve_dataset.py`
- `STLImageToCode` — PyTorch dataset iterating over STL meshes
- `render_7view_montage` — renders 7 canonical views into a single PIL image using Open3D
- `AdaptiveScaler`, `_pad_to_multiple`, etc. — vision preprocessing utilities to align images with Qwen2-VL's patch grid

### `utils_async.py`
- `init_pool` / `close_pool` — manages a `NonDaemonPool` of CadQuery worker processes
- `get_metrics_from_texts` — executes a batch of completion strings as CadQuery programs and returns `{iou, cd, auc}` dicts, with per-sample timeouts

### `normal_metrics.py`
Implements surface normal consistency: samples point clouds from both predicted and ground-truth meshes and computes the AUC of the cosine similarity distribution between nearest-neighbour normal pairs.

### `config.yaml`
GRPO hyperparameters parsed by `TrlParser`. Any field accepted by `trl.GRPOConfig` can be set here.

---

## Monitoring

Training metrics are logged to **Weights & Biases** (if configured) and include:
- `rewards/mean`, `rewards/std`
- `clip_ratio/low_mean`, `clip_ratio/high_mean`, `clip_ratio/region_mean`
- `entropy`

A decoded sample completion is printed to stdout every `gen_sample_steps` steps.

---

## Citation

```bibtex
@article{elistratov2026cadevolve,
  title   = {CADEvolve: Creating Realistic CAD via Program Evolution},
  author  = {Elistratov, Maksim and Barannikov, Marina and Ivanov, Gregory and
             Khrulkov, Valentin and Konushin, Anton and Kuznetsov, Andrey and
             Zhemchuzhnikov, Dmitrii},
  journal = {arXiv preprint arXiv:2602.16317},
  year    = {2026}
}
```
