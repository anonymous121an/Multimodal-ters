# Multimodal-TERS

A PyTorch framework for **molecular structure prediction from Tip-Enhanced Raman Spectroscopy (TERS) data** using a multimodal Attention U-Net. The model combines multi-channel spectral images with frequency peak information via configurable fusion strategies.

---

## Overview

TERS measurements produce spatially-resolved spectral images alongside discrete frequency peaks. This repository trains a **Multimodal Attention U-Net** that jointly processes:

- **Spectral image** — multi-channel spatial map (e.g. 100 channels, H×W)
- **Frequency peaks** — sparse Raman peak positions encoded as a binned vector or a padded/normalized sequence

The network outputs a **4-class segmentation mask** predicting the molecular structure at each pixel.

---

## Project Structure

```
Multimodal-ters/
├── train_multimodal.py          # Main training script
├── train_fusion_search.sh       # SLURM array job for fusion strategy search
├── requirements.txt
├── model_checkpoints/           # Pretrained model weights
│   ├── ablation-none_fusion-none_freq-binning_seed-42/
│   ├── ablation-early_fusion-early_freq-binning_seed-42/
│   ├── ablation-late_fusion-late_freq-binning_seed-42/
│   ├── ablation-attention_fusion-attention_freq-binning_seed-42/
│   ├── ablation-film_fusion-film_freq-binning_seed-42/
│   └── ablation-hybrid_fusion-hybrid_freq-binning_seed-42/
└── src/
    ├── configs/
    │   └── train_multimodal.yaml    # Training configuration
    ├── datasets/
    │   └── multimodal_ters.py       # HDF5 dataset loader
    ├── models/
    │   ├── multimodal_unet.py       # Multimodal Attention U-Net
    │   ├── frequency_encoder.py     # Frequency peak encoder
    │   └── layers.py                # ResBlock, AttentionBlock2d
    ├── fusion_techniques/
    │   └── fusion.py                # EarlyFusion, LateFusion, CrossModalAttention, FiLM
    ├── losses/
    │   └── losses.py                # Dice, BCE, Focal, Combined losses
    ├── metrics/
    │   └── metrics.py               # IoU, Dice, F1, Precision, Recall
    ├── transforms/                  # Normalization and augmentation
    └── trainer/                     # Training and evaluation loops
```

---

## Installation

```bash
git clone https://github.com/anonymous121an/Multimodal-ters.git
cd Multimodal-ters
pip install -r requirements.txt
pip install -e src/fusion_techniques/   # Install fusion_techniques as a local package
```

**Requirements:** Python 3.8+, PyTorch 2.10, CUDA recommended.

---

## Data Format

Data is stored in **HDF5** files with the following structure:

```
train.h5
├── channels_100   (N, H, W, 100)   # Spectral image channels
├── targets        (N, 4, H, W)     # 4-class segmentation masks
└── frequencies/
    ├── 0          (n_peaks,)        # Raman peak positions [cm⁻¹] for sample 0
    ├── 1          ...
    └── ...
```

Set `train_path` and `val_path` in the config to point to your `.h5` files.

---

## Configuration

All training parameters are defined in `src/configs/train_multimodal.yaml`:

```yaml
# Paths
train_path: /path/to/train.h5
val_path:   /path/to/val.h5
save_dir:   /path/to/checkpoints

# Model
fusion_type:    late        # none | early | late | attention | film | hybrid
num_channels:   100         # Spectral channels to use
max_freqs:      100         # Max frequency peaks (for normalize encoding)
freq_encoding:  binning     # binning | normalize

# Training
epochs:       100
batch_size:   16
lr:           0.0001
loss_fn:      dice_loss     # dice_loss | bce_loss | focal_loss | combined_loss
num_workers:  8
seed:         42

# Logging
use_wandb:        true
wandb_project:    multimodal-ters
wandb_run_name:   fusion_{fusion_type}   # {fusion_type} is auto-filled

# Mode
compare: false   # If true, runs all fusion types sequentially and compares
```

---

## Fusion Strategies

| `fusion_type` | Description |
|---|---|
| `none` | Image-only baseline — frequency peaks are ignored |
| `early` | Frequency features projected to a spatial map and concatenated to the input |
| `late` | Frequency features injected at the U-Net bottleneck |
| `attention` | Transformer-style cross-attention between decoder skip connections and frequency sequence |
| `film` | Feature-wise Linear Modulation (FiLM) of decoder features conditioned on frequency |
| `hybrid` | Combines both `attention` and `film` in the decoder |

---

## Frequency Encoding

| `freq_encoding` | Description |
|---|---|
| `binning` | Multi-hot 400-bin vector over [0, 4000] cm⁻¹ (default) |
| `normalize` | Padded sequence of peaks normalized to [0, 1], with a validity mask |

---

## Training

**Single run using config defaults:**

```bash
python train_multimodal.py --config src/configs/train_multimodal.yaml
```

**Override a config value from the CLI:**

```bash
python train_multimodal.py --config src/configs/train_multimodal.yaml --fusion_type attention
```

**Image-only baseline:**

```bash
python train_multimodal.py --config src/configs/train_multimodal.yaml --fusion_type none
```

**Compare all fusion strategies in one run:**

```bash
python train_multimodal.py --config src/configs/train_multimodal.yaml --compare
```

**Disable W&B logging:**

```bash
python train_multimodal.py --config src/configs/train_multimodal.yaml --no-use-wandb
```

---

## SLURM Fusion Search

To train all 6 fusion strategies in parallel as a SLURM array job:

```bash
sbatch train_fusion_search.sh
```

This submits array tasks 0–5 mapping to: `none`, `early`, `late`, `attention`, `film`, `hybrid`.

**Override the config path:**

```bash
CONFIG_PATH=/path/to/train_multimodal.yaml sbatch train_fusion_search.sh
```

Each job:
- Requests 1 GPU, 200 GB RAM, 16 CPUs, up to 48 hours
- Logs GPU and resource usage per fusion type
- Activates the virtual environment automatically

---

## Metrics

Models are evaluated using:

- **Dice coefficient**
- **Intersection over Union (IoU)**
- **F1 score**
- **Precision & Recall**

---

## Pretrained Checkpoints

Trained model weights for all 6 fusion strategies are provided in `model_checkpoints/`:

---

## W&B Integration

W&B logging is optional. Enable or disable it in the config or via CLI:

```bash
# Enable
python train_multimodal.py --config src/configs/train_multimodal.yaml --use-wandb

# Disable
python train_multimodal.py --config src/configs/train_multimodal.yaml --no-use-wandb
```

The `wandb_run_name` field supports a `{fusion_type}` placeholder that is automatically substituted at runtime (e.g. `fusion_late`).
