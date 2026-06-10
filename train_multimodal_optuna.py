#!/usr/bin/env python
"""
Multimodal TERS Optuna Hyperparameter Search

Searches over filter configurations and learning rates for a single fusion type.
Each trial is logged as a separate wandb run, grouped by fusion type for easy
comparison in the wandb dashboard.

Usage:
    python train_multimodal_optuna.py --fusion_type late
    python train_multimodal_optuna.py --fusion_type attention --config src/configs/optuna_search.yaml
    python train_multimodal_optuna.py --fusion_type none --n_trials 5  # quick test
"""

import os
import sys
import yaml
import argparse
import random
import numpy as np
import optuna

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import wandb

sys.path.append('/scratch/work/sethih1/Multimodal_TERS')

from src.transforms import NormalizeVectorized, MinimumToZeroVectorized
from src.losses import get_loss_function
from src.datasets.multimodal_ters import MultimodalTERSDataset
from src.models.multimodal_unet import MultimodalAttentionUNet


# =============================================================================
# Reproducibility helpers (identical to train_multimodal.py)
# =============================================================================

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# Train / evaluate (identical structure to train_multimodal.py)
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total = 0
    for batch in loader:
        image = batch['image'].to(device)
        frequencies = batch['frequencies'].to(device)
        freq_mask = batch.get('freq_mask')
        if freq_mask is None:
            freq_mask = torch.ones_like(frequencies)
        freq_mask = freq_mask.to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()
        output = model(image, frequencies, freq_mask)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * image.size(0)
        total += image.size(0)
    return running_loss / total if total > 0 else 0.0


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            image = batch['image'].to(device)
            frequencies = batch['frequencies'].to(device)
            freq_mask = batch.get('freq_mask')
            if freq_mask is None:
                freq_mask = torch.ones_like(frequencies)
            freq_mask = freq_mask.to(device)
            target = batch['target'].to(device)

            output = model(image, frequencies, freq_mask)
            loss = criterion(output, target)

            running_loss += loss.item() * image.size(0)
            total += image.size(0)
            all_preds.append(output.cpu())
            all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_preds_binary = (torch.sigmoid(all_preds) > 0.5).float()

    dice_scores = []
    for c in range(4):
        intersection = (all_preds_binary[:, c] * all_targets[:, c]).sum()
        union = all_preds_binary[:, c].sum() + all_targets[:, c].sum()
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice.item())

    avg_loss = running_loss / total if total > 0 else 0.0
    avg_dice = float(np.mean(dice_scores))
    return avg_loss, avg_dice, dice_scores


# =============================================================================
# Optuna objective
# =============================================================================

def objective(trial, fusion_type, config, device):
    """
    One Optuna trial = one full training run.

    Hyperparameters suggested:
      - filters_idx  : categorical index into config['filters_options']
      - lr           : log-uniform in [lr_low, lr_high]

    Returns best validation Dice (maximised by Optuna).
    Raises optuna.TrialPruned if MedianPruner decides to stop early.
    """
    # --- Suggest hyperparameters ---
    filters_idx = trial.suggest_categorical(
        "filters_idx", list(range(len(config['filters_options'])))
    )
    filters = config['filters_options'][filters_idx]
    kernel_size = [3] * len(filters)
    lr = trial.suggest_float("lr", config['lr_low'], config['lr_high'], log=True)

    # Each trial gets a deterministic-but-unique seed
    trial_seed = config['seed'] + trial.number

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} | fusion={fusion_type} | filters={filters} | "
          f"lr={lr:.2e} | seed={trial_seed}")
    print('='*70)

    # --- WandB run for this trial ---
    wandb.init(
        project=config['wandb_project'],
        name=f"{fusion_type}_trial_{trial.number:02d}",
        group=f"optuna_{fusion_type}",
        tags=["optuna", fusion_type, f"depth_{len(filters)}", f"base_{filters[0]}"],
        config={
            "fusion_type": fusion_type,
            "filters": filters,
            "filters_idx": filters_idx,
            "kernel_size": kernel_size,
            "lr": lr,
            "att_channels": config['att_channels'],
            "batch_size": config['batch_size'],
            "epochs": config['epochs'],
            "loss_fn": config['loss_fn'],
            "freq_encoding": config['freq_encoding'],
            "freq_encoder_type": config.get('freq_encoder_type', 'transformer'),
            "num_bins": config.get('num_bins', 400),
            "max_freqs": config['max_freqs'],
            "num_channels": config['num_channels'],
            "trial_number": trial.number,
            "seed": trial_seed,
        },
        reinit=True,
    )

    seed_everything(trial_seed)

    # --- Datasets ---
    transform = transforms.Compose([
        NormalizeVectorized(),
        MinimumToZeroVectorized(),
    ])
    train_dataset = MultimodalTERSDataset(
        hdf5_path=config['train_path'],
        num_channels=config['num_channels'],
        max_freqs=config['max_freqs'],
        t_image=transform,
        train_aug=True,
        freq_encoding=config['freq_encoding'],
        num_bins=config.get('num_bins', 400),
    )
    val_dataset = MultimodalTERSDataset(
        hdf5_path=config['val_path'],
        num_channels=config['num_channels'],
        max_freqs=config['max_freqs'],
        t_image=transform,
        train_aug=False,
        freq_encoding=config['freq_encoding'],
        num_bins=config.get('num_bins', 400),
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(trial_seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(trial_seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        worker_init_fn=seed_worker,
        generator=val_generator,
    )

    # --- Model ---
    model = MultimodalAttentionUNet(
        in_channels=config['num_channels'],
        out_channels=4,
        filters=filters,
        att_channels=config['att_channels'],
        kernel_size=kernel_size,
        max_freqs=config['max_freqs'],
        freq_embed_dim=config['freq_embed_dim'],
        freq_output_dim=config['freq_output_dim'],
        fusion_type=fusion_type,
        freq_encoder_type=config.get('freq_encoder_type', 'transformer'),
        num_bins=config.get('num_bins', 400),
        bottleneck_dropout_p=trial.suggest_float('bottleneck_dropout_p', 0.0, 0.5) if trial is not None else 0.0,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    wandb.log({"model_params": total_params}, step=0)

    # --- Optimiser / loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = get_loss_function(config['loss_fn'])

    # --- Checkpoint dir ---
    save_dir = os.path.join(config['save_dir'], f"optuna_{fusion_type}")
    os.makedirs(save_dir, exist_ok=True)

    best_dice = 0.0
    atom_names = ['H', 'C', 'N', 'O']
    pruned = False

    try:
        for epoch in range(config['epochs']):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_dice, dice_per_channel = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_dice)

            # Save model whenever validation improves
            if val_dice > best_dice:
                best_dice = val_dice
                save_path = os.path.join(save_dir, f"trial_{trial.number:02d}_best.pt")
                torch.save(model.state_dict(), save_path)

            dice_str = ", ".join(
                [f"{atom_names[i]}:{d:.3f}" for i, d in enumerate(dice_per_channel)]
            )
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"  Epoch {epoch+1:3d}/{config['epochs']} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Dice: {val_dice:.4f} | [{dice_str}] | lr: {current_lr:.2e}"
            )

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "current_lr": current_lr,
                **{f"dice_{atom_names[i]}": dice_per_channel[i] for i in range(4)},
            })

            # Pruning check (MedianPruner compares against other trials)
            trial.report(val_dice, epoch)
            if trial.should_prune():
                print(f"  => Pruned at epoch {epoch+1} (val_dice={val_dice:.4f})")
                pruned = True
                break

    finally:
        train_dataset.close()
        val_dataset.close()
        wandb.log({"best_val_dice": best_dice})
        wandb.finish()

    if pruned:
        raise optuna.TrialPruned()

    return best_dice


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multimodal TERS Optuna Search')
    parser.add_argument('--fusion_type', type=str, required=True,
                        choices=['none', 'early', 'late', 'attention'],
                        help='Fusion strategy to optimise')
    parser.add_argument('--config', type=str,
                        default='src/configs/optuna_search.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--n_trials', type=int, default=None,
                        help='Override n_trials from config (useful for quick tests)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.n_trials is not None:
        config['n_trials'] = args.n_trials

    # Derive data paths from base if not explicitly set
    if 'train_path' not in config:
        config['train_path'] = os.path.join(config['data_base'], 'train.h5')
    if 'val_path' not in config:
        config['val_path'] = os.path.join(config['data_base'], 'val.h5')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("Multimodal TERS — Optuna Hyperparameter Search")
    print("=" * 70)
    print(f"  Device:      {device}")
    print(f"  Fusion type: {args.fusion_type}")
    print(f"  Trials:      {config['n_trials']}")
    print(f"  Epochs:      {config['epochs']}")
    print(f"  WandB proj:  {config['wandb_project']}")
    print(f"  WandB group: optuna_{args.fusion_type}")
    print(f"  Config:      {args.config}")
    print("=" * 70)

    # SQLite study storage — persistent across job restarts / array tasks
    os.makedirs(config['study_storage_dir'], exist_ok=True)
    storage_path = os.path.join(
        config['study_storage_dir'],
        f"study_{args.fusion_type}.db"
    )
    storage = f"sqlite:///{storage_path}"
    study_name = f"multimodal_ters_optuna_{args.fusion_type}"

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=config.get('pruner_n_startup_trials', 3),
        n_warmup_steps=config.get('pruner_n_warmup_steps', 15),
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,   # Resume automatically if job restarts
    )

    already_done = len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = config['n_trials'] - already_done
    print(f"Completed trials so far: {already_done}  |  Remaining: {remaining}")

    if remaining <= 0:
        print("All trials already complete. Exiting.")
    else:
        study.optimize(
            lambda trial: objective(trial, args.fusion_type, config, device),
            n_trials=remaining,
            gc_after_trial=True,
        )

    # --- Final summary ---
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print("\n" + "=" * 70)
    print(f"SEARCH COMPLETE — fusion_type={args.fusion_type}")
    print("=" * 70)
    print(f"  Completed: {len(completed)}  |  Pruned: {len(pruned_trials)}")

    if completed:
        best = study.best_trial
        best_filters = config['filters_options'][best.params['filters_idx']]
        print(f"\n  Best trial:  #{best.number}")
        print(f"  Val Dice:    {best.value:.4f}")
        print(f"  Filters:     {best_filters}")
        print(f"  LR:          {best.params['lr']:.2e}")

        print("\n  All completed trials (ranked):")
        for t in sorted(completed, key=lambda x: x.value, reverse=True):
            f_idx = t.params.get('filters_idx', '?')
            filters = config['filters_options'][f_idx] if isinstance(f_idx, int) else '?'
            print(f"    Trial {t.number:02d}: dice={t.value:.4f} | "
                  f"filters={filters} | lr={t.params.get('lr', 0):.2e}")


if __name__ == '__main__':
    main()
