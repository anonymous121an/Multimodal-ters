#!/usr/bin/env python
"""
Multimodal TERS Training Script

Train a Multimodal Attention U-Net for molecular structure prediction from TERS data.
Combines spectral images with frequency peaks using various fusion strategies.

Usage:
    python train_multimodal.py --fusion_type late --epochs 50 --lr 1e-4
    python train_multimodal.py --fusion_type none  # Image-only baseline
    python train_multimodal.py --compare  # Compare all fusion strategies
"""

import os
import wandb
import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Add project to path
sys.path.append('/scratch/work/sethih1/Multimodal_TERS')

from src.models.layers import ResBlock, AttentionBlock2d
from src.transforms import NormalizeVectorized, MinimumToZeroVectorized
from src.losses import get_loss_function

from src.fusion_techniques import EarlyFusion, LateFusion, CrossModalAttention, FiLM
from src.datasets.multimodal_ters import MultimodalTERSDataset
from src.models.frequency_encoder import FrequencyEncoder
from src.models.multimodal_unet import MultimodalAttentionUNet


def seed_everything(seed):
    """Set all relevant RNG seeds for reproducible training."""
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
    """Ensure each DataLoader worker has a deterministic NumPy/Python RNG."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
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
    """Evaluate model on validation set."""
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
    avg_dice = np.mean(dice_scores)
    
    return avg_loss, avg_dice, dice_scores


def train(args):
    """Main training function."""
    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using seed: {args.seed}")

    # Initialize wandb
    wandb_project = os.environ.get("WANDB_PROJECT", "multimodal-ters")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", f"fusion_{args.fusion_type}")
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=vars(args)
    )
    
    # Transform
    transform = transforms.Compose([
        NormalizeVectorized(),
        MinimumToZeroVectorized()
    ])
    
    # Create datasets
    train_dataset = MultimodalTERSDataset(
        hdf5_path=args.train_path,
        num_channels=args.num_channels,
        max_freqs=args.max_freqs,
        t_image=transform,
        train_aug=True,
        freq_encoding=args.freq_encoding
    )
    
    val_dataset = MultimodalTERSDataset(
        hdf5_path=args.val_path,
        num_channels=args.num_channels,
        max_freqs=args.max_freqs,
        t_image=transform,
        train_aug=False,
        freq_encoding=args.freq_encoding
    )
    
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=train_generator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=val_generator
    )
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"Frequency encoding: {args.freq_encoding}")
    
    # Create model
    model = MultimodalAttentionUNet(
        in_channels=args.num_channels,
        out_channels=4,
        filters=[64, 128, 256, 512],
        att_channels=64,
        kernel_size=[3, 3, 3, 3],
        max_freqs=args.max_freqs,
        freq_embed_dim=128,
        freq_output_dim=512,
        fusion_type=args.fusion_type
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: fusion_type='{args.fusion_type}', parameters={total_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = get_loss_function(args.loss_fn)
    
    print(f"Training: epochs={args.epochs}, lr={args.lr}, loss={args.loss_fn}")
    print("-" * 70)
    
    # Training loop
    best_dice = 0.0
    atom_names = ['H', 'C', 'N', 'O']
    
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, dice_per_channel = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(args.save_dir, f'best_multimodal_{args.fusion_type}.pt')
            torch.save(model.state_dict(), save_path)

        dice_str = ", ".join([f"{atom_names[i]}:{d:.3f}" for i, d in enumerate(dice_per_channel)])
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Dice: {val_dice:.4f} | [{dice_str}]")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            **{f"dice_{atom_names[i]}": dice_per_channel[i] for i in range(4)}
        })
    
    print("-" * 70)
    print(f"Best validation Dice: {best_dice:.4f}")
    
    # Cleanup
    train_dataset.close()
    val_dataset.close()
    
    return best_dice


def compare_fusion_strategies(args):
    """Compare all fusion strategies."""
    fusion_types = ['none', 'early', 'late', 'attention', 'film', 'hybrid']
    results = {}
    wandb_project = os.environ.get("WANDB_PROJECT", "multimodal-ters")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", "fusion_comparison")
    wandb.init(project=wandb_project, name=wandb_run_name, config=vars(args))

    for fusion in fusion_types:
        print(f"\n{'='*70}")
        print(f"Training with fusion type: {fusion}")
        print('='*70)

        args.fusion_type = fusion
        args.epochs = 20  # Quick comparison
        best_dice = train(args)
        results[fusion] = best_dice

        # Log best dice for each fusion type
        wandb.log({f"best_dice_{fusion}": best_dice})

    # Summary
    print("\n" + "="*70)
    print("FUSION STRATEGY COMPARISON")
    print("="*70)
    for fusion, dice in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{fusion:12s}: Dice={dice:.4f}")

    # Log all fusion results as a single figure for comparison
    fig, ax = plt.subplots()
    ax.bar(results.keys(), results.values())
    ax.set_ylabel("Best Dice Score")
    ax.set_xlabel("Fusion Type")
    ax.set_title("Fusion Strategy Comparison")
    wandb.log({"fusion_comparison": wandb.Image(fig)})
    plt.close(fig)

    return results


def main():
    parser = argparse.ArgumentParser(description='Multimodal TERS Training')
    
    # Data paths
    parser.add_argument('--train_path', type=str, 
                        default='/scratch/phys/sin/sethih1/Multimodal_TERS/planar_hdf5_0.05/train.h5')
    parser.add_argument('--val_path', type=str,
                        default='/scratch/phys/sin/sethih1/Multimodal_TERS/planar_hdf5_0.05/val.h5')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    # Model config
    parser.add_argument('--fusion_type', type=str, default='late',
                        choices=['none', 'early', 'late', 'attention', 'film', 'hybrid'])
    parser.add_argument('--num_channels', type=int, default=100)
    parser.add_argument('--max_freqs', type=int, default=60)
    
    # Training config
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss_fn', type=str, default='dice_loss',
                        choices=['dice_loss', 'bce_loss', 'focal_loss', 'combined_loss'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--freq_encoding', type=str, default='binning',
                        choices=['binning', 'normalize'])
    
    # Mode
    parser.add_argument('--compare', action='store_true', help='Compare all fusion strategies')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.compare:
        compare_fusion_strategies(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
