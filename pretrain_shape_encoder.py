#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ShapeDisc Encoder Pretraining Script (Reconstruction-based)

Uses self-supervised reconstruction task to pretrain the SE(3)-equivariant
shape encoder, following the design in shape_plan.md.

Strategy:
- Input: Binary mask or SDF with noise perturbation
- Task: Autoencoder - encode to shape embedding, then reconstruct
- Augmentation: Gaussian noise only (preserves SE(3) properties)
- Loss: MSE + Dice reconstruction loss
- Prior extraction: Per-label Gaussian (μ_l, Σ_l) from embeddings

Key Design Principles:
1. No rotation/scaling augmentation (preserves SE(3) equivariance)
2. Self-supervised (each mask supervises itself)
3. Per-label distribution modeling (not inter-class discrimination)
4. Works with limited data (10 subjects sufficient)
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import nibabel as nib
from scipy.ndimage import zoom, distance_transform_edt, gaussian_filter
from tqdm import tqdm

# Import proto_atlas
from proto_atlas import SE3Encoder


# ---------------------------
# Shape Decoder
# ---------------------------
class ShapeDecoder(nn.Module):
    """
    3D Decoder to reconstruct shape from embedding.

    Architecture: Fully connected → 3D transposed convolutions
    Input: (batch, emb_dim)
    Output: (batch, 1, D, H, W)
    """
    def __init__(self, emb_dim: int = 128, target_size: Tuple[int, int, int] = (96, 96, 96)):
        super().__init__()
        self.emb_dim = emb_dim
        self.target_size = target_size

        # Initial spatial size after FC layers
        self.init_size = 6  # 6x6x6
        self.init_channels = 256

        # Fully connected layers to create initial 3D feature map
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.init_channels * self.init_size ** 3),
            nn.ReLU()
        )

        # 3D Transposed Convolutional layers
        # 6x6x6 -> 12x12x12 -> 24x24x24 -> 48x48x48 -> 96x96x96
        self.deconv = nn.Sequential(
            # 6 -> 12
            nn.ConvTranspose3d(self.init_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(),

            # 12 -> 24
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(),

            # 24 -> 48
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(),

            # 48 -> 96
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(16),
            nn.ReLU(),

            # Final layer
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: (batch, emb_dim)

        Returns:
            reconstructed: (batch, 1, D, H, W)
        """
        batch_size = embedding.size(0)

        # FC layers
        x = self.fc(embedding)  # (batch, init_channels * init_size^3)

        # Reshape to 3D
        x = x.view(batch_size, self.init_channels, self.init_size, self.init_size, self.init_size)

        # Deconvolution
        x = self.deconv(x)  # (batch, 1, 96, 96, 96)

        return x


# ---------------------------
# Dataset
# ---------------------------
class ShapeReconstructionDataset(Dataset):
    """
    Dataset for self-supervised shape reconstruction.

    For each mask:
    1. Convert to SDF (optional)
    2. Add Gaussian noise perturbation
    3. Return (noisy, clean) pairs for reconstruction
    """

    def __init__(
        self,
        data_dir: str,
        labels: List[str],
        crop_size: Tuple[int, int, int] = (96, 96, 96),
        n_perturbations: int = 10,
        noise_std: float = 0.02,
        use_sdf: bool = True,
        seed: int = 42
    ):
        """
        Args:
            data_dir: Root directory containing annotation files
            labels: List of anatomical structure names
            crop_size: Size to crop/resize volumes to (D, H, W)
            n_perturbations: Number of noise perturbations per mask
            noise_std: Standard deviation of Gaussian noise
            use_sdf: Whether to use signed distance fields
            seed: Random seed
        """
        self.data_dir = Path(data_dir)
        self.labels = labels
        self.crop_size = crop_size
        self.n_perturbations = n_perturbations
        self.noise_std = noise_std
        self.use_sdf = use_sdf

        np.random.seed(seed)
        random.seed(seed)

        # Find all annotation files
        self.annotation_files = []
        if self.data_dir.is_dir():
            seg_files = list(self.data_dir.glob('*_seg.nii.gz'))
            if len(seg_files) > 0:
                self.annotation_files.extend(seg_files)

        print(f"Found {len(self.annotation_files)} annotation files")

        # Extract all masks
        self.base_samples = []  # List of (clean_mask, label_idx, subject_id)
        for ann_path in tqdm(self.annotation_files, desc="Loading annotations"):
            self._load_annotations(ann_path)

        print(f"Base samples (subjects × labels): {len(self.base_samples)}")

        # Create perturbed samples
        self.samples = []  # List of (noisy, clean, label_idx)
        for clean_mask, label_idx, subject_id in tqdm(self.base_samples, desc="Creating perturbations"):
            # Add original (no perturbation)
            self.samples.append((clean_mask, clean_mask, label_idx))

            # Add perturbed versions
            for _ in range(self.n_perturbations):
                noisy_mask = self._add_noise(clean_mask.numpy())
                noisy_tensor = torch.from_numpy(noisy_mask).float()
                self.samples.append((noisy_tensor, clean_mask, label_idx))

        print(f"Total training samples (with perturbations): {len(self.samples)}")

    def _load_annotations(self, ann_path: Path):
        """Load annotations and extract individual masks"""
        try:
            nii = nib.load(str(ann_path))

            # ✅ CRITICAL: Reorient to RAS before processing
            nii = nib.as_closest_canonical(nii)
            data = nii.get_fdata().astype(np.int32)

            subject_id = ann_path.stem.replace('_seg', '')

            for label_idx, label_name in enumerate(self.labels):
                voxel_value = label_idx + 1
                binary_mask = (data == voxel_value).astype(np.float32)

                if binary_mask.sum() == 0:
                    warnings.warn(f"No voxels for {label_name} in {subject_id}")
                    continue

                # Preprocess mask
                processed_mask = self._preprocess_mask(binary_mask)

                # Convert to SDF if requested
                if self.use_sdf:
                    processed_mask = self._mask_to_sdf(processed_mask)

                mask_tensor = torch.from_numpy(processed_mask).float()
                self.base_samples.append((mask_tensor, label_idx, subject_id))

        except Exception as e:
            warnings.warn(f"Failed to load {ann_path}: {e}")

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Crop and resize mask to target size"""
        # Find bounding box
        coords = np.argwhere(mask > 0.5)
        if len(coords) == 0:
            return np.zeros(self.crop_size, dtype=np.float32)

        # Extract ROI with padding
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        pad = 10
        z_min = max(0, z_min - pad)
        y_min = max(0, y_min - pad)
        x_min = max(0, x_min - pad)
        z_max = min(mask.shape[0], z_max + pad)
        y_max = min(mask.shape[1], y_max + pad)
        x_max = min(mask.shape[2], x_max + pad)

        roi = mask[z_min:z_max, y_min:y_max, x_min:x_max]

        # Resize to target size
        scale = [self.crop_size[i] / roi.shape[i] for i in range(3)]
        resized = zoom(roi, scale, order=1)

        # Ensure exact target size
        result = np.zeros(self.crop_size, dtype=np.float32)
        slices = tuple(slice(0, min(resized.shape[i], self.crop_size[i])) for i in range(3))
        result[slices] = resized[slices]

        return result

    def _mask_to_sdf(self, mask: np.ndarray) -> np.ndarray:
        """Convert binary mask to signed distance field"""
        # Binarize first
        binary = (mask > 0.5).astype(np.float32)

        # Compute SDF
        pos_dist = distance_transform_edt(binary)
        neg_dist = distance_transform_edt(1 - binary)
        sdf = pos_dist - neg_dist

        # Normalize to [-1, 1] range
        if sdf.std() > 0:
            sdf = sdf / (abs(sdf).max() + 1e-8)

        return sdf.astype(np.float32)

    def _add_noise(self, mask: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to mask (light perturbation only)"""
        noise = np.random.randn(*mask.shape) * self.noise_std
        noisy = mask + noise

        # Optional: add very slight elastic deformation
        if np.random.rand() < 0.3:  # 30% chance
            sigma = np.random.uniform(1.0, 2.0)
            noisy = gaussian_filter(noisy, sigma=sigma)

        return noisy.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        noisy, clean, label_idx = self.samples[idx]

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        noisy = noisy.unsqueeze(0)
        clean = clean.unsqueeze(0)

        return {
            'noisy': noisy,        # (1, D, H, W)
            'clean': clean,        # (1, D, H, W)
            'label': label_idx
        }


# ---------------------------
# Loss Functions
# ---------------------------
class ReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss: MSE + Dice
    """
    def __init__(self, lambda_dice: float = 0.5):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.mse = nn.MSELoss()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Soft Dice loss

        Args:
            pred: (B, 1, D, H, W) in [0, 1]
            target: (B, 1, D, H, W) in [0, 1] or binary
        """
        smooth = 1e-6

        # Flatten spatial dimensions
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Dice coefficient
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + smooth) / (union + smooth)

        # Dice loss
        return 1.0 - dice.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: (B, 1, D, H, W) reconstructed
            target: (B, 1, D, H, W) ground truth

        Returns:
            Dict with total loss and components
        """
        # MSE loss
        loss_mse = self.mse(pred, target)

        # Dice loss (binarize target for SDF)
        target_binary = (target > 0.0).float() if target.min() < 0 else target
        loss_dice = self.dice_loss(pred, target_binary)

        # Total loss
        loss_total = loss_mse + self.lambda_dice * loss_dice

        return {
            'total': loss_total,
            'mse': loss_mse,
            'dice': loss_dice
        }


# ---------------------------
# Training
# ---------------------------
def train_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""
    encoder.train()
    decoder.train()

    total_loss = 0.0
    total_mse = 0.0
    total_dice = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        noisy = batch['noisy'].to(device)  # (B, 1, D, H, W)
        clean = batch['clean'].to(device)  # (B, 1, D, H, W)

        # Forward pass
        embedding = encoder(noisy)             # (B, emb_dim)
        reconstructed = decoder(embedding)     # (B, 1, D, H, W)

        # Compute loss
        loss_dict = criterion(reconstructed, clean)
        loss = loss_dict['total']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        total_mse += loss_dict['mse'].item()
        total_dice += loss_dict['dice'].item()
        n_batches += 1

        pbar.set_postfix({
            'loss': loss.item(),
            'mse': loss_dict['mse'].item(),
            'dice': loss_dict['dice'].item()
        })

    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'dice': total_dice / n_batches
    }


@torch.no_grad()
def validate(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model"""
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_dice = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)

        embedding = encoder(noisy)
        reconstructed = decoder(embedding)

        loss_dict = criterion(reconstructed, clean)

        total_loss += loss_dict['total'].item()
        total_mse += loss_dict['mse'].item()
        total_dice += loss_dict['dice'].item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'dice': total_dice / n_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Pretrain Shape Encoder (Reconstruction-based)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing *_seg.nii.gz files')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/shape_pretrain',
                        help='Directory to save checkpoints')
    parser.add_argument('--use_se3', action='store_true',
                        help='Use SE(3)-equivariant encoder (RXFM_Net)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--crop_size', type=int, default=96,
                        help='Crop size for volumes (cubic)')
    parser.add_argument('--n_perturbations', type=int, default=10,
                        help='Number of noise perturbations per mask')
    parser.add_argument('--noise_std', type=float, default=0.02,
                        help='Standard deviation of Gaussian noise')
    parser.add_argument('--use_sdf', action='store_true',
                        help='Use signed distance fields')
    parser.add_argument('--lambda_dice', type=float, default=0.5,
                        help='Weight for Dice loss')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Labels (subcortical structures)
    labels = ["hippocampus", "amygdala", "caudate", "putamen", "pallidum", "thalamus", "accumbens"]
    print(f"\nTraining with {len(labels)} labels: {labels}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build dataset
    print("\n" + "="*60)
    print("Building Dataset")
    print("="*60)
    crop_size = (args.crop_size, args.crop_size, args.crop_size)
    full_dataset = ShapeReconstructionDataset(
        data_dir=args.data_dir,
        labels=labels,
        crop_size=crop_size,
        n_perturbations=args.n_perturbations,
        noise_std=args.noise_std,
        use_sdf=args.use_sdf,
        seed=args.seed
    )

    # Split train/val
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Build encoder and decoder
    print("\n" + "="*60)
    print("Building Encoder/Decoder")
    print("="*60)
    print(f"Using {'SE(3)-equivariant' if args.use_se3 else '3D CNN'} encoder")

    encoder = SE3Encoder(emb_dim=args.emb_dim, use_e3nn=args.use_se3).to(device)
    decoder = ShapeDecoder(emb_dim=args.emb_dim, target_size=crop_size).to(device)

    # Count parameters
    n_params_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_params_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Encoder parameters: {n_params_enc:,}")
    print(f"Decoder parameters: {n_params_dec:,}")
    print(f"Total trainable parameters: {n_params_enc + n_params_dec:,}")

    # Loss and optimizer
    criterion = ReconstructionLoss(lambda_dice=args.lambda_dice)
    optimizer = Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 60)

        # Train
        train_metrics = train_epoch(
            encoder, decoder, train_loader, criterion, optimizer, device, epoch
        )
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"MSE: {train_metrics['mse']:.4f}, "
              f"Dice: {train_metrics['dice']:.4f}")

        # Validate
        val_metrics = validate(encoder, decoder, val_loader, criterion, device)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"MSE: {val_metrics['mse']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}")

        # Step scheduler
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'args': vars(args)
        }

        # Save latest
        latest_path = os.path.join(args.output_dir, 'shape_encoder_latest.pth')
        torch.save(checkpoint, latest_path)

        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = os.path.join(args.output_dir, 'shape_encoder_best.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss={val_metrics['loss']:.4f})")

        # Save periodic checkpoints
        if epoch % 10 == 0:
            periodic_path = os.path.join(args.output_dir, f'shape_encoder_epoch{epoch}.pth')
            torch.save(checkpoint, periodic_path)

    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run extract_shape_prototypes.py to extract per-label Gaussian priors")
    print("  2. Visualize reconstructions with visualize_shape_reconstruction.py")
    print("  3. Integrate into segmentation training")
    print("="*60)


if __name__ == '__main__':
    main()
