#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Location Encoder Pretraining with Left-Right Split

This script extends pretrain_loc_encoder.py to:
1. Split bilateral labels (1-7) into left-right labels (1-14)
2. Extract enhanced location features (centroid + adjacency)
3. Train EnhancedRelationEncoder with reconstruction task

Based on:
- pretrain_loc_encoder.py (training framework)
- location_encoder_enhanced.py (enhanced features)
- split_bilateral_labels.py (L-R splitting logic)
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Optional
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import nibabel as nib
from tqdm import tqdm

# Import enhanced encoder from location_encoder_enhanced.py
from location_encoder_enhanced import (
    EnhancedRelationEncoder,
    EnhancedRelationDecoder,
    extract_enhanced_location_features
)


# ---------------------------
# Left-Right Splitting Utilities
# ---------------------------
def split_bilateral_labels(data: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Split bilateral labels (1-7) into left-right labels (1-14)

    Mapping:
        1 (Hippocampus) -> 1 (L-Hippo), 2 (R-Hippo)
        2 (Amygdala) -> 3 (L-Amyg), 4 (R-Amyg)
        3 (Caudate) -> 5 (L-Caud), 6 (R-Caud)
        4 (Putamen) -> 7 (L-Put), 8 (R-Put)
        5 (Pallidum) -> 9 (L-Pall), 10 (R-Pall)
        6 (Thalamus) -> 11 (L-Thal), 12 (R-Thal)
        7 (Accumbens) -> 13 (L-Acc), 14 (R-Acc)

    Args:
        data: (D, H, W) segmentation with labels 1-7
        affine: NIfTI affine matrix

    Returns:
        split_data: (D, H, W) segmentation with labels 1-14
    """
    bilateral_to_lr = {
        1: (1, 2),    # Hippocampus
        2: (3, 4),    # Amygdala
        3: (5, 6),    # Caudate
        4: (7, 8),    # Putamen
        5: (9, 10),   # Pallidum
        6: (11, 12),  # Thalamus
        7: (13, 14),  # Accumbens
    }

    split_data = np.zeros_like(data, dtype=np.uint8)

    # Determine which voxel dimension corresponds to RAS X (left-right axis)
    abs_coeffs = [abs(affine[0, 0]), abs(affine[0, 1]), abs(affine[0, 2])]
    lr_axis = np.argmax(abs_coeffs)  # Left-right axis dimension
    lr_increasing = affine[0, lr_axis] > 0  # True if increasing index = left

    # Split bilateral labels using midline on correct axis
    for bilateral_label, (left_label, right_label) in bilateral_to_lr.items():
        mask = (data == bilateral_label)

        if not mask.any():
            continue

        # Find midline on the left-right axis
        coords = np.where(mask)
        midline_pos = (coords[lr_axis].min() + coords[lr_axis].max()) / 2

        # Create masks for left and right based on the correct axis
        if lr_axis == 0:
            if lr_increasing:
                left_mask = mask & (np.arange(data.shape[0])[:, None, None] > midline_pos)
                right_mask = mask & (np.arange(data.shape[0])[:, None, None] <= midline_pos)
            else:
                left_mask = mask & (np.arange(data.shape[0])[:, None, None] <= midline_pos)
                right_mask = mask & (np.arange(data.shape[0])[:, None, None] > midline_pos)
        elif lr_axis == 1:
            if lr_increasing:
                left_mask = mask & (np.arange(data.shape[1])[None, :, None] > midline_pos)
                right_mask = mask & (np.arange(data.shape[1])[None, :, None] <= midline_pos)
            else:
                left_mask = mask & (np.arange(data.shape[1])[None, :, None] <= midline_pos)
                right_mask = mask & (np.arange(data.shape[1])[None, :, None] > midline_pos)
        else:  # lr_axis == 2
            if lr_increasing:
                left_mask = mask & (np.arange(data.shape[2])[None, None, :] > midline_pos)
                right_mask = mask & (np.arange(data.shape[2])[None, None, :] <= midline_pos)
            else:
                left_mask = mask & (np.arange(data.shape[2])[None, None, :] <= midline_pos)
                right_mask = mask & (np.arange(data.shape[2])[None, None, :] > midline_pos)

        split_data[left_mask] = left_label
        split_data[right_mask] = right_label

    return split_data


# ---------------------------
# Protocol Pairs Definition (for 14 L-R labels)
# ---------------------------
def get_enhanced_protocol_pairs() -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Define protocol pairs for 14 left-right split labels.

    Returns:
        labels: List of 14 label names
        protocol_pairs: List of (i, j) pairs (indices 0-13)
    """
    # 14 labels after L-R split
    labels = [
        "L-Hippocampus",    # 0 (label 1)
        "R-Hippocampus",    # 1 (label 2)
        "L-Amygdala",       # 2 (label 3)
        "R-Amygdala",       # 3 (label 4)
        "L-Caudate",        # 4 (label 5)
        "R-Caudate",        # 5 (label 6)
        "L-Putamen",        # 6 (label 7)
        "R-Putamen",        # 7 (label 8)
        "L-Pallidum",       # 8 (label 9)
        "R-Pallidum",       # 9 (label 10)
        "L-Thalamus",       # 10 (label 11)
        "R-Thalamus",       # 11 (label 12)
        "L-Accumbens",      # 12 (label 13)
        "R-Accumbens"       # 13 (label 14)
    ]

    protocol_pairs = []

    # Intra-hemispheric relations (within left or right)
    # Left hemisphere
    protocol_pairs.extend([
        (0, 2),   # L-Hippo - L-Amyg
        (0, 10),  # L-Hippo - L-Thal
        (2, 10),  # L-Amyg - L-Thal
        (4, 6),   # L-Caud - L-Put (striatum)
        (6, 8),   # L-Put - L-Pall
        (4, 8),   # L-Caud - L-Pall
        (4, 12),  # L-Caud - L-Acc
        (6, 12),  # L-Put - L-Acc
        (2, 6),   # L-Amyg - L-Put
        (0, 4),   # L-Hippo - L-Caud
        (10, 6),  # L-Thal - L-Put
        (10, 8),  # L-Thal - L-Pall
        (10, 4),  # L-Thal - L-Caud
        (10, 12), # L-Thal - L-Acc
        (12, 2),  # L-Acc - L-Amyg
    ])

    # Right hemisphere (mirror of left)
    protocol_pairs.extend([
        (1, 3),   # R-Hippo - R-Amyg
        (1, 11),  # R-Hippo - R-Thal
        (3, 11),  # R-Amyg - R-Thal
        (5, 7),   # R-Caud - R-Put
        (7, 9),   # R-Put - R-Pall
        (5, 9),   # R-Caud - R-Pall
        (5, 13),  # R-Caud - R-Acc
        (7, 13),  # R-Put - R-Acc
        (3, 7),   # R-Amyg - R-Put
        (1, 5),   # R-Hippo - R-Caud
        (11, 7),  # R-Thal - R-Put
        (11, 9),  # R-Thal - R-Pall
        (11, 5),  # R-Thal - R-Caud
        (11, 13), # R-Thal - R-Acc
        (13, 3),  # R-Acc - R-Amyg
    ])

    # Inter-hemispheric relations (between left and right)
    # Same structure across hemispheres
    protocol_pairs.extend([
        (0, 1),   # L-Hippo - R-Hippo
        (2, 3),   # L-Amyg - R-Amyg
        (4, 5),   # L-Caud - R-Caud
        (6, 7),   # L-Put - R-Put
        (8, 9),   # L-Pall - R-Pall
        (10, 11), # L-Thal - R-Thal
        (12, 13), # L-Acc - R-Acc
    ])

    print(f"\n{'='*60}")
    print("Enhanced Protocol Pairs (L-R Split):")
    print(f"{'='*60}")
    print(f"Total labels: {len(labels)}")
    print(f"Total protocol pairs: {len(protocol_pairs)}")
    print(f"  - Intra-hemispheric (left): 15 pairs")
    print(f"  - Intra-hemispheric (right): 15 pairs")
    print(f"  - Inter-hemispheric: 7 pairs")
    print(f"{'='*60}\n")

    return labels, protocol_pairs


# ---------------------------
# Dataset
# ---------------------------
class EnhancedRelationDataset(Dataset):
    """
    Dataset for enhanced relation reconstruction with L-R split.

    For each subject:
    1. Load segmentation (labels 1-7)
    2. Split into L-R labels (labels 1-14)
    3. Extract enhanced features (centroid + adjacency) for protocol pairs
    4. Apply perturbations for data augmentation
    """

    def __init__(
        self,
        data_dir: str,
        labels: List[str],
        protocol_pairs: List[Tuple[int, int]],
        n_perturbations: int = 100,
        noise_std: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            data_dir: Root directory containing *_seg.nii.gz files
            labels: List of 14 label names (after L-R split)
            protocol_pairs: List of (i, j) pairs
            n_perturbations: Number of noise perturbations per subject
            noise_std: Standard deviation of perturbation noise
            seed: Random seed
        """
        self.data_dir = Path(data_dir)
        self.labels = labels
        self.protocol_pairs = protocol_pairs
        self.K = len(protocol_pairs)
        self.n_perturbations = n_perturbations
        self.noise_std = noise_std

        np.random.seed(seed)
        random.seed(seed)

        # Find all annotation files
        self.annotation_files = []
        if self.data_dir.is_dir():
            seg_files = list(self.data_dir.glob('*_seg.nii.gz'))
            if len(seg_files) > 0:
                self.annotation_files.extend(seg_files)

        print(f"Found {len(self.annotation_files)} annotation files")

        # Extract ground truth enhanced relation features
        self.gt_relations = []  # List of (K, 7) arrays
        for ann_path in tqdm(self.annotation_files, desc="Extracting enhanced relation features"):
            rel_vecs = self._load_annotations(ann_path)
            if rel_vecs is not None:
                self.gt_relations.append(rel_vecs)

        print(f"Successfully loaded {len(self.gt_relations)} subjects")

        # Generate perturbed samples
        self.samples = []  # List of (perturbed, original) pairs
        for gt_rel in self.gt_relations:
            # Original (no perturbation)
            self.samples.append((gt_rel, gt_rel))

            # Perturbed versions
            for _ in range(self.n_perturbations):
                noise = np.random.randn(*gt_rel.shape) * self.noise_std
                perturbed = gt_rel + noise
                self.samples.append((perturbed.astype(np.float32), gt_rel))

        print(f"Total training samples (with perturbations): {len(self.samples)}")

    def _load_annotations(self, ann_path: Path) -> Optional[np.ndarray]:
        """Load annotations, split L-R, and extract enhanced features"""
        try:
            nii = nib.load(str(ann_path))

            # Reorient to RAS
            nii = nib.as_closest_canonical(nii)
            data = nii.get_fdata().astype(np.int32)
            affine = nii.affine
            spacing=nii.header['pixdim'][1:4]
            # spacing=[1.0,1.0,1.0]

            # Split bilateral labels (1-7) -> L-R labels (1-14)
            split_data = split_bilateral_labels(data, affine)

            # Extract enhanced features using protocol pairs
            # protocol_pairs are in 0-indexed format, but split_data uses 1-14 labels
            # Convert protocol pairs to label indices for extraction
            protocol_pairs_labels = [(i+1, j+1) for (i, j) in self.protocol_pairs]

            # Use extract_enhanced_location_features from location_encoder_enhanced.py
            enhanced_features = extract_enhanced_location_features(
                segmentation=split_data,
                protocol_pairs=protocol_pairs_labels,
                use_probability=False,
                prob_threshold=0.5,
                spacing=spacing
            )  # (K, 7) - [rel_pos(3), adj_ratio(1), adj_dir(3)]

            return enhanced_features.astype(np.float32)

        except Exception as e:
            warnings.warn(f"Failed to load {ann_path}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        perturbed, original = self.samples[idx]

        return {
            'perturbed': torch.from_numpy(perturbed).float(),  # (K, 7)
            'original': torch.from_numpy(original).float()     # (K, 7)
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
) -> float:
    """Train for one epoch"""
    encoder.train()
    decoder.train()

    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        perturbed = batch['perturbed'].to(device)  # (B, K, 7)
        original = batch['original'].to(device)    # (B, K, 7)

        # Forward pass: encode perturbed, decode to reconstruct original
        embedding = encoder(perturbed)               # (B, d_r)
        reconstructed = decoder(embedding)           # (B, K, 7)

        # Compute reconstruction loss
        loss = criterion(reconstructed, original)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Validate the model"""
    encoder.eval()
    decoder.eval()

    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Validating"):
        perturbed = batch['perturbed'].to(device)
        original = batch['original'].to(device)

        embedding = encoder(perturbed)
        reconstructed = decoder(embedding)

        loss = criterion(reconstructed, original)
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def extract_gaussian_prior(
    annotation_paths: List[Path],
    labels: List[str],
    protocol_pairs: List[Tuple[int, int]],
    encoder: nn.Module,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Gaussian prior (μ_R, Σ_R) from annotations using trained encoder.

    Returns:
        mu_R: (d_r,) mean vector
        Sigma_R: (d_r, d_r) covariance matrix
    """
    encoder.eval()
    all_embeddings = []

    print("\n" + "="*60)
    print("Extracting Location Prior (μ_R, Σ_R)")
    print("="*60)

    for ann_path in tqdm(annotation_paths, desc="Encoding relations"):
        try:
            nii = nib.load(str(ann_path))
            nii = nib.as_closest_canonical(nii)
            data = nii.get_fdata().astype(np.int32)
            affine = nii.affine
            spacing=nii.header['pixdim'][1:4]
            # spacing=[1.0,1.0,1.0]

            # Split L-R
            split_data = split_bilateral_labels(data, affine)

            # Extract enhanced features
            protocol_pairs_labels = [(i+1, j+1) for (i, j) in protocol_pairs]
            enhanced_features = extract_enhanced_location_features(
                segmentation=split_data,
                protocol_pairs=protocol_pairs_labels,
                use_probability=False,
                spacing=spacing
            )  # (K, 7)

            features_tensor = torch.from_numpy(enhanced_features).unsqueeze(0).to(device)  # (1, K, 7)
            print(f"features_tensor dtype: {features_tensor.dtype}")

            features_tensor = features_tensor.float()

            # Encode
            embedding = encoder(features_tensor)  # (1, d_r)
            all_embeddings.append(embedding.cpu().numpy())

        except Exception as e:
            warnings.warn(f"Failed to process {ann_path}: {e}")
            # import traceback
            import traceback
            traceback.print_exc()
            # print(f"Error processing file: {ann_path}")

    if len(all_embeddings) == 0:
        raise ValueError("No embeddings extracted! Check your annotation files.")

    all_embeddings = np.vstack(all_embeddings)  # (N_subjects, d_r)

    # Compute Gaussian parameters
    mu_R = np.mean(all_embeddings, axis=0)       # (d_r,)
    Sigma_R = np.cov(all_embeddings, rowvar=False)  # (d_r, d_r)

    print(f"\nExtracted Gaussian prior from {len(all_embeddings)} subjects")
    print(f"  μ_R shape: {mu_R.shape}")
    print(f"  Σ_R shape: {Sigma_R.shape}")
    print(f"  Embedding mean norm: {np.linalg.norm(mu_R):.4f}")
    print(f"  Covariance trace: {np.trace(Sigma_R):.4f}")

    return mu_R, Sigma_R


def main():
    parser = argparse.ArgumentParser(description='Pretrain Enhanced LocDisc Encoder with L-R Split')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing *_seg.nii.gz files (bilateral labels 1-7)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints and prior')
    parser.add_argument('--d_r', type=int, default=64,
                        help='Relation embedding dimension')
    parser.add_argument('--n_perturbations', type=int, default=100,
                        help='Number of noise perturbations per subject')
    parser.add_argument('--noise_std', type=float, default=0.1,
                        help='Standard deviation of perturbation noise')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--val_only', action='store_true',
                        help='If set, only perform validation (no training)')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get 14 labels and protocol pairs (after L-R split)
    labels, protocol_pairs = get_enhanced_protocol_pairs()
    K = len(protocol_pairs)

    print(f"\nTraining with {len(labels)} labels (L-R split)")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save protocol pairs
    protocol_file = os.path.join(args.output_dir, 'protocol_pairs_enhanced.npy')
    np.save(protocol_file, np.array(protocol_pairs))
    print(f"Saved protocol pairs to: {protocol_file}")

            # Build encoder and decoder
    print("\n" + "="*60)
    print("Building Enhanced Relation Encoder/Decoder")
    print("="*60)
    encoder = EnhancedRelationEncoder(K=K, d_r=args.d_r).to(device)
    decoder = EnhancedRelationDecoder(K=K, d_r=args.d_r).to(device)
    if not args.val_only:
        # Build dataset
        print("\n" + "="*60)
        print("Building Dataset")
        print("="*60)
        full_dataset = EnhancedRelationDataset(
            data_dir=args.data_dir,
            labels=labels,
            protocol_pairs=protocol_pairs,
            n_perturbations=args.n_perturbations,
            noise_std=args.noise_std,
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



        # Count parameters
        n_params_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        n_params_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"Encoder parameters: {n_params_enc:,}")
        print(f"Decoder parameters: {n_params_dec:,}")
        print(f"Total trainable parameters: {n_params_enc + n_params_dec:,}")

        # Loss and optimizer
        criterion = nn.MSELoss()
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
            train_loss = train_epoch(
                encoder, decoder, train_loader, criterion, optimizer, device, epoch
            )
            print(f"Train Loss: {train_loss:.6f}")

            # Validate
            val_loss = validate(encoder, decoder, val_loader, criterion, device)
            print(f"Val Loss:   {val_loss:.6f}")

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
                'train_loss': train_loss,
                'val_loss': val_loss,
                'protocol_pairs': protocol_pairs,
                'K': K,
                'd_r': args.d_r,
                'labels': labels,
                'args': vars(args)
            }

            # Save latest
            latest_path = os.path.join(args.output_dir, 'enhanced_loc_encoder_latest.pth')
            torch.save(checkpoint, latest_path)

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.output_dir, 'enhanced_loc_encoder_best.pth')
                torch.save(checkpoint, best_path)
                print(f"✓ Saved best model (val_loss={val_loss:.6f})")

            # Save periodic checkpoints
            if epoch % 10 == 0:
                periodic_path = os.path.join(args.output_dir, f'enhanced_loc_encoder_epoch{epoch}.pth')
                torch.save(checkpoint, periodic_path)

        print("\n" + "="*60)
        print("Training Completed!")
        print("="*60)
        print(f"Best validation loss: {best_val_loss:.6f}")

        # Extract Gaussian prior (μ_R, Σ_R)
        print("\n" + "="*60)
        print("Extracting Location Prior")
        print("="*60)

    # Load best model
    best_path = os.path.join(args.output_dir, 'enhanced_loc_encoder_best.pth')
    best_checkpoint = torch.load(best_path)
    encoder.load_state_dict(best_checkpoint['encoder_state_dict'])

    # Get annotation files
    annotation_files = list(Path(args.data_dir).glob('*_seg.nii.gz'))

    # Extract prior
    mu_R, Sigma_R = extract_gaussian_prior(
        annotation_files, labels, protocol_pairs, encoder, device
    )

    # Save prior
    prior_data = {
        'mu_R': mu_R,
        'Sigma_R': Sigma_R,
        'protocol_pairs': protocol_pairs,
        'd_r': args.d_r,
        'labels': labels,
        'K': K
    }
    prior_path = os.path.join(args.output_dir, 'enhanced_location_prior.npz')
    np.savez(prior_path, **prior_data)
    print(f"\n✓ Saved enhanced location prior to: {prior_path}")

    print("\n" + "="*60)
    print("All artifacts saved:")
    print("="*60)
    print(f"  1. Best encoder:      {best_path}")
    print(f"  2. Latest encoder:    {latest_path}")
    print(f"  3. Location prior:    {prior_path}")
    print(f"  4. Protocol pairs:    {protocol_file}")
    print("="*60)
    print("\nNext steps:")
    print("  1. Use the pretrained enhanced encoder in training")
    print("  2. Load enhanced location prior (μ_R, Σ_R)")
    print("  3. Update ProtoAtlasMSELossAdapter to use enhanced features")
    print("="*60)


if __name__ == '__main__':
    main()