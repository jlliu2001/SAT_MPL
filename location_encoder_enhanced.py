#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Location Encoder with Adjacency Information

Extends the basic centroid-based relation encoding with:
1. Adjacency detection (binary: adjacent or not)
2. Adjacency degree (ratio of adjacent voxels to boundary voxels)
3. Adjacency direction (vector from adjacency centroid to region centroid)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion
from typing import Tuple, List


def compute_adjacency_features(
    mask_i: np.ndarray,
    mask_j: np.ndarray,
    dilation_radius: int = 1
) -> Tuple[float, np.ndarray]:
    """
    计算两个mask之间的邻接特征

    Args:
        mask_i: Binary mask for region i (D, H, W)
        mask_j: Binary mask for region j (D, H, W)
        dilation_radius: Dilation radius for adjacency detection

    Returns:
        adjacency_ratio: 邻接体素占边界体素的比例 [0, 1]
        adjacency_direction: 从邻接区域质心指向脑区质心的向量 (3,)
    """
    # 1. 计算mask_i的边界（与其他区域或边界相邻的体素）
    eroded_i = binary_erosion(mask_i, iterations=1)
    boundary_i = mask_i & ~eroded_i  # 边界体素
    boundary_i_count = boundary_i.sum()

    if boundary_i_count == 0:
        return 0.0, np.array([0.0, 0.0, 0.0])

    # 2. 膨胀mask_j以检测邻接
    dilated_j = binary_dilation(mask_j, iterations=dilation_radius)

    # 3. 计算邻接区域：mask_i的边界与膨胀后的mask_j的交集
    adjacent_voxels = boundary_i & dilated_j
    adjacent_count = adjacent_voxels.sum()

    # 4. 邻接比例
    adjacency_ratio = float(adjacent_count) / float(boundary_i_count)

    # 5. 邻接方向向量
    if adjacent_count > 0:
        # 邻接体素的质心
        adjacent_coords = np.argwhere(adjacent_voxels)
        adjacency_centroid = adjacent_coords.mean(axis=0)  # (3,)

        # mask_i的质心
        region_coords = np.argwhere(mask_i)
        region_centroid = region_coords.mean(axis=0)  # (3,)

        # 指向向量：从邻接质心指向区域质心
        adjacency_direction = region_centroid - adjacency_centroid
    else:
        adjacency_direction = np.array([0.0, 0.0, 0.0])

    return adjacency_ratio, adjacency_direction


def extract_enhanced_location_features(
    segmentation: np.ndarray,  # (D, H, W) with label values
    protocol_pairs: List[Tuple[int, int]],  # List of (label_i, label_j) pairs
    use_probability: bool = False,  # If True, segmentation is probability map
    prob_threshold: float = 0.5
) -> np.ndarray:
    """
    提取增强的位置特征向量

    Args:
        segmentation: Segmentation mask (hard labels) or probability map
        protocol_pairs: List of protocol pairs (label_i, label_j)
        use_probability: Whether input is probability map
        prob_threshold: Threshold for probability map binarization

    Returns:
        feature_vector: (K, 7) array where K is number of protocol pairs
                        Each row: [relative_pos (3), adj_ratio (1), adj_direction (3)]
    """
    features = []

    for (label_i, label_j) in protocol_pairs:
        # Extract masks for label_i and label_j
        if use_probability:
            # For probability maps: (C, D, H, W)
            mask_i = (segmentation[label_i] > prob_threshold).astype(np.float32)
            mask_j = (segmentation[label_j] > prob_threshold).astype(np.float32)
        else:
            # For hard labels: (D, H, W)
            mask_i = (segmentation == label_i).astype(np.float32)
            mask_j = (segmentation == label_j).astype(np.float32)

        # 1. Relative position (centroid difference)
        coords_i = np.argwhere(mask_i > 0.5)
        coords_j = np.argwhere(mask_j > 0.5)

        if len(coords_i) == 0 or len(coords_j) == 0:
            # If either region is empty, use zero features
            features.append(np.zeros(7, dtype=np.float32))
            continue

        centroid_i = coords_i.mean(axis=0)  # (3,)
        centroid_j = coords_j.mean(axis=0)  # (3,)
        relative_pos = centroid_i - centroid_j  # (3,)

        # 2. Adjacency features
        adjacency_ratio, adjacency_direction = compute_adjacency_features(
            mask_i.astype(bool), mask_j.astype(bool), dilation_radius=1
        )

        # 3. Concatenate: [relative_pos (3), adj_ratio (1), adj_direction (3)]
        feature = np.concatenate([
            relative_pos,
            [adjacency_ratio],
            adjacency_direction
        ])  # (7,)

        features.append(feature)

    return np.array(features)  # (K, 7)


def extract_enhanced_location_features_from_probmap(
    prob_map: np.ndarray,  # (C, D, H, W) probability map
    protocol_pairs: List[Tuple[int, int]],
    prob_threshold: float = 0.5,
    spacing: Tuple[float, float, float] = None
) -> np.ndarray:
    """
    从概率图中提取增强的位置特征（用于分割loss计算）

    Args:
        prob_map: (C, D, H, W) probability map after sigmoid
        protocol_pairs: List of protocol pairs (label_idx_i, label_idx_j)
        prob_threshold: Threshold for soft binarization
        spacing: Voxel spacing (z, y, x) in mm. If provided, multiplies coordinates to get mm units

    Returns:
        feature_vector: (K, 7) array
    """
    features = []

    for (label_i, label_j) in protocol_pairs:
        prob_i = prob_map[label_i]  # (D, H, W)
        prob_j = prob_map[label_j]  # (D, H, W)

        # 1. Compute weighted centroid (using probabilities)
        coords = np.indices(prob_i.shape).transpose(1, 2, 3, 0)  # (D, H, W, 3)

        # Apply spacing if provided (convert voxel coordinates to mm)
        if spacing is not None:
            spacing_array = np.array(spacing, dtype=np.float32)  # (3,)
            coords = coords * spacing_array  # (D, H, W, 3)

        # Weighted centroid for region i
        weight_i = prob_i / (prob_i.sum() + 1e-8)
        centroid_i = (coords * weight_i[..., None]).sum(axis=(0, 1, 2))  # (3,)

        # Weighted centroid for region j
        weight_j = prob_j / (prob_j.sum() + 1e-8)
        centroid_j = (coords * weight_j[..., None]).sum(axis=(0, 1, 2))  # (3,)

        relative_pos = centroid_i - centroid_j

        # 2. Soft adjacency features (based on probability overlap)
        # Binarize with threshold for adjacency computation
        mask_i = (prob_i > prob_threshold).astype(np.float32)
        mask_j = (prob_j > prob_threshold).astype(np.float32)

        adjacency_ratio, adjacency_direction = compute_adjacency_features(
            mask_i.astype(bool), mask_j.astype(bool), dilation_radius=1
        )

        # Apply spacing to adjacency direction if provided
        if spacing is not None:
            adjacency_direction = adjacency_direction * spacing_array

        # 3. Concatenate
        feature = np.concatenate([
            relative_pos,
            [adjacency_ratio],
            adjacency_direction
        ])

        features.append(feature)

    return np.array(features)


# ---------------------------
# Enhanced Relation Encoder
# ---------------------------
class EnhancedRelationEncoder(nn.Module):
    """
    Enhanced MLP encoder for location features with adjacency information.

    Input: Stacked enhanced feature vectors for K protocol pairs
           Shape: (batch, K, 7) where each feature is:
                  [relative_pos (3), adjacency_ratio (1), adjacency_direction (3)]
    Output: Location embedding u ∈ R^d_r
    """
    def __init__(self, K: int, d_r: int = 64):
        super().__init__()
        self.K = K
        self.d_r = d_r

        # Process each enhanced relation vector independently
        self.relation_mlp = nn.Sequential(
            nn.Linear(7, 32),  # 7 = 3 (rel_pos) + 1 (adj_ratio) + 3 (adj_dir)
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        # Aggregate all K relations
        self.aggregate = nn.Sequential(
            nn.Linear(K * 64, 128),
            nn.ReLU(),
            nn.Linear(128, d_r)
        )

    def forward(self, relation_vectors):
        """
        Args:
            relation_vectors: (batch, K, 7) enhanced relation vectors

        Returns:
            embeddings: (batch, d_r) L2-normalized embeddings
        """
        batch_size = relation_vectors.size(0)

        # Process each relation independently
        rel_features = []
        for k in range(self.K):
            # print(relation_vectors[:, k, :].shape)
            feat = self.relation_mlp(relation_vectors[:, k, :])  # (batch, 64)
            rel_features.append(feat)

        # Concatenate and aggregate
        all_feats = torch.cat(rel_features, dim=1)  # (batch, K*64)
        embedding = self.aggregate(all_feats)  # (batch, d_r)

        # L2 normalize
        return F.normalize(embedding, dim=-1)


class EnhancedRelationDecoder(nn.Module):
    """
    Decoder to reconstruct enhanced relation vectors from embeddings.

    Input: Embedding u ∈ R^d_r
    Output: Reconstructed relation vectors (batch, K, 7)
    """
    def __init__(self, K: int, d_r: int = 64):
        super().__init__()
        self.K = K
        self.d_r = d_r

        # Decode from embedding to relation features
        self.decode = nn.Sequential(
            nn.Linear(d_r, 128),
            nn.ReLU(),
            nn.Linear(128, K * 64),
            nn.ReLU()
        )

        # Reconstruct each enhanced relation vector
        self.relation_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)  # 7-dimensional output
        )

    def forward(self, embedding):
        """
        Args:
            embedding: (batch, d_r)

        Returns:
            relation_vectors: (batch, K, 7)
        """
        batch_size = embedding.size(0)

        # Decode to relation features
        features = self.decode(embedding)  # (batch, K*64)
        features = features.view(batch_size, self.K, 64)  # (batch, K, 64)

        # Reconstruct each relation
        reconstructed = []
        for k in range(self.K):
            rel_vec = self.relation_head(features[:, k, :])  # (batch, 7)
            reconstructed.append(rel_vec)

        return torch.stack(reconstructed, dim=1)  # (batch, K, 7)


if __name__ == '__main__':
    # Test adjacency feature extraction
    print("Testing enhanced location feature extraction...")

    # Create dummy segmentation
    seg = np.zeros((96, 96, 96), dtype=np.int32)
    seg[20:40, 20:40, 20:40] = 1  # Region 1
    seg[35:55, 20:40, 20:40] = 2  # Region 2 (adjacent to 1)
    seg[60:80, 60:80, 60:80] = 3  # Region 3 (not adjacent)

    protocol_pairs = [(1, 2), (1, 3), (2, 3)]

    features = extract_enhanced_location_features(seg, protocol_pairs)
    print(f"Feature shape: {features.shape}")
    print(f"Features:\n{features}")

    # Test encoder
    K = len(protocol_pairs)
    encoder = EnhancedRelationEncoder(K=K, d_r=64)
    decoder = EnhancedRelationDecoder(K=K, d_r=64)

    # Dummy input
    batch_features = torch.from_numpy(features).unsqueeze(0).float()  # (1, K, 7)

    embedding = encoder(batch_features)
    reconstructed = decoder(embedding)

    print(f"\nEmbedding shape: {embedding.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Reconstruction error: {F.mse_loss(reconstructed, batch_features).item():.6f}")