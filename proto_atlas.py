# proto_atlas.py
"""
Proto-Atlas prior + discriminators + loss + evaluator (PyTorch).

Classes:
- SE3Encoder (tries to use e3nn; fallback 3D CNN)
- ShapePrior
- LocPrior
- ShapeDisc
- LocDisc
- ProtoAtlasLoss
- ProtoAtlasEvaluator

Example usage at bottom.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback

abnormal=False
# abnormal=True

# ---------------------------
# SE(3)-equivariant encoder using EquiTrack RXFM_Net
# ---------------------------
class SE3Encoder(nn.Module):
    """
    SE(3)-equivariant encoder using EquiTrack's RXFM_Net.

    When use_e3nn=True, uses the real SE(3)-equivariant network from EquiTrack.
    Otherwise, falls back to a standard 3D CNN.

    Input: signed-distance map or probability mask: (B, 1, D, H, W)
    Output: embedding (B, emb_dim)
    """
    def __init__(self, emb_dim: int = 128, use_e3nn: bool = False):
        super().__init__()
        self.emb_dim = emb_dim
        self._use_e3nn = use_e3nn

        if use_e3nn:
            try:
                # Import EquiTrack's SE(3)-equivariant network
                import sys
                import os

                # Try to import from EquiTrack repository
                equitrack_path = "/data0/user/jlliu/git_pull_repos/SAT_MPL/EquiTrack"
                if os.path.exists(equitrack_path):
                    sys.path.insert(0, equitrack_path)

                from equitrack.networks import RXFM_Net

                # Build RXFM_Net with desired output channels
                self.se3_net = RXFM_Net(output_chans=emb_dim)
                self.se3_net.eval()
                self.se3_net = self.se3_net.float()
                for module in self.se3_net.modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        module.eval()
                        module.track_running_stats = True  # 确保使用 running stats


                # Add global pooling and projection to get fixed-size embedding
                self.global_pool = nn.AdaptiveAvgPool3d(1)
                self.proj = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(emb_dim, emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, emb_dim)
                )

                print(f"✓ Successfully loaded EquiTrack SE(3)-equivariant encoder (RXFM_Net)")
                print(f"  Output channels: {emb_dim}")
                print(f"  This encoder is truly SE(3)-equivariant!")

            except ImportError as e:
                warnings.warn(f"Failed to import EquiTrack RXFM_Net: {e}. "
                              "Make sure EquiTrack repository is in parent directory. "
                              "Falling back to 3D CNN encoder.")
                print(traceback.format_exc())
                self._use_e3nn = False
            except Exception as e:
                warnings.warn(f"Error building SE(3) encoder: {e}. "
                              "Falling back to 3D CNN encoder.")
                self._use_e3nn = False

        if not self._use_e3nn:
            # Fallback 3D CNN encoder (works robustly but not SE(3)-equivariant)
            print(f"Using fallback 3D CNN encoder (not SE(3)-equivariant)")
            # NOTE: Using InstanceNorm3d with track_running_stats=True for stability
            # This allows using running statistics in eval mode, avoiding NaN with batch_size=1
            self.encoder = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=3, padding=1),
                nn.InstanceNorm3d(16, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.1),
                nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # downsample
                nn.InstanceNorm3d(32, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.1),
                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.InstanceNorm3d(64, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.1),
                nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(128, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.1),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(128, emb_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, D, H, W)
        returns: (B, emb_dim)
        """
        if self._use_e3nn:
            # SE(3)-equivariant forward pass using RXFM_Net
            features = self.se3_net(x)  # (B, emb_dim, D, H, W)
            pooled = self.global_pool(features)  # (B, emb_dim, 1, 1, 1)
            embedding = self.proj(pooled)  # (B, emb_dim)
            return embedding
        else:
            return self.encoder(x)


# ---------------------------
# Utilities
# ---------------------------
def to_device(t, device):
    if isinstance(t, torch.Tensor):
        return t.to(device)
    elif isinstance(t, np.ndarray):
        return torch.from_numpy(t).to(device)
    else:
        return t

def safe_inverse_cov(cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # cov: (..., d, d)
    # returns pseudo-inverse with damping
    cov = cov.clone()
    eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype).expand(cov.shape)
    cov = cov + eps * eye
    inv = torch.linalg.inv(cov)
    return inv

def mahalanobis_squared(x: torch.Tensor, mu: torch.Tensor, inv_cov: torch.Tensor) -> torch.Tensor:
    # x, mu : (B, d)
    # inv_cov: (d, d) or (B, d, d)
    diff = (x - mu).unsqueeze(-2)  # (B, 1, d)
    if inv_cov.dim() == 2:
        m = torch.matmul(diff, inv_cov)  # (B, 1, d)
        m = torch.matmul(m, diff.transpose(-1, -2)).squeeze(-1).squeeze(-1)  # (B,)
    else:
        # per-sample cov
        m = torch.matmul(diff, inv_cov)  # (B, 1, d)
        m = torch.matmul(m, diff.transpose(-1, -2)).squeeze(-1).squeeze(-1)
    return m  # (B,)

# ---------------------------
# ShapePrior: stores per-label Gaussian priors (following shape_plan.md)
# ---------------------------
class ShapePrior:
    """
    Holds per-label Gaussian priors (μ_l, Σ_l) in shape embedding space.

    Following shape_plan.md design:
    - Each label l has its own Gaussian distribution (μ_l, Σ_l)
    - Used for Mahalanobis distance scoring
    - Allows per-label shape variation modeling

    Attributes:
        labels: List of label names
        emb_dim: Embedding dimension
        gaussian: Dict[label -> (μ_l, Σ_l, Σ_l^(-1))]
    """
    def __init__(self,
                 labels: List[str],
                 emb_dim: int,
                 gaussian: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None):
        """
        Args:
            labels: List of label names (e.g., ["hippocampus", "amygdala", ...])
            emb_dim: Embedding dimension
            gaussian: Optional dict {label: (μ_l, Σ_l)} where
                      μ_l is (emb_dim,) mean vector
                      Σ_l is (emb_dim, emb_dim) covariance matrix
        """
        self.labels = labels
        self.emb_dim = emb_dim
        self.gaussian = {}

        if gaussian:
            for label, (mean, cov) in gaussian.items():
                mean = np.asarray(mean).astype(np.float32)
                cov = np.asarray(cov).astype(np.float32)
                assert mean.shape == (emb_dim,), f"Mean shape mismatch for {label}"
                assert cov.shape == (emb_dim, emb_dim), f"Cov shape mismatch for {label}"

                # Store mean, cov, and inverse covariance
                mu_tensor = torch.tensor(mean, dtype=torch.float32)
                Sigma_tensor = torch.tensor(cov, dtype=torch.float32)
                Sigma_inv = self._safe_inverse(Sigma_tensor)

                self.gaussian[label] = {
                    'mu': mu_tensor,
                    'Sigma': Sigma_tensor,
                    'Sigma_inv': Sigma_inv
                }

    def _safe_inverse(self, cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Compute inverse covariance with regularization for numerical stability"""
        cov_reg = cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        return torch.linalg.inv(cov_reg)

    def make_spd_inverse(self,label, eps_rel=1e-6, eps_abs=1e-12):
        """
        输入:
        Sigma_inv: (d,d) 或 (d,d) 的 torch.tensor，可能非对称或非正定
        eps_rel: 相对截断因子（相对于最大特征值）
        eps_abs: 绝对最小截断值
        返回:
        Sigma_inv_reg: 对称且正定的逆协方差矩阵
        """
        # 使用 double 提高数值稳定性
        Sigma_inv = self.gaussian[label]['Sigma_inv']  # (emb_dim, emb_dim)
        A = Sigma_inv.to(dtype=torch.float64).clone()

        # 对称化
        A = 0.5 * (A + A.t())

        # 特征分解（对称矩阵用 eigh）
        eigvals, eigvecs = torch.linalg.eigh(A)  # eigvals 升序

        # 选取截断阈值：max(eps_abs, eps_rel * max_eig)
        max_eig = eigvals.max().clamp_min(0.0)
        eps = max(eps_abs, (eps_rel * max_eig).item())

        # clip 特征值（保证严格 > 0）
        eigvals_clipped = torch.clamp(eigvals, min=eps)

        # 重构矩阵
        A_reg = (eigvecs * eigvals_clipped.unsqueeze(0)) @ eigvecs.t()

        # 确保返回 float32 与原始 device 一致（可按需保留 float64）
        return A_reg.to(dtype=Sigma_inv.dtype)

    def mahalanobis_distance(self, embedding: torch.Tensor, label: str) -> torch.Tensor:
        """
        Compute Mahalanobis distance from embedding to label's Gaussian prior.

        d_l = sqrt((v - μ_l)^T Σ_l^(-1) (v - μ_l))

        Args:
            embedding: (batch, emb_dim) or (emb_dim,)
            label: Label name

        Returns:
            distances: (batch,) or scalar
        """
        if label not in self.gaussian:
            raise ValueError(f"No Gaussian prior for label '{label}'")

        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)  # (1, emb_dim)

        mu = self.gaussian[label]['mu'].to(embedding.device)          # (emb_dim,)
        # mu = F.normalize(mu, p=2, dim=-1)
        print(f'mu for {label}: {mu}')
        # Sigma_inv = self.gaussian[label]['Sigma_inv'].to(embedding.device)  # (emb_dim, emb_dim)
        Sigma_inv=self.make_spd_inverse(label).to(embedding.device)
        print(f'Sigma_inv for {label}: {Sigma_inv}')
        # eigvals = torch.linalg.eigvalsh(Sigma_inv)
        # print("min eigenvalue:", eigvals.min())
        # eps = 1e-3
        # Sigma = Sigma_inv + eps * torch.eye(Sigma_inv.size(0), device=Sigma_inv.device)
        # Sigma_inv = torch.inverse(Sigma)
        eigvals = torch.linalg.eigvalsh(Sigma_inv)
        print("min eigenvalue:", eigvals.min())

        diff = embedding - mu  # (batch, emb_dim)

        # D^2 = (v - μ)^T Σ^-1 (v - μ)
        # mahal_sq = torch.sum(diff @ Sigma_inv * diff, dim=1)  # (batch,)
        mahal_sq = torch.einsum('bi,ij,bj->b', diff, Sigma_inv, diff)

#        Clamp small negative values to zero
        mahal_sq = mahal_sq.clamp(min=0.0)
        print('mahal_sq:',mahal_sq)

        return torch.sqrt(mahal_sq + 1e-12)  # Add epsilon for stability

    def cosine_similarity(self, embedding: torch.Tensor, label: str) -> torch.Tensor:
        """
        Compute cosine similarity from embedding to label's mean prototype.

        This method is more robust than Mahalanobis distance when:
        - Training samples are limited
        - Covariance matrix is ill-conditioned
        - Embedding space is high-dimensional

        similarity = (v · μ_l) / (||v|| * ||μ_l||)

        Args:
            embedding: (batch, emb_dim) or (emb_dim,)
            label: Label name

        Returns:
            similarity: (batch,) or scalar in range [0, 1]
                       (normalized from [-1, 1] to [0, 1])
        """
        if label not in self.gaussian:
            raise ValueError(f"No Gaussian prior for label '{label}'")

        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)  # (1, emb_dim)

        # Get mean prototype for this label
        mu = self.gaussian[label]['mu'].to(embedding.device)  # (emb_dim,)

        # L2 normalize both embedding and prototype
        embedding_norm = F.normalize(embedding, p=2, dim=-1)  # (batch, emb_dim)
        mu_norm = F.normalize(mu, p=2, dim=-1)  # (emb_dim,)

        # Compute cosine similarity: dot product of normalized vectors
        # Result is in [-1, 1]
        sim = torch.sum(embedding_norm * mu_norm, dim=-1)  # (batch,)

        # Convert from [-1, 1] to [0, 1] for consistency with score range
        # sim = 1 means perfect match (score = 1)
        # sim = -1 means opposite direction (score = 0)
        score = (sim + 1.0) / 2.0  # (batch,)

        return score

    def to(self, device):
        """Move all tensors to device"""
        for label in self.gaussian:
            self.gaussian[label]['mu'] = self.gaussian[label]['mu'].to(device)
            self.gaussian[label]['Sigma'] = self.gaussian[label]['Sigma'].to(device)
            self.gaussian[label]['Sigma_inv'] = self.gaussian[label]['Sigma_inv'].to(device)
        return self

    def get_labels(self):
        return self.labels

    def has_gaussian(self, label: str) -> bool:
        return label in self.gaussian

    def add_gaussian(self, label: str, mean: np.ndarray, cov: np.ndarray):
        """Add Gaussian prior for a label"""
        mean = np.asarray(mean).astype(np.float32)
        cov = np.asarray(cov).astype(np.float32)

        mu_tensor = torch.tensor(mean, dtype=torch.float32)
        Sigma_tensor = torch.tensor(cov, dtype=torch.float32)
        Sigma_inv = self._safe_inverse(Sigma_tensor)

        self.gaussian[label] = {
            'mu': mu_tensor,
            'Sigma': Sigma_tensor,
            'Sigma_inv': Sigma_inv
        }

    @staticmethod
    def load_from_file(prior_path: str) -> 'ShapePrior':
        """
        Load ShapePrior from saved .npz file.

        Args:
            prior_path: Path to shape_priors.npz file

        Returns:
            ShapePrior instance
        """
        data = np.load(prior_path, allow_pickle=True)

        labels = data['labels'].tolist()
        emb_dim = int(data['emb_dim'])

        # Reconstruct gaussian dict
        gaussian = {}
        for label in labels:
            mu_key = f'{label}_mu'
            sigma_key = f'{label}_Sigma'
            if mu_key in data and sigma_key in data:
                gaussian[label] = (data[mu_key], data[sigma_key])

        return ShapePrior(labels=labels, emb_dim=emb_dim, gaussian=gaussian)

# ---------------------------
# Utility functions for extracting prototypes from real annotations
# ---------------------------
def load_nifti_annotation(nifti_path: str) -> np.ndarray:
    """
    Load a NIfTI annotation file and reorient to RAS.
    Returns: 3D numpy array with integer labels in RAS orientation
    """
    try:
        import nibabel as nib
        nii = nib.load(nifti_path)

        # ✅ CRITICAL: Reorient to RAS (canonical orientation)
        # This ensures consistent orientation across all subjects
        nii = nib.as_closest_canonical(nii)

        data = nii.get_fdata().astype(np.int32)
        return data
    except ImportError:
        raise ImportError("nibabel is required to load NIfTI files. Install with: pip install nibabel")

def extract_prototypes_from_annotations(
    annotation_paths: List[str],
    labels: List[str],
    encoder: nn.Module,
    device: torch.device,
    method: str = 'kmeans',
    n_prototypes: int = 5,
    use_sdf: bool = False
) -> Dict[str, np.ndarray]:
    """
    Extract shape prototypes from a set of annotated images.

    Args:
        annotation_paths: List of paths to NIfTI annotation files
        labels: List of label names (e.g., ["hippocampus", "amygdala", ...])
                Label i corresponds to voxel value i+1 in the annotation
        encoder: The SE3Encoder to extract embeddings
        device: torch device
        method: 'kmeans' or 'mean' - how to extract prototypes
        n_prototypes: Number of prototypes per label (for kmeans)
        use_sdf: If True, convert binary masks to signed distance fields

    Returns:
        Dictionary mapping label names to prototype arrays (n_prototypes, emb_dim)
    """
    from sklearn.cluster import KMeans
    from scipy.ndimage import distance_transform_edt

    encoder.eval()
    all_embeddings = {label: [] for label in labels}

    with torch.no_grad():
        for ann_path in annotation_paths:
            # Load annotation
            ann_data = load_nifti_annotation(ann_path)

            # Process each label
            for label_idx, label_name in enumerate(labels):
                # Label index starts from 1 (0 is background)
                voxel_value = label_idx + 1
                binary_mask = (ann_data == voxel_value).astype(np.float32)

                if binary_mask.sum() == 0:
                    warnings.warn(f"No voxels found for label {label_name} (value {voxel_value}) in {ann_path}")
                    continue

                # Optionally convert to signed distance field
                if use_sdf:
                    # Compute signed distance field
                    pos_dist = distance_transform_edt(binary_mask)
                    neg_dist = distance_transform_edt(1 - binary_mask)
                    sdf = pos_dist - neg_dist
                    # Normalize
                    if sdf.std() > 0:
                        sdf = (sdf - sdf.mean()) / sdf.std()
                    mask_input = sdf
                else:
                    mask_input = binary_mask

                # Prepare input tensor (B, 1, D, H, W)
                mask_tensor = torch.from_numpy(mask_input).unsqueeze(0).unsqueeze(0).float().to(device)

                # Extract embedding
                emb = encoder(mask_tensor)  # (1, emb_dim)
                all_embeddings[label_name].append(emb.cpu().numpy())

    # Build prototypes
    prototypes = {}
    for label_name in labels:
        embs = all_embeddings[label_name]
        if len(embs) == 0:
            warnings.warn(f"No embeddings found for label {label_name}, using random prototypes")
            emb_dim = list(encoder.parameters())[0].shape[0] if len(list(encoder.parameters())) > 0 else 128
            prototypes[label_name] = np.random.randn(n_prototypes, emb_dim).astype(np.float32)
            continue

        embs = np.concatenate(embs, axis=0)  # (n_samples, emb_dim)

        if method == 'kmeans' and len(embs) >= n_prototypes:
            # Use KMeans to find prototypes
            kmeans = KMeans(n_clusters=n_prototypes, random_state=42, n_init=10)
            kmeans.fit(embs)
            prototypes[label_name] = kmeans.cluster_centers_.astype(np.float32)
        elif method == 'mean' or len(embs) < n_prototypes:
            # Use mean as single prototype, or all samples if too few
            if len(embs) < n_prototypes:
                # Use all available embeddings
                prototypes[label_name] = embs.astype(np.float32)
            else:
                # Compute mean and add some variations
                mean_emb = embs.mean(axis=0, keepdims=True)
                prototypes[label_name] = mean_emb.astype(np.float32)
        else:
            prototypes[label_name] = embs[:n_prototypes].astype(np.float32)

    return prototypes

def compute_centroid_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Compute centroid of a binary mask.
    Returns: (3,) array with (z, y, x) coordinates
    """
    if mask.sum() == 0:
        return np.array([0.0, 0.0, 0.0])

    coords = np.argwhere(mask > 0.5)  # (N, 3)
    centroid = coords.mean(axis=0)  # (3,)
    return centroid

def compute_relative_position(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """
    Compute relative position vector from c1 to c2.
    Returns: (3,) array
    """
    return c2 - c1

def extract_location_prototypes_from_annotations(
    annotation_paths: List[str],
    labels: List[str],
    relation_pairs: List[Tuple[int, int]],
    loc_encoder: nn.Module,
    device: torch.device,
    method: str = 'kmeans',
    n_prototypes: int = 10
) -> np.ndarray:
    """
    Extract location prototypes from annotated images.

    Args:
        annotation_paths: List of paths to NIfTI annotation files
        labels: List of label names
        relation_pairs: List of (i, j) pairs indicating relations to encode
        loc_encoder: The LocDisc encoder (MLP) to use
        device: torch device
        method: 'kmeans' or 'mean'
        n_prototypes: Number of prototypes

    Returns:
        Array of location prototypes (n_prototypes, emb_dim)
    """
    from sklearn.cluster import KMeans

    loc_encoder.eval()
    all_relation_embeddings = []

    with torch.no_grad():
        for ann_path in annotation_paths:
            # Load annotation
            ann_data = load_nifti_annotation(ann_path)

            # Compute centroids for all labels
            centroids = []
            for label_idx in range(len(labels)):
                voxel_value = label_idx + 1
                binary_mask = (ann_data == voxel_value).astype(np.float32)
                centroid = compute_centroid_from_mask(binary_mask)
                centroids.append(centroid)

            centroids = np.array(centroids)  # (L, 3)

            # Compute relation vectors
            relation_vectors = []
            for (i, j) in relation_pairs:
                rel_vec = centroids[i] - centroids[j]  # (3,)
                relation_vectors.append(rel_vec)

            relation_vectors = np.array(relation_vectors)  # (K, 3)
            rel_flat = relation_vectors.flatten()  # (K*3,)

            # Encode using the MLP encoder
            rel_tensor = torch.from_numpy(rel_flat).unsqueeze(0).float().to(device)  # (1, K*3)
            emb = loc_encoder(rel_tensor)  # (1, emb_dim)
            all_relation_embeddings.append(emb.cpu().numpy())

    if len(all_relation_embeddings) == 0:
        warnings.warn("No location embeddings extracted, using random prototypes")
        emb_dim = 128  # default
        return np.random.randn(n_prototypes, emb_dim).astype(np.float32)

    all_relation_embeddings = np.concatenate(all_relation_embeddings, axis=0)  # (n_samples, emb_dim)

    if method == 'kmeans' and len(all_relation_embeddings) >= n_prototypes:
        kmeans = KMeans(n_clusters=n_prototypes, random_state=42, n_init=10)
        kmeans.fit(all_relation_embeddings)
        prototypes = kmeans.cluster_centers_.astype(np.float32)
    elif method == 'mean' or len(all_relation_embeddings) < n_prototypes:
        if len(all_relation_embeddings) < n_prototypes:
            prototypes = all_relation_embeddings.astype(np.float32)
        else:
            mean_emb = all_relation_embeddings.mean(axis=0, keepdims=True)
            prototypes = mean_emb.astype(np.float32)
    else:
        prototypes = all_relation_embeddings[:n_prototypes].astype(np.float32)

    return prototypes

# ---------------------------
# LocPrior: relation prototypes or gaussian (following location_plan.md)
# ---------------------------
class LocPrior:
    """
    Stores relation protocol info and Gaussian prior (μ_R, Σ_R) for relation embeddings.

    Following location_plan.md design:
    - Protocol pairs R: anatomically meaningful (i,j) relations
    - Gaussian distribution (μ_R, Σ_R) in embedding space
    - Mahalanobis distance for scoring

    Attributes:
        relation_pairs: List of (i, j) protocol pairs
        emb_dim: Embedding dimension d_r
        mu_R: Mean vector (d_r,) in embedding space
        Sigma_R: Covariance matrix (d_r, d_r)
        Sigma_R_inv: Inverse covariance for Mahalanobis distance
    """
    def __init__(self,
                 relation_pairs: List[Tuple[int, int]],
                 emb_dim: int,
                 mu_R: Optional[np.ndarray] = None,
                 Sigma_R: Optional[np.ndarray] = None,
                 means: Optional[np.ndarray] = None,
                 stds: Optional[np.ndarray] = None):
        """
        Args:
            relation_pairs: List of (i, j) protocol pairs (anatomical relations)
            emb_dim: Embedding dimension d_r
            mu_R: Mean vector (d_r,), optional (will use zeros if None)
            Sigma_R: Covariance matrix (d_r, d_r), optional (will use identity if None)
            means: (K, 7) mean vectors for direct mode (each protocol pair's 7-dim feature mean)
            stds: (K, 7) std vectors for direct mode (each protocol pair's 7-dim feature std)
        """
        self.relation_pairs = relation_pairs
        self.emb_dim = emb_dim
        self.K = len(relation_pairs)

        # Initialize Gaussian parameters (for encoder-based mode)
        if mu_R is not None:
            self.mu_R = torch.tensor(mu_R.astype(np.float32))
        else:
            self.mu_R = torch.zeros(emb_dim, dtype=torch.float32)

        if Sigma_R is not None:
            self.Sigma_R = torch.tensor(Sigma_R.astype(np.float32))
        else:
            self.Sigma_R = torch.eye(emb_dim, dtype=torch.float32)

        # Compute inverse covariance for Mahalanobis distance
        self.Sigma_R_inv = self._safe_inverse(self.Sigma_R)

        # Initialize direct mode parameters
        if means is not None:
            self.means = torch.tensor(means.astype(np.float32))  # (K, 7)
        else:
            self.means = None

        if stds is not None:
            self.stds = torch.tensor(stds.astype(np.float32))  # (K, 7)
        else:
            self.stds = None

    def _safe_inverse(self, cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Compute inverse covariance with damping for numerical stability"""
        cov_reg = cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        return torch.linalg.inv(cov_reg)

    def mahalanobis_distance(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis distance: d_R = sqrt((u - μ_R)^T Σ_R^(-1) (u - μ_R))

        Args:
            embedding: (batch, d_r) or (d_r,)

        Returns:
            distances: (batch,) or scalar
        """
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)  # (1, d_r)

        diff = embedding - self.mu_R.to(embedding.device)  # (batch, d_r)
        print('self.mu_R:',self.mu_R)

        # D^2 = (u - μ)^T Σ^-1 (u - μ)
        mahal_sq = torch.sum(
            diff @ self.Sigma_R_inv.to(embedding.device) * diff,
            dim=1
        )  # (batch,)
        print('self.Sigma_R_inv:',self.Sigma_R_inv)

        return torch.sqrt(mahal_sq + 1e-8)  # Add small epsilon for stability

    def cosine_similarity(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity from embedding to location mean prototype.

        This method is more robust than Mahalanobis distance when:
        - Training samples are limited
        - Covariance matrix is ill-conditioned
        - Embedding space is high-dimensional

        similarity = (u · μ_R) / (||u|| * ||μ_R||)

        Args:
            embedding: (batch, d_r) or (d_r,)

        Returns:
            similarity: (batch,) or scalar in range [0, 1]
                       (normalized from [-1, 1] to [0, 1])
        """
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)  # (1, d_r)

        # Get mean prototype for location
        mu_R = self.mu_R.to(embedding.device)  # (d_r,)

        # L2 normalize both embedding and prototype
        embedding_norm = F.normalize(embedding, p=2, dim=-1)  # (batch, d_r)
        mu_R_norm = F.normalize(mu_R, p=2, dim=-1)  # (d_r,)

        # Compute cosine similarity: dot product of normalized vectors
        # Result is in [-1, 1]
        sim = torch.sum(embedding_norm * mu_R_norm, dim=-1)  # (batch,)

        # Convert from [-1, 1] to [0, 1] for consistency with score range
        # sim = 1 means perfect match (score = 1)
        # sim = -1 means opposite direction (score = 0)
        score = (sim + 1.0) / 2.0  # (batch,)

        return score

    def compute_direct_score_method_a(
        self,
        features: torch.Tensor,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        方案A：按类别对独立计算标准化距离，然后平均

        对每个类别对k:
          - 计算标准化距离: d_k = sqrt(sum((x_k - µ_k)² / σ_k²)) / sqrt(7)
          - 转换为得分: score_k = exp(-β * d_k)

        最终得分: location_score = mean(score_k)

        Args:
            features: (B, K, 7) location features
            beta: Scaling parameter for distance-to-score conversion

        Returns:
            scores: (B,) overall location scores
            per_pair_scores: (B, K) per-protocol-pair scores
        """
        if self.means is None or self.stds is None:
            raise ValueError("Direct mode requires means and stds. Use load_from_file with mean prior.")

        B, K, feat_dim = features.shape
        device = features.device

        means = self.means.to(device)  # (K, 7)
        stds = self.stds.to(device)    # (K, 7)

        # Compute standardized distance for each protocol pair
        # diff: (B, K, 7)
        diff = features - means.unsqueeze(0)  # (B, K, 7)
        print('diff:',diff)

        # Standardize by dividing by std
        standardized_diff = diff / (stds.unsqueeze(0) + 1e-8)  # (B, K, 7)

        # Compute normalized Euclidean distance for each pair
        # d_k = sqrt(sum((x_k - µ_k)² / σ_k²)) / sqrt(7)
        distances_sq = torch.sum(standardized_diff**2, dim=2)  # (B, K)
        distances = torch.sqrt(distances_sq + 1e-8) / np.sqrt(feat_dim)  # (B, K)
        
        


        # Convert to scores: score_k = exp(-β * d_k)
        # print('beta:',beta)
        per_pair_scores = torch.exp(-beta * distances)  # (B, K)

        # Average over all pairs
        # 取均值时排除0值
        # per_pair_scores: (B, K)
        # abnormal=False
        if abnormal:
            overall_scores = torch.mean(per_pair_scores, dim=1)  # (B,)
        else:
            mask = per_pair_scores != 0  # (B, K)
            # 为了处理每个batch，按行计算非零元素的均值，若全为0则结果为nan
            sum_scores = torch.sum(per_pair_scores * mask, dim=1)
            count_nonzero = torch.sum(mask, dim=1)
            # 避免除零错误，如果count_nonzero为0则赋值为1，结果会是0（此情况极罕见）
            count_nonzero_safe = torch.where(count_nonzero == 0, torch.ones_like(count_nonzero), count_nonzero)
            overall_scores = sum_scores / count_nonzero_safe  # (B,)

        return overall_scores, per_pair_scores

    def compute_direct_score_method_c(
        self,
        features: torch.Tensor,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        方案C：加权欧氏距离（简化版）

        计算整体欧氏距离：d = sqrt(sum_k ||x_k - µ_k||²) / sqrt(K*7)
        转换为得分：score = exp(-β * d)

        Args:
            features: (B, K, 7) location features
            beta: Scaling parameter

        Returns:
            scores: (B,) location scores
        """
        if self.means is None:
            raise ValueError("Direct mode requires means. Use load_from_file with mean prior.")

        B, K, feat_dim = features.shape
        device = features.device

        means = self.means.to(device)  # (K, 7)

        # Compute Euclidean distance
        diff = features - means.unsqueeze(0)  # (B, K, 7)
        distances_sq = torch.sum(diff ** 2, dim=(1, 2))  # (B,)
        distances = torch.sqrt(distances_sq + 1e-8) / np.sqrt(K * feat_dim)  # (B,)

        # Convert to scores
        scores = torch.exp(-beta * distances)  # (B,)

        return scores

    def to(self, device):
        """Move tensors to device"""
        self.mu_R = self.mu_R.to(device)
        self.Sigma_R = self.Sigma_R.to(device)
        self.Sigma_R_inv = self.Sigma_R_inv.to(device)

        if self.means is not None:
            self.means = self.means.to(device)
        if self.stds is not None:
            self.stds = self.stds.to(device)

        return self

    @staticmethod
    def load_from_file(prior_path: str) -> 'LocPrior':
        """
        Load LocPrior from saved .npz file.

        Supports two types of prior files:
        1. Encoder-based prior: contains mu_R, Sigma_R, d_r
        2. Direct (mean) prior: contains means, stds

        Args:
            prior_path: Path to location_prior.npz or location_mean_prior.npz file

        Returns:
            LocPrior instance
        """
        data = np.load(prior_path, allow_pickle=True)

        # Determine file type
        if 'mu_R' in data:
            # Encoder-based prior
            return LocPrior(
                relation_pairs=data['protocol_pairs'].tolist(),
                emb_dim=int(data['d_r']),
                mu_R=data['mu_R'],
                Sigma_R=data['Sigma_R']
            )
        elif 'means' in data:
            # Direct (mean) prior
            return LocPrior(
                relation_pairs=data['protocol_pairs'].tolist(),
                emb_dim=0,  # Not used in direct mode
                means=data['means'],
                stds=data.get('stds', None)  # stds might be optional for method C
            )
        else:
            raise ValueError(f"Unknown prior file format: {prior_path}")

# ---------------------------
# ShapeDisc: embedding and scoring (following shape_plan.md)
# ---------------------------
class ShapeDisc(nn.Module):
    """
    Shape discriminator following shape_plan.md design.

    Forward pass:
    1. Encode predicted mask with E_shape: M^l → v
    2. Compute Mahalanobis distance to label-specific prior: d_l = distance(v, μ_l, Σ_l)
    3. Score: S_shape = exp(-β * d_l)

    The encoder E_shape should be pretrained and frozen during segmentation training.
    """
    def __init__(self,
                 emb_dim: int,
                 shape_prior: ShapePrior,
                 beta: float = 1.0,
                 freeze_encoder: bool = True,
                 use_e3nn: bool = False):
        """
        Args:
            emb_dim: Embedding dimension
            shape_prior: ShapePrior with per-label (μ_l, Σ_l)
            beta: Scaling parameter for Mahalanobis distance
            freeze_encoder: Whether to freeze encoder weights
            use_e3nn: Whether to use SE(3)-equivariant encoder
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.beta = beta
        self.prior = shape_prior
        self.encoder = SE3Encoder(emb_dim=emb_dim, use_e3nn=use_e3nn)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, mask: torch.Tensor, label: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for shape discrimination.

        Args:
            mask: (B, 1, D, H, W) — predicted mask (soft probabilities or SDF)
            label: Label name (str)

        Returns:
            scores: (B,) shape scores S_shape ∈ [0, 1]
            embeddings: (B, emb_dim) shape embeddings
        """
        # Step 1: Encode with E_shape
        v = self.encoder(mask)  # (B, emb_dim)

        # Step 2: Compute Mahalanobis distance to label-specific prior
        d_l = self.prior.mahalanobis_distance(v, label)  # (B,)

        # Step 3: Score = exp(-β * d_l)
        scores = torch.exp(-self.beta * d_l)  # (B,)

        return scores, v

    def load_pretrained_encoder(self, checkpoint_path: str, device: torch.device):
        """
        Load pretrained encoder weights from checkpoint.

        Args:
            checkpoint_path: Path to pretrained encoder checkpoint (.pth file)
            device: Device to load weights to
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        try:
            # Try to load state dict directly
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=True)
            print(f"✓ Loaded pretrained shape encoder from: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️  Warning: Strict loading failed: {e}")
            print(f"   Attempting to load with strict=False (may miss some parameters)...")

            # Try non-strict loading
            missing_keys, unexpected_keys = self.encoder.load_state_dict(
                checkpoint['encoder_state_dict'], strict=False
            )

            if missing_keys:
                print(f"   Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"   Unexpected keys: {unexpected_keys}")

            # Check if running stats are missing
            if any('running_mean' in k or 'running_var' in k for k in missing_keys):
                print(f"\n⚠️  IMPORTANT: Running statistics missing in pretrained model!")
                print(f"   This is expected if the model was trained with InstanceNorm3d without track_running_stats.")
                print(f"   The encoder will work but may produce suboptimal results.")
                print(f"   Recommendation: Retrain the shape encoder with the updated code.")

        print(f"  Trained for {checkpoint.get('epoch', 'unknown')} epochs")
        if 'val_metrics' in checkpoint and 'loss' in checkpoint['val_metrics']:
            print(f"  Val loss: {checkpoint['val_metrics']['loss']:.4f}")

# ---------------------------
# LocDisc: relation encoder + scorer (following location_plan.md)
# ---------------------------
class LocDisc(nn.Module):
    """
    Location discriminator following location_plan.md design.

    Forward pass:
    1. Compute soft centroids from predicted masks: {M^l} → {c_l}
    2. Compute protocol relation vectors: r_ij = c_i - c_j for (i,j) ∈ R
    3. Encode with E_rel: {r_ij} → u ∈ R^d_r
    4. Score with Mahalanobis distance: S_loc = exp(-w * d_R(u, μ_R, Σ_R))

    The encoder E_rel should be pretrained and frozen during segmentation training.
    """
    def __init__(self,
                 num_labels: int,
                 relation_pairs: List[Tuple[int, int]],
                 emb_dim: int,
                 loc_prior: LocPrior,
                 w: float = 1.0,
                 freeze_encoder: bool = True,
                 pretrained_encoder: Optional[nn.Module] = None):
        """
        Args:
            num_labels: Number of labels (L)
            relation_pairs: Protocol pairs R = [(i,j), ...]
            emb_dim: Relation embedding dimension d_r
            loc_prior: LocPrior with (μ_R, Σ_R)
            w: Scaling parameter for Mahalanobis distance
            freeze_encoder: Whether to freeze encoder weights
            pretrained_encoder: Optional pretrained RelationEncoder from pretraining
        """
        super().__init__()
        self.num_labels = num_labels
        self.relation_pairs = relation_pairs
        self.K = len(relation_pairs)
        self.emb_dim = emb_dim
        self.prior = loc_prior
        self.w = nn.Parameter(torch.tensor(w), requires_grad=not freeze_encoder)

        # Build or load relation encoder
        if pretrained_encoder is not None:
            self.encoder = pretrained_encoder
        else:
            # Build new RelationEncoder (same architecture as in pretrain_loc_encoder.py)
            self.encoder = self._build_relation_encoder()

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _build_relation_encoder(self) -> nn.Module:
        """
        Build RelationEncoder with same architecture as pretrain_loc_encoder.py

        Input: (batch, K, 3) relation vectors
        Output: (batch, d_r) normalized embeddings
        """
        class RelationEncoder(nn.Module):
            def __init__(self, K: int, d_r: int):
                super().__init__()
                self.K = K
                self.d_r = d_r

                # Process each relation vector independently
                self.relation_mlp = nn.Sequential(
                    nn.Linear(3, 16),
                    nn.ReLU(),
                    nn.Linear(16, 32)
                )

                # Aggregate all K relations
                self.aggregate = nn.Sequential(
                    nn.Linear(K * 32, 128),
                    nn.ReLU(),
                    nn.Linear(128, d_r)
                )

            def forward(self, relation_vectors):
                """
                Args:
                    relation_vectors: (batch, K, 3)
                Returns:
                    embeddings: (batch, d_r) L2-normalized
                """
                batch_size = relation_vectors.size(0)

                # Process each relation independently
                rel_features = []
                for k in range(self.K):
                    feat = self.relation_mlp(relation_vectors[:, k, :])  # (batch, 32)
                    rel_features.append(feat)

                # Concatenate and aggregate
                all_feats = torch.cat(rel_features, dim=1)  # (batch, K*32)
                embedding = self.aggregate(all_feats)  # (batch, d_r)

                # L2 normalize for stability
                return F.normalize(embedding, dim=-1)

        return RelationEncoder(K=self.K, d_r=self.emb_dim)

    def compute_soft_centroids(self, soft_masks: torch.Tensor) -> torch.Tensor:
        """
        Compute soft centroids from probability masks.

        Args:
            soft_masks: (B, L, D, H, W) soft probabilities

        Returns:
            centroids: (B, L, 3) (z,y,x) voxel coordinates
        """
        B, L, D, H, W = soft_masks.shape
        device = soft_masks.device

        # Create coordinate grids
        z = torch.arange(D, device=device, dtype=soft_masks.dtype)
        y = torch.arange(H, device=device, dtype=soft_masks.dtype)
        x = torch.arange(W, device=device, dtype=soft_masks.dtype)
        zz = z.view(1, 1, D, 1, 1)
        yy = y.view(1, 1, 1, H, 1)
        xx = x.view(1, 1, 1, 1, W)

        # Compute weighted average (soft centroid)
        mass = soft_masks.sum(dim=[2,3,4]) + 1e-8  # (B, L)
        cz = (soft_masks * zz).sum(dim=[2,3,4]) / mass
        cy = (soft_masks * yy).sum(dim=[2,3,4]) / mass
        cx = (soft_masks * xx).sum(dim=[2,3,4]) / mass

        centroids = torch.stack([cz, cy, cx], dim=-1)  # (B, L, 3)
        return centroids

    def forward(self, soft_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for location discrimination.

        Args:
            soft_masks: (B, L, D, H, W) predicted segmentation probabilities

        Returns:
            scores: (B,) location scores S_loc ∈ [0, 1]
            embedding: (B, d_r) relation embeddings
        """
        B, L, D, H, W = soft_masks.shape
        assert L == self.num_labels, f"Expected {self.num_labels} labels, got {L}"

        # Step 1: Compute soft centroids
        centroids = self.compute_soft_centroids(soft_masks)  # (B, L, 3)

        # Step 2: Compute protocol relation vectors
        relation_vectors = []
        for (i, j) in self.relation_pairs:
            r_ij = centroids[:, i, :] - centroids[:, j, :]  # (B, 3)
            relation_vectors.append(r_ij)

        relation_vectors = torch.stack(relation_vectors, dim=1)  # (B, K, 3)

        # Step 3: Encode with E_rel
        embedding = self.encoder(relation_vectors)  # (B, d_r)

        # Step 4: Compute Mahalanobis distance and score
        # S_loc = exp(-w * d_R)
        distances = self.prior.mahalanobis_distance(embedding)  # (B,)
        scores = torch.exp(-self.w * distances)  # (B,)

        return scores, embedding

    def load_pretrained_encoder(self, checkpoint_path: str, device: torch.device):
        """
        Load pretrained encoder weights from checkpoint.

        Args:
            checkpoint_path: Path to pretrained encoder checkpoint (.pth file)
            device: Device to load weights to
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        try:
            # Try to load state dict directly
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=True)
            print(f"✓ Loaded pretrained location encoder from: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️  Warning: Strict loading failed: {e}")
            print(f"   Attempting to load with strict=False...")

            # Try non-strict loading
            missing_keys, unexpected_keys = self.encoder.load_state_dict(
                checkpoint['encoder_state_dict'], strict=False
            )

            if missing_keys:
                print(f"   Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"   Unexpected keys: {unexpected_keys}")

        print(f"  Trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"  Best val loss: {checkpoint.get('val_loss', 'N/A')}")

# ---------------------------
# ProtoAtlasLoss: combined loss to train G
# ---------------------------
class ProtoAtlasLoss(nn.Module):
    """
    Combine supervised loss (optional) with shape & location prior losses.

    L_total = lambda_sup * L_sup + lambda_s * sum(1 - S_shape(label)) + lambda_l * (1 - S_loc)
    where S_* in (0,1], so losses are in [0,1]
    """
    def __init__(self,
                 shape_disc: ShapeDisc,
                 loc_disc: LocDisc,
                 lambda_sup: float = 1.0,
                 lambda_s: float = 0.5,
                 lambda_l: float = 0.5,
                 use_cross_entropy: bool = True):
        super().__init__()
        self.shape_disc = shape_disc
        self.loc_disc = loc_disc
        self.lambda_sup = lambda_sup
        self.lambda_s = lambda_s
        self.lambda_l = lambda_l
        self.ce = nn.CrossEntropyLoss() if use_cross_entropy else None
        self.dice_eps = 1e-6

    @staticmethod
    def soft_dice(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
        # pred: (B, L, D, H, W) probs; target: (B, L, D, H, W) one-hot or soft
        num = 2.0 * (pred * target).sum(dim=[2,3,4])
        den = pred.sum(dim=[2,3,4]) + target.sum(dim=[2,3,4]) + eps
        dice = (num / den).mean(dim=1)  # mean across labels -> (B,)
        return dice  # (B,)

    def forward(self,
                pred: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                label_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        pred: (B, L, D, H, W) soft probabilities (after softmax)
        target: (B, L, D, H, W) one-hot ground truth (optional)
        label_names: list of labels length L (required for shape_disc)
        returns dict with losses and raw components
        """
        device = pred.device
        B, L, D, H, W = pred.shape
        assert label_names is not None and len(label_names) == L

        # Supervised loss
        if target is not None:
            # assume target has shape (B, L, D, H, W) one-hot; use dice as consistency
            dice = self.soft_dice(pred, target)  # (B,)
            L_sup = 1.0 - dice.mean()
        else:
            L_sup = torch.tensor(0.0, device=device)

        # Shape losses
        L_shape_total = 0.0
        shape_scores = {}
        for li, lab in enumerate(label_names):
            # build mask input for encoder: take predicted channel, unsqueeze channel dim
            mask = pred[:, li:li+1, ...]  # (B,1,D,H,W)
            s_scores, _ = self.shape_disc(mask, lab)
            shape_scores[lab] = s_scores.detach().cpu().numpy()
            L_shape_total = L_shape_total + (1.0 - s_scores).mean()

        L_shape_total = L_shape_total / float(L)

        # Location loss
        loc_scores, _ = self.loc_disc(pred)
        L_loc = (1.0 - loc_scores).mean()

        L_total = self.lambda_sup * L_sup + self.lambda_s * L_shape_total + self.lambda_l * L_loc
        return {
            'loss_total': L_total,
            'loss_sup': L_sup,
            'loss_shape': L_shape_total,
            'loss_loc': L_loc,
            'shape_scores': shape_scores,
            'loc_scores': loc_scores.detach().cpu().numpy()
        }

# ---------------------------
# ProtoAtlasEvaluator: Evaluate segmentation quality using prototypes
# ---------------------------
class ProtoAtlasEvaluator:
    """
    Evaluate segmentation quality without ground truth using shape and location prototypes.

    This evaluator scores predicted segmentation results based on:
    1. Shape plausibility: Compare predicted masks with learned shape prototypes (Gaussian priors)
    2. Location plausibility: Compare spatial relationships with learned location prototypes (Gaussian prior)

    Score range: [0, 1], where higher scores indicate better segmentation quality.

    Usage:
        evaluator = ProtoAtlasEvaluator(
            shape_encoder=trained_shape_encoder,
            shape_prior=loaded_shape_prior,
            loc_encoder=trained_loc_encoder,
            loc_prior=loaded_loc_prior,
            protocol_pairs=[(0,1), (0,2), ...],
            use_enhanced_location=True,  # Use enhanced features with adjacency
            device=torch.device('cuda')
        )

        scores = evaluator.evaluate(pred_probabilities, label_names)
    """
    def __init__(
        self,
        shape_encoder: nn.Module,
        shape_prior: ShapePrior,
        loc_encoder: nn.Module,
        loc_prior: LocPrior,
        protocol_pairs: List[Tuple[int, int]],
        spacing: List[Tuple[int, int, int]],
        use_enhanced_location: bool = True,
        beta: float = 1.0,
        w: float = 1.0,
        alpha: float = 0.5,
        device: Optional[torch.device] = None,
        use_loc: bool = True,
        use_shape: bool = True,
        distance_metric: str = 'mahalanobis',
        use_fp16: bool = False,
        use_direct_location: bool = False,
        direct_location_method: str = 'method_a'

    ):
        """
        Args:
            shape_encoder: Trained shape encoder (SE3Encoder)
            shape_prior: ShapePrior with per-label Gaussian priors (μ_l, Σ_l)
            loc_encoder: Trained location encoder (EnhancedRelationEncoder or RelationEncoder)
            loc_prior: LocPrior with global Gaussian prior (μ_R, Σ_R) or mean prior (means, stds)
            protocol_pairs: List of (i, j) protocol pairs for location encoding
            use_enhanced_location: If True, use 7-dim enhanced features; if False, use 3-dim centroid features
            beta: Scaling parameter for shape Mahalanobis distance (only used when distance_metric='mahalanobis')
            w: Scaling parameter for location Mahalanobis distance (only used when distance_metric='mahalanobis')
            alpha: Weight for combining shape and location scores (overall = alpha*shape + (1-alpha)*loc)
            device: Device for computation
            distance_metric: Distance metric to use - 'mahalanobis' or 'cosine'
                           'mahalanobis': Use Mahalanobis distance + exponential transformation
                           'cosine': Use cosine similarity (more robust for small sample sizes)
            use_fp16: If True, use half-precision (fp16) for inference to reduce memory usage
            use_direct_location: If True, use direct feature comparison without encoder
            direct_location_method: 'method_a' or 'method_c' for direct mode
        """
        self.shape_encoder = shape_encoder
        self.shape_prior = shape_prior
        self.loc_encoder = loc_encoder
        self.loc_prior = loc_prior
        self.protocol_pairs = protocol_pairs
        self.use_enhanced_location = use_enhanced_location
        self.beta = beta
        self.w = w
        self.alpha = alpha
        self.device = device if device is not None else torch.device('cpu')
        self.use_shape = use_shape
        self.use_loc = use_loc
        self.spacing = spacing
        self.distance_metric = distance_metric
        self.use_fp16 = use_fp16
        self.use_direct_location = use_direct_location
        self.direct_location_method = direct_location_method

        # Validate parameters
        if distance_metric not in ['mahalanobis', 'cosine']:
            raise ValueError(f"distance_metric must be 'mahalanobis' or 'cosine', got '{distance_metric}'")

        if direct_location_method not in ['method_a', 'method_c']:
            raise ValueError(f"direct_location_method must be 'method_a' or 'method_c', got '{direct_location_method}'")

        # Move to device and set to eval mode
        self.shape_encoder.to(self.device).eval()
        self.loc_encoder.to(self.device).eval()
        self.shape_prior.to(self.device)
        self.loc_prior.to(self.device)

        # Convert to fp16 if requested
        if self.use_fp16 and self.device.type == 'cuda':
            print(f"Converting models to FP16 for memory optimization...")
            self.shape_encoder.half()
            self.loc_encoder.half()
            # Priors stay in fp32 for numerical stability
            print(f"✓ Models converted to FP16")

        # Freeze parameters
        for p in self.shape_encoder.parameters():
            p.requires_grad = False
        for p in self.loc_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def compute_shape_scores(
        self,
        pred: torch.Tensor,
        label_names: List[str]
    ) -> Dict[str, float]:
        """
        Compute per-label shape plausibility scores.

        Args:
            pred: (B, L, D, H, W) predicted probability masks
            label_names: List of label names (length L)

        Returns:
            Dict mapping label names to shape scores [0, 1]
        """
        B, L, D, H, W = pred.shape
        shape_scores = {}

        for li, label_name in enumerate(label_names):
            # Extract probability mask for this label
            mask = pred[:, li:li+1, ...]  # (B, 1, D, H, W)

            # Convert to fp16 if using mixed precision
            if self.use_fp16 and self.device.type == 'cuda':
                mask = mask.half()

            # Encode with shape encoder (with autocast if fp16 enabled)
            if self.use_fp16 and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    embedding = self.shape_encoder(mask)  # (B, emb_dim)
            else:
                embedding = self.shape_encoder(mask)  # (B, emb_dim)

            # Convert back to fp32 for numerical stability in distance computation
            embedding = embedding.float()

            # Normalize embedding (good practice for both metrics)


            # Choose metric based on distance_metric parameter
            if self.distance_metric == 'cosine':
                # Use cosine similarity (returns score directly in [0, 1])
                embedding = F.normalize(embedding, p=2, dim=-1)
                print(f'shape embedding for {label_name}:', embedding)
                score = self.shape_prior.cosine_similarity(embedding, label_name)  # (B,)
                print(f'shape cosine score for {label_name}:', score)
            else:  # 'mahalanobis'
                # Compute Mahalanobis distance to label-specific Gaussian prior
                distance = self.shape_prior.mahalanobis_distance(embedding, label_name)  # (B,)
                print(f'shape mahalanobis distance for {label_name}:', distance)

                # Convert distance to score: score = exp(-beta * distance)
                score = torch.exp(-self.beta * distance)  # (B,)

            # Average over batch (usually B=1 for inference)
            shape_scores[label_name] = float(score.mean().cpu().item())

            # Memory optimization: explicitly delete intermediate tensors and clear cache
            del mask, embedding, score
            if self.distance_metric == 'mahalanobis':
                del distance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return shape_scores

    @torch.no_grad()
    def compute_location_score(
        self,
        pred: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compute global location plausibility score.

        Args:
            pred: (B, L, D, H, W) predicted probability masks

        Returns:
            Dict containing:
                - 'score': float, overall location score [0, 1]
                - 'per_region_scores': Dict[str, float], scores for each of 7 brain regions
                - 'per_pair_scores': Optional[(B, K)] per-protocol-pair scores (only in direct mode method_a)
        """
        B, L, D, H, W = pred.shape

        if self.use_enhanced_location:
            # Use enhanced location features (7-dim: rel_pos + adjacency)
            location_features = self._extract_enhanced_features_batch(pred)  # (B, K, 7)
        else:
            # Use simple centroid-based features (3-dim: rel_pos only)
            location_features = self._extract_centroid_features_batch(pred)  # (B, K, 3)

        result = {}

        if self.use_direct_location:
            # Direct mode: compute scores without encoder
            if self.direct_location_method == 'method_a':
                # Method A: per-pair standardized distance
                overall_scores, per_pair_scores = self.loc_prior.compute_direct_score_method_a(
                    location_features, beta=self.w
                )  # (B,), (B, K)

                result['score'] = float(overall_scores.mean().cpu().item())
                result['per_pair_scores'] = per_pair_scores.cpu()  # (B, K)

                # Compute per-region scores
                result['per_region_scores'] = self._compute_per_region_scores(
                    per_pair_scores.cpu().numpy()  # (B, K)
                )

            else:  # method_c
                # Method C: weighted Euclidean distance
                overall_scores = self.loc_prior.compute_direct_score_method_c(
                    location_features, beta=self.w
                )  # (B,)

                result['score'] = float(overall_scores.mean().cpu().item())
                result['per_pair_scores'] = None

                # For method_c, we can still compute per-region scores by computing individual distances
                # Extract per-pair distances for region-wise analysis
                per_pair_scores = self._compute_per_pair_scores_method_c(location_features)
                result['per_region_scores'] = self._compute_per_region_scores(
                    per_pair_scores.cpu().numpy()  # (B, K)
                )

            print(f'Direct location score ({self.direct_location_method}):', result['score'])

        else:
            # Encoder-based mode (original implementation)
            # Convert to fp16 if using mixed precision
            if self.use_fp16 and self.device.type == 'cuda':
                location_features = location_features.half()

            # Encode with location encoder (with autocast if fp16 enabled)
            if self.use_fp16 and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    embedding = self.loc_encoder(location_features)  # (B, d_r)
            else:
                embedding = self.loc_encoder(location_features)  # (B, d_r)

            # Convert back to fp32 for numerical stability
            embedding = embedding.float()
            print('location embedding:', embedding)

            # Choose metric based on distance_metric parameter
            if self.distance_metric == 'cosine':
                # Use cosine similarity (returns score directly in [0, 1])
                score = self.loc_prior.cosine_similarity(embedding)  # (B,)
                print('location cosine score:', score)
            else:  # 'mahalanobis'
                # Compute Mahalanobis distance to location Gaussian prior
                distance = self.loc_prior.mahalanobis_distance(embedding)  # (B,)
                print('location mahalanobis distance:', distance)

                # Convert distance to score: score = exp(-w * distance)
                score = torch.exp(-self.w * distance)  # (B,)

            # Average over batch
            result['score'] = float(score.mean().cpu().item())
            result['per_pair_scores'] = None
            result['per_region_scores'] = {}  # Not available in encoder mode

            # Memory cleanup
            del embedding, score
            if self.distance_metric == 'mahalanobis':
                del distance

        # Memory optimization
        del location_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def _compute_per_pair_scores_method_c(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pair scores for method C (for per-region analysis)

        Args:
            features: (B, K, 7) location features

        Returns:
            per_pair_scores: (B, K) scores for each protocol pair
        """
        B, K, feat_dim = features.shape
        device = features.device

        means = self.loc_prior.means.to(device)  # (K, 7)

        # Compute Euclidean distance for each pair
        diff = features - means.unsqueeze(0)  # (B, K, 7)
        distances_sq = torch.sum(diff ** 2, dim=2)  # (B, K)
        distances = torch.sqrt(distances_sq + 1e-8) / np.sqrt(feat_dim)  # (B, K)

        # Convert to scores
        per_pair_scores = torch.exp(-self.w * distances)  # (B, K)

        return per_pair_scores

    def _compute_per_region_scores(self, per_pair_scores: np.ndarray) -> Dict[str, float]:
        """
        Compute per-region location scores from per-pair scores.

        For each of the 7 bilateral brain regions, compute the average score
        of all protocol pairs involving that region's left or right labels.

        Brain region mapping (bilateral label -> L/R labels):
            1 (Hippocampus) -> 1 (L), 2 (R)
            2 (Amygdala) -> 3 (L), 4 (R)
            3 (Caudate) -> 5 (L), 6 (R)
            4 (Putamen) -> 7 (L), 8 (R)
            5 (Pallidum) -> 9 (L), 10 (R)
            6 (Thalamus) -> 11 (L), 12 (R)
            7 (Accumbens) -> 13 (L), 14 (R)

        Args:
            per_pair_scores: (B, K) scores for each protocol pair

        Returns:
            Dict mapping region names to their average scores
        """
        # Average over batch
        per_pair_scores_avg = per_pair_scores.mean(axis=0)  # (K,)

        # Define region mappings
        region_names = ['hippocampus', 'amygdala', 'caudate', 'putamen',
                       'pallidum', 'thalamus', 'accumbens']

        # Each region has left (odd) and right (even) labels
        # Labels are 0-indexed in protocol_pairs: 0-13 correspond to labels 1-14
        region_to_label_indices = {
            'hippocampus': [0, 1],    # Labels 1, 2
            'amygdala': [2, 3],       # Labels 3, 4
            'caudate': [4, 5],        # Labels 5, 6
            'putamen': [6, 7],        # Labels 7, 8
            'pallidum': [8, 9],       # Labels 9, 10
            'thalamus': [10, 11],     # Labels 11, 12
            'accumbens': [12, 13],    # Labels 13, 14
        }

        # For each region, find all protocol pairs involving that region's labels
        per_region_scores = {}

        for region_name in region_names:
            label_indices = region_to_label_indices[region_name]

            # Find all protocol pairs involving this region
            pair_mask = []
            for k, (i, j) in enumerate(self.protocol_pairs):
                # abnormal=False
                if not abnormal:
                    if (i,j) in [(0,4),(10,12),(1,5),(11,13)]:
                        continue
                if i in label_indices or j in label_indices:
                    pair_mask.append(k)

            if len(pair_mask) > 0:
                # Average scores for pairs involving this region
                print(f'region_name:{region_name},score: {per_pair_scores_avg[pair_mask]}')
                region_score = per_pair_scores_avg[pair_mask].mean()
                per_region_scores[region_name] = float(region_score)
            else:
                per_region_scores[region_name] = 0.0

        return per_region_scores

    def _extract_centroid_features_batch(
        self,
        pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract centroid-based relation features (simple 3-dim version).

        Args:
            pred: (B, L, D, H, W) probability masks

        Returns:
            features: (B, K, 3) relation vectors
        """
        B, L, D, H, W = pred.shape
        device = pred.device

        # Compute soft centroids for all labels
        centroids = self._compute_soft_centroids(pred)  # (B, L, 3)

        # Compute relation vectors for protocol pairs
        relation_vectors = []
        for (i, j) in self.protocol_pairs:
            r_ij = centroids[:, i, :] - centroids[:, j, :]  # (B, 3)
            relation_vectors.append(r_ij)

        return torch.stack(relation_vectors, dim=1)  # (B, K, 3)

    def _extract_enhanced_features_batch(
        self,
        pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract enhanced location features (7-dim: rel_pos + adjacency).

        Args:
            pred: (B, L, D, H, W) probability masks

        Returns:
            features: (B, K, 7) enhanced relation features
        """
        B, L, D, H, W = pred.shape
        print('pred.shape:',pred.shape)
        device = pred.device

        # Process each sample in batch
        batch_features = []
        for b in range(B):
            # Extract features for this sample
            prob_map = pred[b].cpu().numpy()  # (L, D, H, W)

            try:
                # Import enhanced feature extraction
                from location_encoder_enhanced import extract_enhanced_location_features_from_probmap

                features = extract_enhanced_location_features_from_probmap(
                    prob_map=prob_map,
                    protocol_pairs=self.protocol_pairs,
                    prob_threshold=0.5,
                    spacing=self.spacing
                )  # (K, 7)
            except ImportError:
                warnings.warn("Failed to import enhanced feature extraction. Using centroid-only features.")
                # Fallback: use centroid features padded with zeros
                centroids = self._compute_soft_centroids(pred[b:b+1])  # (1, L, 3)
                relation_vectors = []
                for (i, j) in self.protocol_pairs:
                    r_ij = centroids[0, i, :] - centroids[0, j, :]  # (3,)
                    # Pad with zeros for adjacency features
                    r_ij_padded = torch.cat([r_ij, torch.zeros(4, device=device)])  # (7,)
                    relation_vectors.append(r_ij_padded.cpu().numpy())
                features = np.array(relation_vectors)  # (K, 7)

            batch_features.append(torch.from_numpy(features).float())

        return torch.stack(batch_features).to(device)  # (B, K, 7)

    def _compute_soft_centroids(
        self,
        soft_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soft centroids from probability masks.

        Args:
            soft_masks: (B, L, D, H, W) soft probabilities

        Returns:
            centroids: (B, L, 3) (z,y,x) voxel coordinates
        """
        B, L, D, H, W = soft_masks.shape
        device = soft_masks.device

        # Create coordinate grids
        z = torch.arange(D, device=device, dtype=soft_masks.dtype)
        y = torch.arange(H, device=device, dtype=soft_masks.dtype)
        x = torch.arange(W, device=device, dtype=soft_masks.dtype)
        zz = z.view(1, 1, D, 1, 1)
        yy = y.view(1, 1, 1, H, 1)
        xx = x.view(1, 1, 1, 1, W)

        # Compute weighted average (soft centroid)
        mass = soft_masks.sum(dim=[2,3,4]) + 1e-8  # (B, L)
        cz = (soft_masks * zz).sum(dim=[2,3,4]) / mass
        cy = (soft_masks * yy).sum(dim=[2,3,4]) / mass
        cx = (soft_masks * xx).sum(dim=[2,3,4]) / mass

        centroids = torch.stack([cz, cy, cx], dim=-1)  # (B, L, 3)
        return centroids

    @torch.no_grad()
    def evaluate(
        self,
        pred: torch.Tensor,
        pred_loc: torch.Tensor,
        label_names: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate segmentation quality using shape and location prototypes.

        Args:
            pred: (B, L, D, H, W) predicted probability masks (after softmax)
            label_names: List of label names (length L)

        Returns:
            Dictionary with evaluation scores:
            {
                'per_label': {
                    label_name: {
                        'shape_score': float [0, 1]
                    },
                    ...
                },
                'location': {
                    'score': float [0, 1]
                },
                'shape_mean': float [0, 1],  # Mean of all shape scores
                'overall': float [0, 1]  # Weighted combination
            }
        """
        device = self.device
        pred = pred.to(device)
        pred_loc = pred_loc.to(device)
        B, L = pred.shape[0], pred.shape[1]

        if L != len(label_names):
            raise ValueError(f"Number of labels in pred ({L}) does not match label_names ({len(label_names)})")

        if self.use_shape:
            # Compute shape scores (per-label)
            shape_scores = self.compute_shape_scores(pred, label_names)
            print('shape_scores:',shape_scores)
        else:
            shape_scores = {}
            for label_name in label_names:
                shape_scores[label_name] = 0.0


        if self.use_loc:
            # Compute location score (global)
            print('self.protocol_pairs:',self.protocol_pairs)
            location_result = self.compute_location_score(pred_loc)  # Returns dict
            
            location_score = location_result['score']
            per_region_scores = location_result.get('per_region_scores', {})
        else:
            location_score = 0.0
            per_region_scores = {}

        # Compute mean shape score
        mean_shape = float(np.mean(list(shape_scores.values()))) if len(shape_scores) > 0 else 0.0

        # Compute overall score (weighted combination)
        overall = self.alpha * mean_shape + (1.0 - self.alpha) * location_score

        # Build per-label results
        per_label = {
            label_name: {'shape_score': shape_scores[label_name]}
            for label_name in label_names
        }

        return {
            'per_label': per_label,
            'location': {
                'score': location_score,
                'per_region_scores': per_region_scores
            },
            'shape_mean': mean_shape,
            'overall': overall
        }

    @staticmethod
    def from_checkpoints(
        shape_encoder_path: str,
        shape_prior_path: str,
        loc_encoder_path: str,
        loc_prior_path: str,
        spacing: List[Tuple[int, int, int]],
        use_enhanced_location: bool = True,
        use_se3_shape: bool = False,
        beta: float = 1.0,
        w: float = 1.0,
        alpha: float = 0.5,
        device: Optional[torch.device] = None,
        use_loc: bool = True,
        use_shape: bool = True,
        distance_metric: str = 'mahalanobis',
        use_fp16: bool = False,
        use_direct_location: bool = False,
        direct_location_method: str = 'method_a'
    ) -> 'ProtoAtlasEvaluator':
        """
        Convenience method to load evaluator from checkpoint files.

        Args:
            shape_encoder_path: Path to shape encoder checkpoint (.pth)
            shape_prior_path: Path to shape prior file (.npz)
            loc_encoder_path: Path to location encoder checkpoint (.pth), optional if use_direct_location=True
            loc_prior_path: Path to location prior file (.npz) or location_mean_prior.npz
            use_enhanced_location: If True, use EnhancedRelationEncoder; else use simple RelationEncoder
            use_se3_shape: If True, use SE(3)-equivariant shape encoder
            beta: Scaling for shape Mahalanobis distance (only used when distance_metric='mahalanobis')
            w: Scaling for location Mahalanobis distance (only used when distance_metric='mahalanobis')
            alpha: Weight for combining scores
            device: Device for computation
            distance_metric: Distance metric to use - 'mahalanobis' or 'cosine'
            use_fp16: If True, use half-precision (fp16) for inference to reduce memory usage
            use_direct_location: If True, use direct feature comparison without encoder
            direct_location_method: 'method_a' or 'method_c' for direct mode

        Returns:
            ProtoAtlasEvaluator instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load shape encoder and prior
        print("Loading shape encoder and prior...")
        shape_checkpoint = torch.load(shape_encoder_path, map_location=device)
        emb_dim = shape_checkpoint.get('args', {}).get('emb_dim', 128)

        shape_encoder = SE3Encoder(emb_dim=emb_dim, use_e3nn=use_se3_shape)
        shape_encoder.load_state_dict(shape_checkpoint['encoder_state_dict'], strict=False)
        shape_encoder.to(device).eval()

        shape_prior = ShapePrior.load_from_file(shape_prior_path)
        shape_prior.to(device)

        print(f"✓ Shape encoder loaded (emb_dim={emb_dim})")
        print(f"✓ Shape prior loaded for {len(shape_prior.labels)} labels")

        # Load location prior first to get protocol_pairs
        print("\nLoading location prior...")
        loc_prior = LocPrior.load_from_file(loc_prior_path)
        loc_prior.to(device)
        protocol_pairs = loc_prior.relation_pairs
        K = len(protocol_pairs)

        # Load location encoder (skip if using direct mode)
        if use_direct_location:
            print(f"✓ Using direct location mode ({direct_location_method})")
            print(f"✓ Skipping location encoder loading")
            # Create a dummy encoder (not used)
            loc_encoder = nn.Identity()
        else:
            print("Loading location encoder...")
            loc_checkpoint = torch.load(loc_encoder_path, map_location=device)
            d_r = loc_checkpoint.get('d_r', 64)
            print(f'  d_r: {d_r}')

            if use_enhanced_location:
                from location_encoder_enhanced import EnhancedRelationEncoder
                loc_encoder = EnhancedRelationEncoder(K=K, d_r=d_r)
            else:
                # Build simple RelationEncoder
                class RelationEncoder(nn.Module):
                    def __init__(self, K: int, d_r: int):
                        super().__init__()
                        self.K = K
                        self.d_r = d_r
                        self.relation_mlp = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 32))
                        self.aggregate = nn.Sequential(nn.Linear(K * 32, 128), nn.ReLU(), nn.Linear(128, d_r))

                    def forward(self, relation_vectors):
                        batch_size = relation_vectors.size(0)
                        rel_features = [self.relation_mlp(relation_vectors[:, k, :]) for k in range(self.K)]
                        all_feats = torch.cat(rel_features, dim=1)
                        embedding = self.aggregate(all_feats)
                        return F.normalize(embedding, dim=-1)

                loc_encoder = RelationEncoder(K=K, d_r=d_r)

            loc_encoder.load_state_dict(loc_checkpoint['encoder_state_dict'], strict=False)
            loc_encoder.to(device).eval()
            print(f"✓ Location encoder loaded (d_r={d_r}, K={K})")

        print(f"✓ Location prior loaded (K={K} protocol pairs)")

        return ProtoAtlasEvaluator(
            shape_encoder=shape_encoder,
            shape_prior=shape_prior,
            loc_encoder=loc_encoder,
            loc_prior=loc_prior,
            protocol_pairs=protocol_pairs,
            use_enhanced_location=use_enhanced_location,
            beta=beta,
            w=w,
            alpha=alpha,
            device=device,
            use_loc=use_loc,
            use_shape=use_shape,
            spacing=spacing,
            distance_metric=distance_metric,
            use_fp16=use_fp16,
            use_direct_location=use_direct_location,
            direct_location_method=direct_location_method
        )

    def load_spacing(self, spacing):
        self.spacing=spacing

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Quick demonstration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = ['L_A', 'R_A', 'L_B', 'R_B']  # example 4 vascular labels
    L = len(labels)
    emb_dim = 128

    # 1) build a shape prior (fake prototypes here)
    prototypes = {}
    for lab in labels:
        # 5 prototypes per label randomly (in real case, compute embeddings from SDMs)
        prototypes[lab] = np.random.randn(5, emb_dim).astype(np.float32)
    shape_prior = ShapePrior(labels=labels, emb_dim=emb_dim, prototypes=prototypes)
    shape_prior.to(device)

    # 2) build loc prior (relation pairs: some example pairs)
    relation_pairs = [(0,1), (2,3), (0,2), (1,3)]  # pairs of label indices
    loc_protos = np.random.randn(10, emb_dim).astype(np.float32)
    loc_prior = LocPrior(relation_pairs=relation_pairs, emb_dim=emb_dim, prototypes=loc_protos)
    loc_prior.to(device)

    # 3) build discriminators
    shape_disc = ShapeDisc(emb_dim=emb_dim, shape_prior=shape_prior, scoring='proto_cosine', beta=1.0).to(device)
    loc_disc = LocDisc(num_labels=L, relation_pairs=relation_pairs, emb_dim=emb_dim,
                       loc_prior=loc_prior, scoring='proto_cosine', gamma=1.0).to(device)

    # 4) build loss wrapper
    atlas_loss = ProtoAtlasLoss(shape_disc=shape_disc, loc_disc=loc_disc, lambda_sup=1.0, lambda_s=0.5, lambda_l=0.5)

    # 5) fake pred and target
    B = 1
    D, H, W = 64, 64, 64
    # fake predictions — must sum to 1 across channels (softmax)
    logits = torch.randn(B, L, D, H, W, device=device)
    pred = F.softmax(logits, dim=1)
    # fake one-hot GT (for demonstration)
    gt_idx = torch.randint(0, L, (B, D, H, W), device=device)
    target = F.one_hot(gt_idx, num_classes=L).permute(0,4,1,2,3).float()

    # 6) compute loss
    out = atlas_loss(pred, target=target, label_names=labels)
    print("Loss components:", {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in out.items() if 'scores' not in k})
    # shape scores small sample
    print("Loc score (sample):", out['loc_scores'][:5])

    # 7) evaluator
    evaluator = ProtoAtlasEvaluator(shape_disc=shape_disc, loc_disc=loc_disc, device=device)
    eval_res = evaluator.evaluate(pred[0:1], labels)
    print("Eval result:", eval_res)

# ---------------------------
# Pretraining strategies for ShapeDisc and LocDisc encoders
# ---------------------------

class ShapeEncoderPretrainer:
    """
    Pretraining strategy for ShapeDisc encoder.

    Strategy: Contrastive learning on shape embeddings
    - Positive pairs: augmentations of the same anatomical structure
    - Negative pairs: different anatomical structures
    - Loss: InfoNCE (contrastive loss)

    Data preparation:
    1. Collect annotated 3D NIfTI images with subcortical labels
    2. For each label, extract binary masks
    3. Apply data augmentation: rotation, scaling, elastic deformation
    4. Create positive pairs (same structure, different augmentations)
    5. Create negative pairs (different structures)
    """

    def __init__(self, encoder: SE3Encoder, device: torch.device, temperature: float = 0.07):
        self.encoder = encoder
        self.device = device
        self.temperature = temperature
        self.encoder.to(device)

    def contrastive_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss.

        Args:
            embeddings: (N, emb_dim) normalized embeddings
            labels: (N,) integer labels indicating which samples are from same class

        Returns:
            scalar loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature  # (N, N)

        # Create mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(self.device)  # (N, N)

        # Mask out diagonal (self-similarity)
        mask = mask - torch.eye(mask.shape[0], device=self.device)

        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        loss = -mean_log_prob_pos.mean()

        return loss

    def augment_mask(self, mask: np.ndarray, augmentation_type: str = 'rotation') -> np.ndarray:
        """
        Apply augmentation to a binary mask.

        Args:
            mask: (D, H, W) binary mask
            augmentation_type: 'rotation', 'scaling', 'elastic', or 'noise'

        Returns:
            Augmented mask
        """
        from scipy.ndimage import rotate, zoom, gaussian_filter

        if augmentation_type == 'rotation':
            # Random rotation
            angle = np.random.uniform(-30, 30)
            axis = np.random.choice([0, 1, 2])
            axes = [(0, 1), (0, 2), (1, 2)][axis]
            mask = rotate(mask, angle, axes=axes, reshape=False, order=1)

        elif augmentation_type == 'scaling':
            # Random scaling
            scale = np.random.uniform(0.8, 1.2)
            mask = zoom(mask, scale, order=1)
            # Crop or pad to original size
            original_shape = mask.shape
            if mask.shape[0] > original_shape[0]:
                # Crop
                start = (mask.shape[0] - original_shape[0]) // 2
                mask = mask[start:start+original_shape[0]]
            elif mask.shape[0] < original_shape[0]:
                # Pad
                pad_width = [(0, original_shape[0] - mask.shape[0])] * 3
                mask = np.pad(mask, pad_width, mode='constant')

        elif augmentation_type == 'elastic':
            # Elastic deformation (simplified)
            sigma = 5
            mask = gaussian_filter(mask.astype(float), sigma=sigma)

        elif augmentation_type == 'noise':
            # Add Gaussian noise
            noise = np.random.normal(0, 0.1, mask.shape)
            mask = mask + noise

        # Threshold to binary
        mask = (mask > 0.5).astype(np.float32)
        return mask

    def prepare_training_data(self, annotation_paths: List[str], labels: List[str],
                             n_augmentations: int = 5) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Prepare training data for pretraining.

        Args:
            annotation_paths: List of NIfTI annotation file paths
            labels: List of label names
            n_augmentations: Number of augmentations per structure instance

        Returns:
            masks: List of mask tensors (1, 1, D, H, W)
            label_ids: List of label indices
        """
        masks = []
        label_ids = []

        for ann_path in annotation_paths:
            ann_data = load_nifti_annotation(ann_path)

            for label_idx, label_name in enumerate(labels):
                voxel_value = label_idx + 1
                binary_mask = (ann_data == voxel_value).astype(np.float32)

                if binary_mask.sum() == 0:
                    continue

                # Original mask
                mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).unsqueeze(0).float()
                masks.append(mask_tensor)
                label_ids.append(label_idx)

                # Augmented versions
                for _ in range(n_augmentations):
                    aug_type = np.random.choice(['rotation', 'scaling', 'elastic', 'noise'])
                    aug_mask = self.augment_mask(binary_mask, aug_type)
                    aug_tensor = torch.from_numpy(aug_mask).unsqueeze(0).unsqueeze(0).float()
                    masks.append(aug_tensor)
                    label_ids.append(label_idx)

        return masks, label_ids

    def train_epoch(self, masks: List[torch.Tensor], label_ids: List[int],
                    optimizer: torch.optim.Optimizer, batch_size: int = 8) -> float:
        """
        Train for one epoch.

        Args:
            masks: List of mask tensors
            label_ids: List of label indices
            optimizer: PyTorch optimizer
            batch_size: Batch size

        Returns:
            Average loss for the epoch
        """
        self.encoder.train()
        total_loss = 0.0
        n_batches = 0

        # Shuffle data
        indices = list(range(len(masks)))
        np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]

            # Prepare batch
            batch_masks = torch.cat([masks[idx] for idx in batch_indices], dim=0).to(self.device)
            batch_labels = torch.tensor([label_ids[idx] for idx in batch_indices],
                                       dtype=torch.long, device=self.device)

            # Forward pass
            embeddings = self.encoder(batch_masks)

            # Compute loss
            loss = self.contrastive_loss(embeddings, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0


class LocationEncoderPretrainer:
    """
    Pretraining strategy for LocDisc encoder.

    Strategy: Supervised learning on spatial relationships
    - Input: Relation vectors (centroids differences) from annotations
    - Output: Predict spatial relationship category (e.g., left-of, right-of, above, below, front, back)
    - Loss: Classification loss or metric learning

    Data preparation:
    1. Collect annotated 3D NIfTI images
    2. Compute centroids for each structure
    3. Compute pairwise relation vectors
    4. Label each relation with spatial category based on dominant direction
    5. Train encoder to predict these categories
    """

    def __init__(self, encoder: nn.Module, device: torch.device, num_classes: int = 6):
        """
        Args:
            encoder: The MLP encoder from LocDisc
            device: torch device
            num_classes: Number of spatial relationship classes (default 6: left, right, anterior, posterior, superior, inferior)
        """
        self.encoder = encoder
        self.device = device
        self.num_classes = num_classes
        self.encoder.to(device)

        # Add classification head for pretraining
        emb_dim = self.encoder[-1].out_features
        self.classifier = nn.Linear(emb_dim, num_classes).to(device)

    def classify_spatial_relationship(self, rel_vec: np.ndarray) -> int:
        """
        Classify spatial relationship based on dominant direction.

        Args:
            rel_vec: (3,) array (z, y, x) difference vector

        Returns:
            Integer class: 0=left, 1=right, 2=anterior, 3=posterior, 4=superior, 5=inferior
        """
        # Find dominant axis
        abs_vec = np.abs(rel_vec)
        dominant_axis = np.argmax(abs_vec)

        # Determine direction
        if dominant_axis == 2:  # x-axis
            return 0 if rel_vec[2] < 0 else 1  # left vs right
        elif dominant_axis == 1:  # y-axis
            return 2 if rel_vec[1] < 0 else 3  # anterior vs posterior
        else:  # z-axis
            return 4 if rel_vec[0] < 0 else 5  # superior vs inferior

    def prepare_training_data(self, annotation_paths: List[str], labels: List[str],
                             relation_pairs: List[Tuple[int, int]]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Prepare training data for spatial relationship classification.

        Args:
            annotation_paths: List of NIfTI file paths
            labels: List of label names
            relation_pairs: List of (i, j) index pairs

        Returns:
            relation_vectors: List of flattened relation vectors (K*3,)
            spatial_classes: List of integer spatial class labels
        """
        relation_vectors = []
        spatial_classes = []

        for ann_path in annotation_paths:
            ann_data = load_nifti_annotation(ann_path)

            # Compute centroids
            centroids = []
            for label_idx in range(len(labels)):
                voxel_value = label_idx + 1
                binary_mask = (ann_data == voxel_value).astype(np.float32)
                centroid = compute_centroid_from_mask(binary_mask)
                centroids.append(centroid)

            centroids = np.array(centroids)  # (L, 3)

            # For each relation pair
            for (i, j) in relation_pairs:
                rel_vec = centroids[i] - centroids[j]  # (3,)
                spatial_class = self.classify_spatial_relationship(rel_vec)

                # Compute all relation vectors as in LocDisc
                all_rels = []
                for (ii, jj) in relation_pairs:
                    all_rels.append(centroids[ii] - centroids[jj])

                rel_flat = np.concatenate(all_rels)  # (K*3,)
                relation_vectors.append(rel_flat)
                spatial_classes.append(spatial_class)

        return relation_vectors, spatial_classes

    def train_epoch(self, relation_vectors: List[np.ndarray], spatial_classes: List[int],
                    optimizer: torch.optim.Optimizer, batch_size: int = 32) -> float:
        """
        Train for one epoch.

        Args:
            relation_vectors: List of relation vectors (K*3,)
            spatial_classes: List of spatial class labels
            optimizer: PyTorch optimizer
            batch_size: Batch size

        Returns:
            Average loss for the epoch
        """
        self.encoder.train()
        self.classifier.train()

        total_loss = 0.0
        n_batches = 0

        # Shuffle data
        indices = list(range(len(relation_vectors)))
        np.random.shuffle(indices)

        criterion = nn.CrossEntropyLoss()

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]

            # Prepare batch
            batch_vecs = np.stack([relation_vectors[idx] for idx in batch_indices])
            batch_labels = np.array([spatial_classes[idx] for idx in batch_indices])

            batch_vecs = torch.from_numpy(batch_vecs).float().to(self.device)
            batch_labels = torch.from_numpy(batch_labels).long().to(self.device)

            # Forward pass
            embeddings = self.encoder(batch_vecs)
            logits = self.classifier(embeddings)

            # Compute loss
            loss = criterion(logits, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0


# ---------------------------
# Complete example with real annotations
# ---------------------------
def example_with_real_annotations():
    """
    Example showing how to use real annotations to build prototypes and train.

    Required data:
    - A set of 3D NIfTI files (.nii or .nii.gz) with integer labels
    - Label values: 1=hippocampus, 2=amygdala, 3=caudate, 4=putamen, 5=pallidum, 6=thalamus, 7=accumbens
    - Background = 0
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define labels
    labels = ["hippocampus", "amygdala", "caudate", "putamen", "pallidum", "thalamus", "accumbens"]
    L = len(labels)
    emb_dim = 128

    # ===== STEP 1: Prepare annotation paths =====
    # Replace with your actual annotation file paths
    annotation_paths = [
        # "/path/to/subject1_labels.nii.gz",
        # "/path/to/subject2_labels.nii.gz",
        # "/path/to/subject3_labels.nii.gz",
    ]

    print("=" * 80)
    print("STEP 1: Building Prototypes from Real Annotations")
    print("=" * 80)

    if len(annotation_paths) == 0:
        print("WARNING: No annotation paths provided. Using random prototypes for demonstration.")
        print("To use real data, provide paths to NIfTI annotation files.")

        # Fallback to random prototypes
        prototypes = {lab: np.random.randn(5, emb_dim).astype(np.float32) for lab in labels}
        shape_prior = ShapePrior(labels=labels, emb_dim=emb_dim, prototypes=prototypes)

        relation_pairs = [(i, j) for i in range(L) for j in range(i+1, L)][:10]
        loc_protos = np.random.randn(10, emb_dim).astype(np.float32)
        loc_prior = LocPrior(relation_pairs=relation_pairs, emb_dim=emb_dim, prototypes=loc_protos)

    else:
        # ===== Build shape encoder and extract prototypes =====
        shape_encoder = SE3Encoder(emb_dim=emb_dim, use_e3nn=False).to(device)
        shape_encoder.eval()

        print(f"Extracting shape prototypes from {len(annotation_paths)} annotation files...")
        prototypes = extract_prototypes_from_annotations(
            annotation_paths=annotation_paths,
            labels=labels,
            encoder=shape_encoder,
            device=device,
            method='kmeans',
            n_prototypes=5,
            use_sdf=False
        )

        shape_prior = ShapePrior(labels=labels, emb_dim=emb_dim, prototypes=prototypes)
        print(f"Shape prototypes extracted for {len(prototypes)} labels")

        # ===== Build location encoder and extract location prototypes =====
        # Define relation pairs (all pairwise combinations, limit to reasonable number)
        relation_pairs = [(i, j) for i in range(L) for j in range(i+1, L)]
        print(f"Using {len(relation_pairs)} relation pairs for location encoding")

        # Create a temporary LocDisc to get the encoder
        temp_loc_prior = LocPrior(relation_pairs=relation_pairs, emb_dim=emb_dim)
        temp_loc_disc = LocDisc(num_labels=L, relation_pairs=relation_pairs,
                               emb_dim=emb_dim, loc_prior=temp_loc_prior, freeze_encoder=False)
        loc_encoder = temp_loc_disc.encoder.to(device)
        loc_encoder.eval()

        print(f"Extracting location prototypes from {len(annotation_paths)} annotation files...")
        loc_protos = extract_location_prototypes_from_annotations(
            annotation_paths=annotation_paths,
            labels=labels,
            relation_pairs=relation_pairs,
            loc_encoder=loc_encoder,
            device=device,
            method='kmeans',
            n_prototypes=10
        )

        loc_prior = LocPrior(relation_pairs=relation_pairs, emb_dim=emb_dim, prototypes=loc_protos)
        print(f"Location prototypes extracted: {loc_protos.shape}")

    shape_prior.to(device)
    loc_prior.to(device)

    # ===== STEP 2: Build discriminators =====
    print("\n" + "=" * 80)
    print("STEP 2: Building Discriminators")
    print("=" * 80)

    shape_disc = ShapeDisc(emb_dim=emb_dim, shape_prior=shape_prior,
                          scoring='proto_cosine', beta=1.0,
                          freeze_encoder=False, use_e3nn=False).to(device)

    loc_disc = LocDisc(num_labels=L, relation_pairs=relation_pairs, emb_dim=emb_dim,
                      loc_prior=loc_prior, scoring='proto_cosine',
                      gamma=1.0, freeze_encoder=False).to(device)

    print("Shape discriminator created")
    print("Location discriminator created")

    # ===== STEP 3: (Optional) Pretrain encoders =====
    print("\n" + "=" * 80)
    print("STEP 3: Pretraining Encoders (Optional)")
    print("=" * 80)

    if len(annotation_paths) > 0:
        print("\nPretraining Shape Encoder...")
        shape_pretrainer = ShapeEncoderPretrainer(shape_disc.encoder, device)
        shape_optimizer = torch.optim.Adam(shape_disc.encoder.parameters(), lr=1e-4)

        # Prepare training data
        masks, label_ids = shape_pretrainer.prepare_training_data(annotation_paths, labels, n_augmentations=5)
        print(f"Prepared {len(masks)} training samples (with augmentations)")

        # Train for a few epochs
        for epoch in range(3):
            loss = shape_pretrainer.train_epoch(masks, label_ids, shape_optimizer, batch_size=8)
            print(f"  Epoch {epoch+1}: Loss = {loss:.4f}")

        print("\nPretraining Location Encoder...")
        loc_pretrainer = LocationEncoderPretrainer(loc_disc.encoder, device)
        loc_optimizer = torch.optim.Adam(list(loc_disc.encoder.parameters()) +
                                        list(loc_pretrainer.classifier.parameters()), lr=1e-3)

        # Prepare training data
        rel_vecs, spatial_classes = loc_pretrainer.prepare_training_data(annotation_paths, labels, relation_pairs)
        print(f"Prepared {len(rel_vecs)} relation samples")

        # Train for a few epochs
        for epoch in range(5):
            loss = loc_pretrainer.train_epoch(rel_vecs, spatial_classes, loc_optimizer, batch_size=32)
            print(f"  Epoch {epoch+1}: Loss = {loss:.4f}")
    else:
        print("Skipping pretraining (no annotation data provided)")

    # ===== STEP 4: Build loss and use in segmentation fine-tuning =====
    print("\n" + "=" * 80)
    print("STEP 4: Building ProtoAtlas Loss for Segmentation Fine-tuning")
    print("=" * 80)

    # Now freeze encoders for use in segmentation training
    for p in shape_disc.encoder.parameters():
        p.requires_grad = False
    for p in loc_disc.encoder.parameters():
        p.requires_grad = False

    atlas_loss = ProtoAtlasLoss(
        shape_disc=shape_disc,
        loc_disc=loc_disc,
        lambda_sup=1.0,
        lambda_s=0.5,
        lambda_l=0.5
    )

    print("ProtoAtlas loss created with frozen encoders")
    print("Ready to use in segmentation network fine-tuning!")
    print("\nUsage in training loop:")
    print("  pred = segmentation_model(image)  # (B, L, D, H, W) after softmax")
    print("  loss_dict = atlas_loss(pred, target=ground_truth, label_names=labels)")
    print("  total_loss = loss_dict['loss_total']")
    print("  total_loss.backward()")

    # ===== STEP 5: Demo with fake prediction =====
    print("\n" + "=" * 80)
    print("STEP 5: Demo with Fake Prediction")
    print("=" * 80)

    B = 1
    D, H, W = 64, 64, 64
    logits = torch.randn(B, L, D, H, W, device=device)
    pred = F.softmax(logits, dim=1)

    # Compute loss
    out = atlas_loss(pred, target=None, label_names=labels)
    print("Loss components:")
    for k, v in out.items():
        if 'scores' not in k:
            print(f"  {k}: {v.item() if isinstance(v, torch.Tensor) else v}")

    print("\nSetup complete! You can now:")
    print("1. Save the pretrained encoders: torch.save(shape_disc.encoder.state_dict(), 'shape_encoder.pth')")
    print("2. Save the pretrained encoders: torch.save(loc_disc.encoder.state_dict(), 'loc_encoder.pth')")
    print("3. Use atlas_loss in your segmentation training loop")

    return atlas_loss, shape_disc, loc_disc


if __name__ == "__main__":
    # Run the complete example
    example_with_real_annotations()