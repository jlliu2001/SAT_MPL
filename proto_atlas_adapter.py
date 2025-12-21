"""
ProtoAtlas Loss Adapter for train_mplseg.py

This adapter bridges the gap between:
- The training format in train_mplseg.py (sigmoid-based, dynamic labels with query_mask)
- The ProtoAtlasLoss format (expects specific input format)

Key Features:
1. Converts sigmoid probabilities to ProtoAtlas-compatible format
2. Handles query_mask for dynamic label selection
3. Combines supervised loss (Dice + CE) with ProtoAtlas loss (Shape + Location)
4. Supports configurable loss weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from einops import rearrange
import numpy as np
from proto_atlas import ProtoAtlasLoss


class ProtoAtlasLossAdapter(nn.Module):
    """
    Adapter to integrate ProtoAtlasLoss into train_mplseg.py training pipeline.

    This adapter:
    1. Computes original supervised loss (Dice + CE) as before
    2. Converts predictions to ProtoAtlas-compatible format
    3. Computes ProtoAtlas loss (Shape + Location) for valid labels
    4. Combines all losses with configurable weights

    Args:
        proto_atlas_loss: ProtoAtlasLoss instance with pretrained discriminators
        all_label_names: Fixed list of all label names in order (e.g., ["hippocampus", ...])
        lambda_atlas: Weight for ProtoAtlas loss relative to supervised loss
        lambda_shape: Weight for shape loss within ProtoAtlas loss
        lambda_loc: Weight for location loss within ProtoAtlas loss
        use_location_loss: Whether to compute location loss (requires all labels)
    """

    def __init__(self,
                 proto_atlas_loss: ProtoAtlasLoss,
                 all_label_names: List[str],
                 lambda_atlas: float = 0.5,
                 lambda_shape: float = 0.5,
                 lambda_loc: float = 0.5,
                 use_location_loss: bool = True):
        super().__init__()

        self.proto_atlas_loss = proto_atlas_loss
        self.all_label_names = all_label_names
        self.num_labels = len(all_label_names)

        # Loss weights
        self.lambda_atlas = lambda_atlas
        self.lambda_shape = lambda_shape
        self.lambda_loc = lambda_loc
        self.use_location_loss = use_location_loss

        # Update ProtoAtlasLoss internal weights
        self.proto_atlas_loss.lambda_s = lambda_shape
        self.proto_atlas_loss.lambda_l = lambda_loc

        # Create label name to index mapping
        self.label_to_idx = {name: idx for idx, name in enumerate(all_label_names)}

        self.deep_freeze_and_eval(self.proto_atlas_loss.shape_disc.encoder)
        self.deep_freeze_and_eval(self.proto_atlas_loss.loc_disc.encoder)


        print(f"✓ ProtoAtlasLossAdapter initialized:")
        print(f"  Labels: {all_label_names}")
        print(f"  lambda_atlas: {lambda_atlas}")
        print(f"  lambda_shape: {lambda_shape}")
        print(f"  lambda_loc: {lambda_loc}")
        print(f"  use_location_loss: {use_location_loss}")

    def deep_freeze_and_eval(self,module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
        # 递归处理所有子模块
        for child in module.children():
            self.deep_freeze_and_eval(child)

    def forward(self,
                logits: torch.Tensor,
                mask: torch.Tensor,
                query_mask: torch.Tensor,
                text: List[str],
                dice_loss: nn.Module,
                bce_w_logits_loss: nn.Module,
                weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Compute combined supervised + ProtoAtlas loss.

        Args:
            logits: (B, N, H, W, D) - Unsigmoided predictions
            mask: (B, N, H, W, D) - Ground truth binary masks
            query_mask: (B, N) - Binary mask indicating valid labels
            text: List[str] - Label names for this batch (length N)
            dice_loss: Dice loss calculator
            bce_w_logits_loss: BCE loss calculator
            weight: Weight for this scale (for multi-scale training)

        Returns:
            total_loss: Combined weighted loss for backprop
            unreduced_batch_dice_loss: (B,) per-sample dice loss
            unreduced_batch_ce_loss: (B,) per-sample CE loss
            atlas_info: Dict with detailed ProtoAtlas loss components
        """
        device = logits.device
        B, N, H, W, D = logits.shape

        # ========================================
        # 1. Compute Original Supervised Loss
        # ========================================
        prediction = torch.sigmoid(logits)  # (B, N, H, W, D)

        # Dice loss
        batch_dice_loss = dice_loss(prediction, mask)  # (B*N,)
        batch_dice_loss = rearrange(batch_dice_loss, '(b c) -> b c', b=B)  # (B, N)
        batch_dice_loss = batch_dice_loss * query_mask
        reduced_batch_dice_loss = torch.sum(batch_dice_loss) / (torch.sum(query_mask) + 1e-14)
        unreduced_batch_dice_loss = torch.sum(batch_dice_loss, dim=1) / (torch.sum(query_mask, dim=1) + 1e-14)

        # BCE loss
        batch_ce_loss = bce_w_logits_loss(logits, mask)  # (B, N, H, W, D)
        batch_ce_loss = torch.mean(batch_ce_loss, dim=(2, 3, 4))  # (B, N)
        batch_ce_loss = batch_ce_loss * query_mask
        reduced_batch_ce_loss = torch.sum(batch_ce_loss) / (torch.sum(query_mask) + 1e-14)
        unreduced_batch_ce_loss = torch.sum(batch_ce_loss, dim=1) / (torch.sum(query_mask, dim=1) + 1e-14)

        supervised_loss = reduced_batch_dice_loss + reduced_batch_ce_loss

        # ========================================
        # 2. Prepare Data for ProtoAtlas Loss
        # ========================================
        # Convert (B, N, H, W, D) -> (B, N, D, H, W) to match ProtoAtlas expectation
        pred_reordered = prediction.permute(0, 1, 4, 2, 3)  # (B, N, D, H, W)
        mask_reordered = mask.permute(0, 1, 4, 2, 3)  # (B, N, D, H, W)

        # ========================================
        # 3. Compute Shape Loss (per valid label)
        # ========================================
        shape_loss_total = 0.0
        shape_scores_dict = {}
        num_valid_labels = 0

        # DEBUG: Check input
        if torch.isnan(pred_reordered).any():
            print(f"[DEBUG] WARNING: pred_reordered contains NaN!")
            print(f"  NaN count: {torch.isnan(pred_reordered).sum().item()}")

        # Ensure shape_disc is in eval mode
        self.proto_atlas_loss.shape_disc.eval()

        for b in range(B):
            for n in range(N):
                if query_mask[b, n] > 0:  # Only compute for valid labels
                    label_name = text[n]

                    # Extract single label prediction: (1, 1, D, H, W)
                    single_pred = pred_reordered[b:b+1, n:n+1, :, :, :]
                    print('single_pred.shape:',single_pred.shape)
                    print('torch.unique(single_pred):',torch.unique(single_pred))
                    torch.save(single_pred, f'/data0/user/jlliu/git_pull_repos/SAT_MPL/pretrained_encoders/{label_name}_single_pred.pt')

                    # DEBUG: Check single prediction
                    if torch.isnan(single_pred).any() or torch.isinf(single_pred).any():
                        print(f"[DEBUG] WARNING: single_pred for {label_name} has NaN/Inf!")
                        print(f"  Min: {single_pred.min().item()}, Max: {single_pred.max().item()}")
                        print(f"  Mean: {single_pred.mean().item()}")
                        continue

                    # Check if prediction has any foreground
                    if single_pred.sum().item() < 1e-6:
                        print(f"[DEBUG] WARNING: {label_name} has near-zero prediction (all background)")
                        # Skip this label or use a small positive value
                        continue

                    # Compute shape score
                    try:
                        with torch.no_grad():  # Ensure no gradient computation
                            s_scores, emb = self.proto_atlas_loss.shape_disc(single_pred, label_name)

                        # DEBUG: Check outputs
                        if torch.isnan(s_scores).any() or torch.isinf(s_scores).any():
                            print(f"[DEBUG] NaN/Inf detected in shape scores for {label_name}!")
                            print(f"  s_scores: {s_scores}")
                            print(f"  emb stats: min={emb.min()}, max={emb.max()}, mean={emb.mean()}")
                            # Check embedding
                            if torch.isnan(emb).any():
                                print(f"  ERROR: Embedding contains NaN!")
                            continue

                        shape_loss = (1.0 - s_scores).mean()

                        # Check if loss is valid
                        if torch.isnan(shape_loss) or torch.isinf(shape_loss):
                            print(f"[DEBUG] NaN/Inf in shape_loss for {label_name}: {shape_loss}")
                            continue

                        shape_loss_total += shape_loss

                        # Store score for logging
                        if label_name not in shape_scores_dict:
                            shape_scores_dict[label_name] = []
                        shape_scores_dict[label_name].append(s_scores.detach().cpu().item())

                        num_valid_labels += 1
                    except Exception as e:
                        print(f"[DEBUG] Exception in shape loss for {label_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

        # Average shape loss over all valid labels
        if num_valid_labels > 0:
            shape_loss_avg = shape_loss_total / num_valid_labels
        else:
            shape_loss_avg = torch.tensor(0.0, device=device)

        # ========================================
        # 4. Compute Location Loss (requires all labels)
        # ========================================
        loc_loss = torch.tensor(0.0, device=device)
        loc_scores_list = []

        if self.use_location_loss:
            # Ensure loc_disc is in eval mode
            self.proto_atlas_loss.loc_disc.eval()

            # Check if we have all labels for each sample in batch
            # For location loss, we need ALL labels present

            for b in range(B):
                # Check if this sample has all labels
                sample_text = text  # Assuming text is consistent across batch
                sample_query_mask = query_mask[b]  # (N,)

                # Verify all labels are present and valid
                if sample_query_mask.sum() == N and N == self.num_labels:
                    # This sample has all labels, compute location loss
                    sample_pred = pred_reordered[b:b+1, :, :, :, :]  # (1, N, D, H, W)
                    

                    # DEBUG: Check sample prediction
                    if torch.isnan(sample_pred).any() or torch.isinf(sample_pred).any():
                        print(f"[DEBUG] WARNING: sample_pred for location has NaN/Inf!")
                        continue

                    try:
                        with torch.no_grad():  # Ensure no gradient computation
                            loc_scores, loc_emb = self.proto_atlas_loss.loc_disc(sample_pred)

                        # DEBUG: Check outputs
                        if torch.isnan(loc_scores).any() or torch.isinf(loc_scores).any():
                            print(f"[DEBUG] NaN/Inf detected in location scores!")
                            print(f"  loc_scores: {loc_scores}")
                            print(f"  loc_emb stats: min={loc_emb.min()}, max={loc_emb.max()}, mean={loc_emb.mean()}")
                            if torch.isnan(loc_emb).any():
                                print(f"  ERROR: Location embedding contains NaN!")
                            continue

                        sample_loc_loss = (1.0 - loc_scores).mean()

                        # Check if loss is valid
                        if torch.isnan(sample_loc_loss) or torch.isinf(sample_loc_loss):
                            print(f"[DEBUG] NaN/Inf in sample_loc_loss: {sample_loc_loss}")
                            continue

                        loc_loss += sample_loc_loss
                        loc_scores_list.append(loc_scores.detach().cpu().item())
                    except Exception as e:
                        print(f"[DEBUG] Exception in location loss for sample {b}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

            # Average location loss over samples that computed it
            if len(loc_scores_list) > 0:
                loc_loss = loc_loss / len(loc_scores_list)

        # ========================================
        # 5. Combine Losses
        # ========================================
        # ProtoAtlas loss = lambda_shape * shape_loss + lambda_loc * loc_loss
        atlas_loss = self.lambda_shape * shape_loss_avg + self.lambda_loc * loc_loss

        # Total loss = supervised_loss + lambda_atlas * atlas_loss
        total_loss = weight * (supervised_loss + self.lambda_atlas * atlas_loss)

        # ========================================
        # 6. Prepare Return Info
        # ========================================
        atlas_info = {
            'atlas_loss_total': (self.lambda_atlas * atlas_loss).detach().cpu().item(),
            'shape_loss': shape_loss_avg.detach().cpu().item(),
            'loc_loss': loc_loss.detach().cpu().item(),
            'shape_scores': shape_scores_dict,
            'loc_scores': loc_scores_list,
            'num_valid_labels': num_valid_labels,
            'num_loc_computed': len(loc_scores_list)
        }

        return (
            total_loss,
            weight * unreduced_batch_dice_loss,
            weight * unreduced_batch_ce_loss,
            atlas_info
        )


def create_proto_atlas_adapter(
    shape_prior_path: str,
    shape_encoder_path: str,
    loc_prior_path: str,
    loc_encoder_path: str,
    all_label_names: List[str],
    device: torch.device,
    lambda_atlas: float = 0.5,
    lambda_shape: float = 0.5,
    lambda_loc: float = 0.5,
    use_location_loss: bool = True,
    shape_emb_dim: int = 128,
    loc_emb_dim: int = 64,
    use_e3nn: bool = False
) -> ProtoAtlasLossAdapter:
    """
    Convenience function to create ProtoAtlasLossAdapter with pretrained components.

    Args:
        shape_prior_path: Path to shape_priors.npz
        shape_encoder_path: Path to pretrained shape encoder checkpoint
        loc_prior_path: Path to location_prior.npz
        loc_encoder_path: Path to pretrained location encoder checkpoint
        all_label_names: List of all label names
        device: torch device
        lambda_atlas: Weight for atlas loss
        lambda_shape: Weight for shape loss within atlas loss
        lambda_loc: Weight for location loss within atlas loss
        use_location_loss: Whether to use location loss
        shape_emb_dim: Shape embedding dimension
        loc_emb_dim: Location embedding dimension
        use_e3nn: Whether to use SE(3)-equivariant encoder for shape

    Returns:
        ProtoAtlasLossAdapter instance ready for training
    """
    from proto_atlas import ShapePrior, ShapeDisc, LocPrior, LocDisc, ProtoAtlasLoss

    print("=" * 80)
    print("Creating ProtoAtlasLossAdapter with Pretrained Components")
    print("=" * 80)

    # Load Shape Prior and Discriminator
    print(f"\n1. Loading Shape Prior from: {shape_prior_path}")
    shape_prior = ShapePrior.load_from_file(shape_prior_path)
    shape_prior.to(device)
    print(f"   ✓ Loaded shape priors for {len(shape_prior.labels)} labels")

    print(f"\n2. Building Shape Discriminator...")
    shape_disc = ShapeDisc(
        emb_dim=shape_emb_dim,
        shape_prior=shape_prior,
        beta=1.0,
        freeze_encoder=True,  # CRITICAL: Keep encoder frozen
        use_e3nn=use_e3nn
    ).to(device)

    print(f"\n3. Loading Pretrained Shape Encoder from: {shape_encoder_path}")
    shape_disc.load_pretrained_encoder(shape_encoder_path, device)

    # Load Location Prior and Discriminator
    print(f"\n4. Loading Location Prior from: {loc_prior_path}")
    loc_prior = LocPrior.load_from_file(loc_prior_path)
    loc_prior.to(device)
    print(f"   ✓ Loaded location prior with {loc_prior.K} protocol pairs")

    print(f"\n5. Building Location Discriminator...")
    loc_disc = LocDisc(
        num_labels=len(all_label_names),
        relation_pairs=loc_prior.relation_pairs,
        emb_dim=loc_emb_dim,
        loc_prior=loc_prior,
        w=1.0,
        freeze_encoder=True  # CRITICAL: Keep encoder frozen
    ).to(device)

    print(f"\n6. Loading Pretrained Location Encoder from: {loc_encoder_path}")
    loc_disc.load_pretrained_encoder(loc_encoder_path, device)

    # Build ProtoAtlasLoss
    print(f"\n7. Building ProtoAtlasLoss...")
    proto_atlas_loss = ProtoAtlasLoss(
        shape_disc=shape_disc,
        loc_disc=loc_disc,
        lambda_sup=1.0,  # Not used in adapter (we compute supervised loss separately)
        lambda_s=lambda_shape,
        lambda_l=lambda_loc,
        use_cross_entropy=False  # We use BCE in train_mplseg.py
    )

    # Build Adapter
    print(f"\n8. Building ProtoAtlasLossAdapter...")
    adapter = ProtoAtlasLossAdapter(
        proto_atlas_loss=proto_atlas_loss,
        all_label_names=all_label_names,
        lambda_atlas=lambda_atlas,
        lambda_shape=lambda_shape,
        lambda_loc=lambda_loc,
        use_location_loss=use_location_loss
    )

    print("\n" + "=" * 80)
    print("✓ ProtoAtlasLossAdapter created successfully!")
    print("=" * 80)

    return adapter


def test_proto_atlas_adapter(
    shape_encoder_path: str,
    shape_prior_path: str,
    loc_encoder_path: str,
    loc_prior_path: str,
    batch_size: int = 2,
    num_labels: int = 5,
    spatial_size: Tuple[int, int, int] = (64, 64, 64)
):
    """
    Test ProtoAtlasAdapter and ProtoAtlasLoss with random data to check for NaN/Inf issues.

    Args:
        shape_encoder_path: Path to pretrained shape encoder
        shape_prior_path: Path to shape prior file
        loc_encoder_path: Path to pretrained location encoder
        loc_prior_path: Path to location prior file
        batch_size: Batch size for test
        num_labels: Number of labels/classes
        spatial_size: (H, W, D) spatial dimensions
    """
    import numpy as np
    # from train.loss_softmax import DiceLoss, BCEWithLogitsLoss

    print("=" * 80)
    print("Testing ProtoAtlasAdapter and ProtoAtlasLoss")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Define test label names
    # You should replace these with your actual label names
    all_label_names = ["hippocampus", "amygdala", "caudate", "putamen",
                          "pallidum", "thalamus", "accumbens"]

    print(f"\nTest labels: {all_label_names}")

    # ========================================
    # 1. Create ProtoAtlasAdapter
    # ========================================
    print("\n" + "=" * 80)
    print("Step 1: Creating ProtoAtlasAdapter")
    print("=" * 80)

    try:
        adapter = create_proto_atlas_adapter(
            shape_prior_path=shape_prior_path,
            shape_encoder_path=shape_encoder_path,
            loc_prior_path=loc_prior_path,
            loc_encoder_path=loc_encoder_path,
            all_label_names=all_label_names,
            device=device,
            lambda_atlas=0.5,
            lambda_shape=0.5,
            lambda_loc=0.5,
            use_location_loss=True,
            shape_emb_dim=128,
            loc_emb_dim=64,
            use_e3nn=True
        )
        adapter.eval()  # Set to eval mode for testing
        print("\n✓ ProtoAtlasAdapter created successfully!")
    except Exception as e:
        print(f"\n✗ Failed to create adapter: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================
    # 2. Generate Random Test Data
    # ========================================
    print("\n" + "=" * 80)
    print("Step 2: Generating Random Test Data")
    print("=" * 80)

    B = batch_size
    N = num_labels
    H, W, D = spatial_size

    print(f"\nData shape: (B={B}, N={N}, H={H}, W={W}, D={D})")

    # Generate random prediction probabilities (0~1)
    # Use logits (before sigmoid) for the adapter
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate logits (will be converted to probabilities via sigmoid)
    # Using values in range [-3, 3] so sigmoid gives reasonable probabilities
    logits = torch.randn(B, N, H, W, D, device=device) * 2.0
    # logits=torch.load("/data0/user/jlliu/git_pull_repos/SAT_MPL/pretrained_encoders/thalamus_single_pred.pt")

    # Generate binary masks (0 or 1)
    # Create realistic masks with some foreground regions
    mask = torch.zeros(B, N, H, W, D, device=device)
    for b in range(B):
        for n in range(N):
            # Create a random foreground region (sphere-like)
            center = (
                np.random.randint(H//4, 3*H//4),
                np.random.randint(W//4, 3*W//4),
                np.random.randint(D//4, 3*D//4)
            )
            radius = np.random.randint(5, 15)

            # Create coordinate grid
            z, y, x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                torch.arange(D, device=device),
                indexing='ij'
            )

            # Create sphere mask
            dist = torch.sqrt(
                (x - center[2]).float()**2 +
                (y - center[1]).float()**2 +
                (z - center[0]).float()**2
            )
            mask[b, n] = (dist < radius).float()

    # Create query_mask (all labels are valid)
    query_mask = torch.ones(B, N, device=device)

    # Create text labels
    text = all_label_names

    print(f"\nGenerated data statistics:")
    print(f"  Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
    print(f"  Probabilities (after sigmoid): min={torch.sigmoid(logits).min().item():.4f}, max={torch.sigmoid(logits).max().item():.4f}")
    print(f"  Mask: min={mask.min().item():.0f}, max={mask.max().item():.0f}, sum={mask.sum().item():.0f}")
    print(f"  Query mask: all ones, shape={query_mask.shape}")

    # Check for NaN/Inf in input data
    print(f"\nInput data sanity check:")
    print(f"  Logits has NaN: {torch.isnan(logits).any().item()}")
    print(f"  Logits has Inf: {torch.isinf(logits).any().item()}")
    print(f"  Mask has NaN: {torch.isnan(mask).any().item()}")
    print(f"  Mask has Inf: {torch.isinf(mask).any().item()}")

    # ========================================
    # 3. Create Loss Functions
    # ========================================
    print("\n" + "=" * 80)
    print("Step 3: Creating Loss Functions")
    print("=" * 80)

    try:
        from train.loss import BinaryDiceLoss
        dice_loss = BinaryDiceLoss(reduction='none')
        bce_loss = nn.BCEWithLogitsLoss(reduction='none') # safe for amp
        print("✓ Loss functions created successfully!")
    except Exception as e:
        print(f"✗ Failed to create loss functions: {e}")
        print("  Note: Make sure train/loss_softmax.py exists with DiceLoss and BCEWithLogitsLoss")
        return

    # ========================================
    # 4. Run Forward Pass
    # ========================================
    print("\n" + "=" * 80)
    print("Step 4: Running Forward Pass")
    print("=" * 80)

    try:
        with torch.no_grad():  # No gradient computation for testing
            total_loss, unreduced_dice, unreduced_ce, atlas_info = adapter(
                logits=logits,
                mask=mask,
                query_mask=query_mask,
                text=text,
                dice_loss=dice_loss,
                bce_w_logits_loss=bce_loss,
                weight=1.0
            )

        print("\n✓ Forward pass completed!")

    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================
    # 5. Check Outputs for NaN/Inf
    # ========================================
    print("\n" + "=" * 80)
    print("Step 5: Checking Outputs for NaN/Inf")
    print("=" * 80)

    issues_found = False

    # Check total_loss
    print(f"\nTotal Loss:")
    print(f"  Value: {total_loss.item():.6f}")
    if torch.isnan(total_loss):
        print(f"  ✗ WARNING: Total loss is NaN!")
        issues_found = True
    elif torch.isinf(total_loss):
        print(f"  ✗ WARNING: Total loss is Inf!")
        issues_found = True
    else:
        print(f"  ✓ Total loss is valid")

    # Check unreduced_dice
    print(f"\nUnreduced Dice Loss:")
    print(f"  Shape: {unreduced_dice.shape}")
    print(f"  Values: {unreduced_dice.cpu().numpy()}")
    if torch.isnan(unreduced_dice).any():
        print(f"  ✗ WARNING: Contains NaN!")
        issues_found = True
    elif torch.isinf(unreduced_dice).any():
        print(f"  ✗ WARNING: Contains Inf!")
        issues_found = True
    else:
        print(f"  ✓ All values are valid")

    # Check unreduced_ce
    print(f"\nUnreduced CE Loss:")
    print(f"  Shape: {unreduced_ce.shape}")
    print(f"  Values: {unreduced_ce.cpu().numpy()}")
    if torch.isnan(unreduced_ce).any():
        print(f"  ✗ WARNING: Contains NaN!")
        issues_found = True
    elif torch.isinf(unreduced_ce).any():
        print(f"  ✗ WARNING: Contains Inf!")
        issues_found = True
    else:
        print(f"  ✓ All values are valid")

    # Check atlas_info
    print(f"\nAtlas Info:")
    for key, value in atlas_info.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
            if key in ['atlas_loss_total', 'shape_loss', 'loc_loss']:
                if np.isnan(value):
                    print(f"    ✗ WARNING: {key} is NaN!")
                    issues_found = True
                elif np.isinf(value):
                    print(f"    ✗ WARNING: {key} is Inf!")
                    issues_found = True
                else:
                    print(f"    ✓ {key} is valid")
        else:
            print(f"  {key}: {value}")

    # ========================================
    # 6. Summary
    # ========================================
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    if issues_found:
        print("\n✗ TEST FAILED: NaN or Inf values detected!")
        print("  Please review the warnings above and check:")
        print("  1. Pretrained encoder weights are valid")
        print("  2. Prior files contain valid values")
        print("  3. Input data is properly normalized")
    else:
        print("\n✓ TEST PASSED: All outputs are valid (no NaN or Inf)!")
        print("  The ProtoAtlasAdapter is working correctly.")

    print("\n" + "=" * 80)


def test_proto_MSE_atlas_adapter(
    shape_encoder_path: str,
    shape_prior_path: str,
    loc_encoder_path: str,
    loc_prior_path: str,
    batch_size: int = 2,
    num_labels: int = 5,
    spatial_size: Tuple[int, int, int] = (64, 64, 64)
):
    """
    Test ProtoAtlasAdapter and ProtoAtlasLoss with random data to check for NaN/Inf issues.

    Args:
        shape_encoder_path: Path to pretrained shape encoder
        shape_prior_path: Path to shape prior file
        loc_encoder_path: Path to pretrained location encoder
        loc_prior_path: Path to location prior file
        batch_size: Batch size for test
        num_labels: Number of labels/classes
        spatial_size: (H, W, D) spatial dimensions
    """
    import numpy as np
    # from train.loss_softmax import DiceLoss, BCEWithLogitsLoss

    print("=" * 80)
    print("Testing ProtoAtlasAdapter and ProtoAtlasLoss")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Define test label names
    # You should replace these with your actual label names
    all_label_names = ["hippocampus", "amygdala", "caudate", "putamen",
                          "pallidum", "thalamus", "accumbens"]

    print(f"\nTest labels: {all_label_names}")

    # ========================================
    # 1. Create ProtoAtlasAdapter
    # ========================================
    print("\n" + "=" * 80)
    print("Step 1: Creating ProtoAtlasAdapter")
    print("=" * 80)

    try:
        adapter = create_proto_atlas_mse_adapter(
            shape_encoder_path=shape_encoder_path,
            loc_prior_path=loc_prior_path,
            loc_encoder_path=loc_encoder_path,
            all_label_names=all_label_names,
            device=device,
            lambda_atlas=0.5,
            lambda_shape=0.5,
            lambda_loc=0.5,
            use_location_loss=True,
            shape_emb_dim=128,
            loc_emb_dim=64,
            use_e3nn=True
        )
        adapter.eval()  # Set to eval mode for testing
        print("\n✓ ProtoAtlasAdapter created successfully!")
    except Exception as e:
        print(f"\n✗ Failed to create adapter: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================
    # 2. Generate Random Test Data
    # ========================================
    print("\n" + "=" * 80)
    print("Step 2: Generating Random Test Data")
    print("=" * 80)

    B = batch_size
    N = num_labels
    H, W, D = spatial_size

    print(f"\nData shape: (B={B}, N={N}, H={H}, W={W}, D={D})")

    # Generate random prediction probabilities (0~1)
    # Use logits (before sigmoid) for the adapter
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate logits (will be converted to probabilities via sigmoid)
    # Using values in range [-3, 3] so sigmoid gives reasonable probabilities
    logits = torch.randn(B, N, H, W, D, device=device) * 2.0
    # logits=torch.load("/data0/user/jlliu/git_pull_repos/SAT_MPL/pretrained_encoders/thalamus_single_pred.pt")

    # Generate binary masks (0 or 1)
    # Create realistic masks with some foreground regions
    mask = torch.zeros(B, N, H, W, D, device=device)
    for b in range(B):
        for n in range(N):
            # Create a random foreground region (sphere-like)
            center = (
                np.random.randint(H//4, 3*H//4),
                np.random.randint(W//4, 3*W//4),
                np.random.randint(D//4, 3*D//4)
            )
            radius = np.random.randint(5, 15)

            # Create coordinate grid
            z, y, x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                torch.arange(D, device=device),
                indexing='ij'
            )

            # Create sphere mask
            dist = torch.sqrt(
                (x - center[2]).float()**2 +
                (y - center[1]).float()**2 +
                (z - center[0]).float()**2
            )
            mask[b, n] = (dist < radius).float()

    # Create query_mask (all labels are valid)
    query_mask = torch.ones(B, N, device=device)

    # Create text labels
    text = all_label_names

    print(f"\nGenerated data statistics:")
    print(f"  Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
    print(f"  Probabilities (after sigmoid): min={torch.sigmoid(logits).min().item():.4f}, max={torch.sigmoid(logits).max().item():.4f}")
    print(f"  Mask: min={mask.min().item():.0f}, max={mask.max().item():.0f}, sum={mask.sum().item():.0f}")
    print(f"  Query mask: all ones, shape={query_mask.shape}")

    # Check for NaN/Inf in input data
    print(f"\nInput data sanity check:")
    print(f"  Logits has NaN: {torch.isnan(logits).any().item()}")
    print(f"  Logits has Inf: {torch.isinf(logits).any().item()}")
    print(f"  Mask has NaN: {torch.isnan(mask).any().item()}")
    print(f"  Mask has Inf: {torch.isinf(mask).any().item()}")

    # ========================================
    # 3. Create Loss Functions
    # ========================================
    print("\n" + "=" * 80)
    print("Step 3: Creating Loss Functions")
    print("=" * 80)

    try:
        # dice_loss = DiceLoss(reduction='none')
        # bce_loss = BCEWithLogitsLoss(reduction='none')
        from train.loss import BinaryDiceLoss
        dice_loss = BinaryDiceLoss(reduction='none')
        bce_loss = nn.BCEWithLogitsLoss(reduction='none') # safe for amp
        print("✓ Loss functions created successfully!")
    except Exception as e:
        print(f"✗ Failed to create loss functions: {e}")
        print("  Note: Make sure train/loss_softmax.py exists with DiceLoss and BCEWithLogitsLoss")
        return

    # ========================================
    # 4. Run Forward Pass
    # ========================================
    print("\n" + "=" * 80)
    print("Step 4: Running Forward Pass")
    print("=" * 80)

    try:
        with torch.no_grad():  # No gradient computation for testing
            total_loss, unreduced_dice, unreduced_ce, atlas_info = adapter(
                logits=logits,
                mask=mask,
                query_mask=query_mask,
                text=text,
                dice_loss=dice_loss,
                bce_w_logits_loss=bce_loss,
                weight=1.0
            )

        print("\n✓ Forward pass completed!")

    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================
    # 5. Check Outputs for NaN/Inf
    # ========================================
    print("\n" + "=" * 80)
    print("Step 5: Checking Outputs for NaN/Inf")
    print("=" * 80)

    issues_found = False

    # Check total_loss
    print(f"\nTotal Loss:")
    print(f"  Value: {total_loss.item():.6f}")
    if torch.isnan(total_loss):
        print(f"  ✗ WARNING: Total loss is NaN!")
        issues_found = True
    elif torch.isinf(total_loss):
        print(f"  ✗ WARNING: Total loss is Inf!")
        issues_found = True
    else:
        print(f"  ✓ Total loss is valid")

    # Check unreduced_dice
    print(f"\nUnreduced Dice Loss:")
    print(f"  Shape: {unreduced_dice.shape}")
    print(f"  Values: {unreduced_dice.cpu().numpy()}")
    if torch.isnan(unreduced_dice).any():
        print(f"  ✗ WARNING: Contains NaN!")
        issues_found = True
    elif torch.isinf(unreduced_dice).any():
        print(f"  ✗ WARNING: Contains Inf!")
        issues_found = True
    else:
        print(f"  ✓ All values are valid")

    # Check unreduced_ce
    print(f"\nUnreduced CE Loss:")
    print(f"  Shape: {unreduced_ce.shape}")
    print(f"  Values: {unreduced_ce.cpu().numpy()}")
    if torch.isnan(unreduced_ce).any():
        print(f"  ✗ WARNING: Contains NaN!")
        issues_found = True
    elif torch.isinf(unreduced_ce).any():
        print(f"  ✗ WARNING: Contains Inf!")
        issues_found = True
    else:
        print(f"  ✓ All values are valid")

    # Check atlas_info
    print(f"\nAtlas Info:")
    for key, value in atlas_info.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
            if key in ['atlas_loss_total', 'shape_loss', 'loc_loss']:
                if np.isnan(value):
                    print(f"    ✗ WARNING: {key} is NaN!")
                    issues_found = True
                elif np.isinf(value):
                    print(f"    ✗ WARNING: {key} is Inf!")
                    issues_found = True
                else:
                    print(f"    ✓ {key} is valid")
        else:
            print(f"  {key}: {value}")

    # ========================================
    # 6. Summary
    # ========================================
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    if issues_found:
        print("\n✗ TEST FAILED: NaN or Inf values detected!")
        print("  Please review the warnings above and check:")
        print("  1. Pretrained encoder weights are valid")
        print("  2. Prior files contain valid values")
        print("  3. Input data is properly normalized")
    else:
        print("\n✓ TEST PASSED: All outputs are valid (no NaN or Inf)!")
        print("  The ProtoAtlasAdapter is working correctly.")

    print("\n" + "=" * 80)

class ProtoAtlasMSELossAdapter(nn.Module):
    """
    MSE-based Adapter for Encoder Alignment (following Problem 3 in FRAMEWORK_DISCUSSION_SUMMARY.md)

    核心思想：
    - 训练阶段：使用编码器对齐（MSE loss），而非原型距离
    - 推理阶段：使用原型评分（Mahalanobis距离）

    This adapter:
    1. Computes supervised loss (Dice + CE)
    2. Extracts embeddings from both prediction and GT using frozen encoders
    3. Computes MSE loss between pred_embedding and gt_embedding
    4. Does NOT use prototypes during training (prototypes are for inference only)

    Args:
        shape_encoder: Pretrained SE3Encoder (frozen)
        loc_encoder: Pretrained RelationEncoder (frozen)
        all_label_names: Fixed list of all label names
        protocol_pairs: List of (i, j) pairs for location encoding
        lambda_atlas: Weight for atlas loss (shape + loc) relative to supervised loss
        lambda_shape: Weight for shape alignment loss
        lambda_loc: Weight for location alignment loss
        use_location_loss: Whether to compute location loss
    """

    def __init__(self,
                 shape_encoder: nn.Module,
                 loc_encoder: nn.Module,
                 all_label_names: List[str],
                 protocol_pairs: Optional[List[Tuple[int, int]]] = None,
                 lambda_atlas: float = 0.5,
                 lambda_shape: float = 0.5,
                 lambda_loc: float = 0.5,
                 use_location_loss: bool = True):
        super().__init__()

        self.shape_encoder = shape_encoder
        self.loc_encoder = loc_encoder
        self.all_label_names = all_label_names
        self.num_labels = len(all_label_names)
        self.protocol_pairs = protocol_pairs

        # Loss weights
        self.lambda_atlas = lambda_atlas
        self.lambda_shape = lambda_shape
        self.lambda_loc = lambda_loc
        self.use_location_loss = use_location_loss

        # Freeze encoders (critical for training stability)
        for param in self.shape_encoder.parameters():
            param.requires_grad = False
        for param in self.loc_encoder.parameters():
            param.requires_grad = False

        self.shape_encoder.eval()
        self.loc_encoder.eval()

        # Create label name to index mapping
        self.label_to_idx = {name: idx for idx, name in enumerate(all_label_names)}

        print(f"✓ ProtoAtlasMSELossAdapter initialized (Encoder Alignment Mode):")
        print(f"  Labels: {all_label_names}")
        print(f"  lambda_atlas: {lambda_atlas}")
        print(f"  lambda_shape: {lambda_shape}")
        print(f"  lambda_loc: {lambda_loc}")
        print(f"  use_location_loss: {use_location_loss}")
        print(f"  ⚠️  Encoders are FROZEN and in eval mode")
        print(f"  ⚠️  Training uses MSE alignment, NOT prototype distances")

    def forward(self,
                logits: torch.Tensor,
                mask: torch.Tensor,
                query_mask: torch.Tensor,
                text: List[str],
                dice_loss: nn.Module,
                bce_w_logits_loss: nn.Module,
                weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Compute combined supervised + encoder alignment loss.

        Args:
            logits: (B, N, H, W, D) - Unsigmoided predictions
            mask: (B, N, H, W, D) - Ground truth binary masks
            query_mask: (B, N) - Binary mask indicating valid labels
            text: List[str] - Label names for this batch (length N)
            dice_loss: Dice loss calculator
            bce_w_logits_loss: BCE loss calculator
            weight: Weight for this scale

        Returns:
            total_loss: Combined weighted loss for backprop
            unreduced_batch_dice_loss: (B,) per-sample dice loss
            unreduced_batch_ce_loss: (B,) per-sample CE loss
            atlas_info: Dict with detailed loss components
        """
        device = logits.device
        B, N, H, W, D = logits.shape

        # ========================================
        # 1. Compute Original Supervised Loss
        # ========================================
        prediction = torch.sigmoid(logits)  # (B, N, H, W, D)

        # Dice loss
        batch_dice_loss = dice_loss(logits, mask)  # (B*N,)
        # print('batch_dice_loss.shape:',batch_dice_loss.shape)
        # batch_dice_loss = rearrange(batch_dice_loss, '(b c) -> b c', b=B)  # (B, N)
        query_weight = torch.tensor([1.0, 1.5, 1.0, 1.0, 1.5, 1.0, 2.0]).to(device=device)
        query_weight = query_weight.unsqueeze(0).repeat(B, 1)
        batch_dice_loss = batch_dice_loss * query_mask
        batch_dice_loss = batch_dice_loss * query_weight
        reduced_batch_dice_loss = torch.sum(batch_dice_loss) / (torch.sum(query_mask) + 1e-14)
        unreduced_batch_dice_loss = torch.sum(batch_dice_loss, dim=1) / (torch.sum(query_mask, dim=1) + 1e-14)

        # BCE loss
        batch_ce_loss = bce_w_logits_loss(logits, mask)  # (B, N, H, W, D)
        # print('batch_ce_loss.shape:',batch_dice_loss.shape)
        batch_ce_loss = torch.mean(batch_ce_loss, dim=(2, 3, 4))  # (B, N)
        batch_ce_loss = batch_ce_loss * query_mask
        reduced_batch_ce_loss = torch.sum(batch_ce_loss) / (torch.sum(query_mask) + 1e-14)
        unreduced_batch_ce_loss = torch.sum(batch_ce_loss, dim=1) / (torch.sum(query_mask, dim=1) + 1e-14)

        supervised_loss = reduced_batch_dice_loss + reduced_batch_ce_loss

        # ========================================
        # 2. Prepare Data for Encoder Alignment
        # ========================================
        # Convert (B, N, H, W, D) -> (B, N, D, H, W)
        pred_reordered = prediction.permute(0, 1, 4, 2, 3)  # (B, N, D, H, W)
        mask_reordered = mask.permute(0, 1, 4, 2, 3)  # (B, N, D, H, W)

        

        # ========================================
        # 3. Compute Shape Alignment Loss (MSE)
        # ========================================
        shape_loss_total = 0.0
        shape_mse_dict = {}
        num_valid_labels = 0

        with torch.no_grad():  # Encoders are frozen
            for b in range(B):
                for n in range(N):
                    if query_mask[b, n] > 0:
                        label_name = text[n]

                        # Extract single label: (1, 1, D, H, W)
                        single_pred = pred_reordered[b:b+1, n:n+1, :, :, :]


                        # single_pred=torch.load("/data0/user/jlliu/git_pull_repos/SAT_MPL/pretrained_encoders/thalamus_single_pred.pt")
                        single_gt = mask_reordered[b:b+1, n:n+1, :, :, :]

                        with torch.cuda.amp.autocast(enabled=False):
                        # 确保输入是 float32
                            single_pred = single_pred.float()
                            single_gt = single_gt.float()
                            # print('single_pred.dtype:',single_pred.dtype)
                            # print('single_gt.dtype:',single_gt.dtype)

                        
                        # 完全隔离的 no_grad context
                        

                        # Skip if prediction is too small
                        if single_pred.sum().item() < 1e-6:
                            continue

                        try:
                            with torch.no_grad():
                                # Extract embeddings from frozen encoders
                                emb_pred = self.shape_encoder(single_pred)  # (1, emb_dim)
                                emb_gt = self.shape_encoder(single_gt)      # (1, emb_dim)
                                # print(f"  emb_pred was: min={emb_pred.min()}, max={emb_pred.max()}")
                                # print(f"  emb_gt was: min={emb_gt.min()}, max={emb_gt.max()}")
                                # print('emb_pred:',torch.max(emb_pred))
                                # print('emb_gt:',torch.max(emb_gt))
                                emb_pred_norm = F.normalize(emb_pred, p=2, dim=-1)
                                emb_gt_norm = F.normalize(emb_gt, p=2, dim=-1)
                                # print(f"  emb_pred_norm was: min={emb_pred_norm.min()}, max={emb_pred_norm.max()}")
                                # print(f"  emb_gt_norm was: min={emb_gt_norm.min()}, max={emb_gt_norm.max()}")

                                shape_mse = 1 - F.cosine_similarity(emb_pred_norm, emb_gt_norm).mean()

                                # 🔥 核心：MSE对齐loss（不使用原型）
                                # shape_mse = F.mse_loss(emb_pred, emb_gt)
                                # print(f'shape_mse for{label_name}: {shape_mse}')

                                if torch.isnan(shape_mse) or torch.isinf(shape_mse):
                                    continue

                                shape_loss_total += shape_mse

                                # Store for logging
                                if label_name not in shape_mse_dict:
                                    shape_mse_dict[label_name] = []
                                shape_mse_dict[label_name].append(shape_mse.detach().cpu().item())

                                num_valid_labels += 1

                        except Exception as e:
                            print(f"[DEBUG] Exception in shape alignment for {label_name}: {e}")
                            continue

        # Average shape loss
        if num_valid_labels > 0:
            shape_loss_avg = shape_loss_total / num_valid_labels
        else:
            shape_loss_avg = torch.tensor(0.0, device=device, requires_grad=True)

        # ========================================
        # 4. Compute Location Alignment Loss (MSE)
        # ========================================
        loc_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loc_mse_list = []

        if self.use_location_loss and self.protocol_pairs is not None:
            with torch.no_grad():  # Encoders are frozen
                for b in range(B):
                    # Check if this sample has all labels
                    sample_query_mask = query_mask[b]  # (N,)

                    if sample_query_mask.sum() == N and N == self.num_labels:
                        # Extract predictions and GT
                        sample_pred = pred_reordered[b, :, :, :, :]  # (N, D, H, W)
                        sample_gt = mask_reordered[b, :, :, :, :]    # (N, D, H, W)

                        try:
                            # Extract location features
                            loc_feat_pred = self._extract_location_features(
                                sample_pred, self.protocol_pairs
                            )  # (K, 3) for basic, (K, 7) for enhanced
                            loc_feat_gt = self._extract_location_features(
                                sample_gt, self.protocol_pairs
                            )

                            if loc_feat_pred is None or loc_feat_gt is None:
                                continue

                            # Encode to embeddings
                            loc_feat_pred_tensor = torch.from_numpy(loc_feat_pred).unsqueeze(0).float().to(device)  # (1, K, 3or7)
                            loc_feat_gt_tensor = torch.from_numpy(loc_feat_gt).unsqueeze(0).float().to(device)

                            loc_emb_pred = self.loc_encoder(loc_feat_pred_tensor)  # (1, d_r)
                            loc_emb_gt = self.loc_encoder(loc_feat_gt_tensor)      # (1, d_r)

                            # 🔥 核心：MSE对齐loss（不使用原型）
                            sample_loc_mse = F.mse_loss(loc_emb_pred, loc_emb_gt)

                            if torch.isnan(sample_loc_mse) or torch.isinf(sample_loc_mse):
                                continue

                            loc_loss += sample_loc_mse
                            loc_mse_list.append(sample_loc_mse.detach().cpu().item())

                        except Exception as e:
                            print(f"[DEBUG] Exception in location alignment for sample {b}: {e}")
                            continue

            # Average location loss
            if len(loc_mse_list) > 0:
                loc_loss = loc_loss / len(loc_mse_list)

        # ========================================
        # 5. Combine Losses
        # ========================================
        # Atlas loss = lambda_shape * shape_mse + lambda_loc * loc_mse
        atlas_loss = self.lambda_shape * shape_loss_avg + self.lambda_loc * loc_loss

        # Total loss = supervised_loss + lambda_atlas * atlas_loss
        total_loss = weight * (supervised_loss + self.lambda_atlas * atlas_loss)

        # ========================================
        # 6. Prepare Return Info
        # ========================================
        atlas_info = {
            'atlas_loss_total': (self.lambda_atlas * atlas_loss).detach().cpu().item(),
            'shape_loss': shape_loss_avg.detach().cpu().item(),
            'loc_loss': loc_loss.detach().cpu().item(),
            'shape_mse': shape_mse_dict,
            'loc_mse': loc_mse_list,
            'num_valid_labels': num_valid_labels,
            'num_loc_computed': len(loc_mse_list),
            'loss_type': 'MSE_alignment'  # 标识这是MSE对齐loss
        }

        return (
            total_loss,
            weight * unreduced_batch_dice_loss,
            weight * unreduced_batch_ce_loss,
            atlas_info
        )

    def _extract_location_features(self, masks: torch.Tensor, protocol_pairs: List[Tuple[int, int]]) -> Optional[np.ndarray]:
        """
        Extract enhanced location features from masks with left-right split

        Process:
        1. Split 7-channel masks into 14-channel (left-right split)
        2. Extract enhanced features (centroid + adjacency) for protocol pairs
        3. Return features for encoder

        Args:
            masks: (N, D, H, W) - probability masks for N=7 bilateral labels
            protocol_pairs: List of (i, j) pairs for 14 L-R labels (0-indexed)

        Returns:
            features: (K, 7) array of enhanced relation vectors
                      [rel_pos(3), adj_ratio(1), adj_dir(3)]
                      or None if extraction fails
        """
        try:
            from utils.bilateral_split_utils import split_probability_map
            from location_encoder_enhanced import extract_enhanced_location_features

            # Ensure masks are on correct device
            device = masks.device
            N, D, H, W = masks.shape

            # Validate input: should be 7 channels for 7 bilateral structures
            if N != 7:
                print(f"[DEBUG] Warning: Expected 7 channels, got {N}")
                # Fallback to basic extraction if not 7 channels
                return self._extract_location_features_basic(masks, protocol_pairs)

            # Step 1: Split 7-channel probability map into 14-channel (L-R split)
            # Determine L-R axis (assume RAS+ orientation: W axis, left = higher index)
            lr_axis = 2  # W dimension
            lr_increasing = True

            split_masks = split_probability_map(
                prob_map=masks,
                lr_axis=lr_axis,
                lr_increasing=lr_increasing,
                method='argmax'  # Use hard split for stability
            )  # (14, D, H, W)

            # Step 2: Convert to numpy for feature extraction
            split_masks_np = split_masks.cpu().numpy()

            # Step 3: Extract enhanced features using protocol pairs
            # protocol_pairs are in 0-indexed format, but we need 1-indexed for extraction
            protocol_pairs_labels = [(i+1, j+1) for (i, j) in protocol_pairs]

            # Create a fake segmentation volume from probability map for feature extraction
            # Use argmax to get hard labels (0-14, where 0 is background)
            hard_labels = np.argmax(
                np.concatenate([
                    np.zeros((1, D, H, W), dtype=np.float32),  # Background channel
                    split_masks_np
                ], axis=0),
                axis=0
            ).astype(np.int32)  # (D, H, W) with labels 0-14

            # Extract enhanced features
            enhanced_features = extract_enhanced_location_features(
                segmentation=hard_labels,
                protocol_pairs=protocol_pairs_labels,
                use_probability=False,
                prob_threshold=0.5
            )  # (K, 7) - [rel_pos(3), adj_ratio(1), adj_dir(3)]

            return enhanced_features.astype(np.float32)

        except Exception as e:
            print(f"[DEBUG] Failed to extract enhanced location features: {e}")
            import traceback
            traceback.print_exc()

            # Fallback to basic extraction
            print(f"[DEBUG] Falling back to basic location feature extraction")
            return self._extract_location_features_basic(masks, protocol_pairs)

    def _extract_location_features_basic(self, masks: torch.Tensor, protocol_pairs: List[Tuple[int, int]]) -> Optional[np.ndarray]:
        """
        Fallback: Extract basic location features (centroid only) without L-R split

        Args:
            masks: (N, D, H, W) - probability masks
            protocol_pairs: List of (i, j) pairs

        Returns:
            features: (K, 3) array of relative position vectors
                      or None if extraction fails
        """
        try:
            masks_np = masks.cpu().numpy()
            features = []

            for (i, j) in protocol_pairs:
                if i >= masks_np.shape[0] or j >= masks_np.shape[0]:
                    # Index out of range, use zero vector
                    features.append(np.zeros(3, dtype=np.float32))
                    continue

                mask_i = masks_np[i]  # (D, H, W)
                mask_j = masks_np[j]

                # Compute centroids
                coords_i = np.argwhere(mask_i > 0.5)
                coords_j = np.argwhere(mask_j > 0.5)

                if len(coords_i) == 0 or len(coords_j) == 0:
                    # If empty, use zero vector
                    features.append(np.zeros(3, dtype=np.float32))
                    continue

                centroid_i = coords_i.mean(axis=0)  # (3,)
                centroid_j = coords_j.mean(axis=0)

                # Relative position
                rel_pos = centroid_i - centroid_j
                features.append(rel_pos.astype(np.float32))

            return np.array(features)  # (K, 3)

        except Exception as e:
            print(f"[DEBUG] Failed to extract basic location features: {e}")
            return None


def create_proto_atlas_mse_adapter(
    shape_encoder_path: str,
    loc_encoder_path: str,
    loc_prior_path: str,  # Need this for protocol_pairs
    all_label_names: List[str],
    device: torch.device,
    lambda_atlas: float = 0.5,
    lambda_shape: float = 0.5,
    lambda_loc: float = 0.5,
    use_location_loss: bool = True,
    shape_emb_dim: int = 128,
    loc_emb_dim: int = 64,
    use_e3nn: bool = True
) -> ProtoAtlasMSELossAdapter:
    """
    Create ProtoAtlasMSELossAdapter with pretrained encoders.

    核心区别：
    - 不加载原型（不需要shape_prior和loc_prior用于loss计算）
    - 只加载预训练的编码器
    - 训练时使用MSE对齐，推理时才用原型评分

    Args:
        shape_encoder_path: Path to pretrained shape encoder checkpoint
        loc_encoder_path: Path to pretrained location encoder checkpoint
        loc_prior_path: Path to location_prior.npz (for protocol_pairs)
        all_label_names: List of all label names
        device: torch device
        lambda_atlas: Weight for atlas loss
        lambda_shape: Weight for shape alignment loss
        lambda_loc: Weight for location alignment loss
        use_location_loss: Whether to use location loss
        shape_emb_dim: Shape embedding dimension
        loc_emb_dim: Location embedding dimension
        use_e3nn: Whether to use SE(3)-equivariant encoder

    Returns:
        ProtoAtlasMSELossAdapter instance
    """
    from proto_atlas import SE3Encoder

    print("=" * 80)
    print("Creating ProtoAtlasMSELossAdapter (MSE Alignment Mode)")
    print("=" * 80)

    # Load Shape Encoder
    print(f"\n1. Loading Shape Encoder from: {shape_encoder_path}")
    shape_encoder = SE3Encoder(emb_dim=shape_emb_dim, use_e3nn=use_e3nn).to(device)

    checkpoint = torch.load(shape_encoder_path, map_location=device)
    shape_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    shape_encoder.eval()
    for p in shape_encoder.parameters():
            p.requires_grad = False
    print(f"   ✓ Loaded pretrained shape encoder")

    # Load Location Encoder
    print(f"\n2. Loading Location Encoder from: {loc_encoder_path}")

    # Load location prior to get protocol_pairs
    import numpy as np
    loc_prior_data = np.load(loc_prior_path)
    protocol_pairs = loc_prior_data['protocol_pairs'].tolist()
    K = len(protocol_pairs)
    print(f"   ✓ Loaded {K} protocol pairs from location prior")

    # Determine if we're using enhanced encoder (7-dim) or basic (3-dim)
    # Try to infer from checkpoint
    loc_checkpoint = torch.load(loc_encoder_path, map_location=device)

    # Check if it's enhanced encoder by looking at the first layer weight shape
    try:
        first_layer_weight = loc_checkpoint['encoder_state_dict']['relation_mlp.0.weight']
        input_dim = first_layer_weight.shape[1]  # Input dimension

        if input_dim == 7:
            print(f"   Detected Enhanced Location Encoder (7-dim input)")
            from location_encoder_enhanced import EnhancedRelationEncoder
            loc_encoder = EnhancedRelationEncoder(K=K, d_r=loc_emb_dim).to(device)
        else:
            print(f"   Detected Basic Location Encoder (3-dim input)")
            from pretrain_loc_encoder import RelationEncoder
            loc_encoder = RelationEncoder(K=K, d_r=loc_emb_dim).to(device)

    except Exception as e:
        print(f"   Warning: Could not determine encoder type, using basic encoder: {e}")
        from pretrain_loc_encoder import RelationEncoder
        loc_encoder = RelationEncoder(K=K, d_r=loc_emb_dim).to(device)

    loc_encoder.load_state_dict(loc_checkpoint['encoder_state_dict'])
    loc_encoder.eval()
    for p in loc_encoder.parameters():
            p.requires_grad = False
    print(f"   ✓ Loaded pretrained location encoder")

    # Build Adapter
    print(f"\n3. Building ProtoAtlasMSELossAdapter...")
    adapter = ProtoAtlasMSELossAdapter(
        shape_encoder=shape_encoder,
        loc_encoder=loc_encoder,
        all_label_names=all_label_names,
        protocol_pairs=protocol_pairs,
        lambda_atlas=lambda_atlas,
        lambda_shape=lambda_shape,
        lambda_loc=lambda_loc,
        use_location_loss=use_location_loss
    )

    print("\n" + "=" * 80)
    print("✓ ProtoAtlasMSELossAdapter created successfully!")
    print("  Training mode: Encoder Alignment (MSE loss)")
    print("  Inference mode: Use prototypes for scoring (implement separately)")
    print("=" * 80)

    return adapter


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test ProtoAtlasAdapter with random data")
    parser.add_argument("--shape_encoder", type=str, required=True,
                        help="Path to pretrained shape encoder checkpoint")
    parser.add_argument("--shape_prior", type=str, required=True,
                        help="Path to shape_priors.npz file")
    parser.add_argument("--loc_encoder", type=str, required=True,
                        help="Path to pretrained location encoder checkpoint")
    parser.add_argument("--loc_prior", type=str, required=True,
                        help="Path to location_prior.npz file")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for testing (default: 2)")
    parser.add_argument("--num_labels", type=int, default=5,
                        help="Number of labels/classes (default: 5)")
    # parser.add_argument("--spatial_h", type=int, default=64,
    #                     help="Spatial height dimension (default: 64)")
    # parser.add_argument("--spatial_w", type=int, default=64,
    #                     help="Spatial width dimension (default: 64)")
    # parser.add_argument("--spatial_d", type=int, default=64,
    #                     help="Spatial depth dimension (default: 64)")

    args = parser.parse_args()

    test_proto_atlas_adapter(
        shape_encoder_path=args.shape_encoder,
        shape_prior_path=args.shape_prior,
        loc_encoder_path=args.loc_encoder,
        loc_prior_path=args.loc_prior,
        batch_size=args.batch_size,
        num_labels=args.num_labels,
        spatial_size=(96, 96, 96)
    )
    # test_proto_MSE_atlas_adapter(
    #     shape_encoder_path=args.shape_encoder,
    #     shape_prior_path=args.shape_prior,
    #     loc_encoder_path=args.loc_encoder,
    #     loc_prior_path=args.loc_prior,
    #     batch_size=args.batch_size,
    #     num_labels=args.num_labels,
    #     spatial_size=(96, 96, 96)
    # )