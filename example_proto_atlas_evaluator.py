#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: Using ProtoAtlasEvaluator for Segmentation Quality Scoring

This script demonstrates how to use the modified ProtoAtlasEvaluator to score
segmentation results without ground truth, based on shape and location prototypes.

Prerequisites:
1. Trained shape encoder checkpoint (from pretrain_shape_encoder.py)
2. Shape prototypes/priors (from extract_shape_prototypes_fixed.py)
3. Trained location encoder checkpoint (from pretrain_enhanced_loc_encoder.py)
4. Location prototypes/priors (from extract_location_prior.py or pretrain_enhanced_loc_encoder.py)

Features:
- Automatic downsampling for large images to avoid CUDA out of memory errors
  (images with any dimension > max_size will be downsampled using nearest-neighbor interpolation)
- Default max_size is 200, adjustable via --max_size parameter
- Example: [256,256,256] will be downsampled to approximately [128,128,128] with max_size=200
"""

import os
import argparse
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from pathlib import Path
import pandas as pd
from glob import glob
from scipy.ndimage import zoom

from proto_atlas import ProtoAtlasEvaluator


def load_prediction(pred_path: str, num_classes: int = 7, max_size: int = 200, use_downsample=False) -> np.ndarray:
    """
    Load predicted segmentation from NIfTI file.

    Args:
        pred_path: Path to prediction file (can be hard labels or probabilities)
        num_classes: Number of classes (including background)
        max_size: Maximum allowed size for any dimension (will downsample if exceeded)

    Returns:
        pred: (C, D, H, W) probability map
    """
    nii = nib.load(pred_path)
    nii = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    data=np.round(data)

    # Check if downsampling is needed to avoid CUDA out of memory
    original_shape = data.shape
    needs_downsample = any(dim > max_size for dim in original_shape[:3])  # Check first 3 dimensions

    if needs_downsample and use_downsample:
        # Calculate downsampling factors for spatial dimensions
        # if data.ndim == 3:
        #     # Hard labels: (D, H, W)
        #     # zoom_factors = [min(1.0, max_size / dim) for dim in original_shape]
        #     zoom_factors = [0.85 for dim in original_shape]
        # elif data.ndim == 4:
        #     # Probability map: (D, H, W, C)
        #     zoom_factors = [min(1.0, max_size / dim) for dim in original_shape[:3]] + [1.0]
        # else:
        #     zoom_factors = [1.0] * data.ndim

        # print(f"  Downsampling required: {original_shape} -> factors {[f'{z:.3f}' for z in zoom_factors]}")

        # # Perform nearest-neighbor downsampling (order=0 for label preservation)
        # data = zoom(data, zoom_factors, order=0)

        

        coords = np.where(data>0)
        y_min, y_max = coords[0].min(), coords[0].max() + 1
        x_min, x_max = coords[1].min(), coords[1].max() + 1
        z_min, z_max = coords[2].min(), coords[2].max() + 1

        # 添加一些边界padding，使可视化更清晰
        padding = 5
        y_min = max(0, y_min - padding)
        y_max = min(data.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(data.shape[1], x_max + padding)
        z_min = max(0, z_min - padding)
        z_max = min(data.shape[2], z_max + padding)
        data=data[y_min:y_max, x_min:x_max, z_min:z_max]
        print(f"  Downsampled shape: {data.shape}")

    data[data>7]=0
    # data = data.astype(np.float32)

    loc_data=data.astype(np.int32)
    affine = nii.affine
    spacing=nii.header['pixdim'][1:4]
    

    # Split bilateral labels (1-7) -> L-R labels (1-14)
    split_data = split_bilateral_labels(loc_data, affine)

    # Check if data is hard labels or probabilities
    if data.ndim == 3:
        # Hard labels: convert to one-hot probabilities
        print(f"Converting hard labels to probabilities...")
        print('data.shape:',np.unique(data))

        data_int = data.astype(np.int32)

        # Create one-hot encoding
        prob_map = np.zeros((num_classes, *data.shape), dtype=np.float32)
        for c in range(num_classes):
            prob_map[c] = (data_int == c+1).astype(np.float32)
        loc_prob_map = np.zeros((num_classes*2, *data.shape), dtype=np.float32)
        for c in range(num_classes*2):
            loc_prob_map[c] = (split_data == c+1).astype(np.float32)




    elif data.ndim == 4:
        # Already probability map
        print(f"Using probability map...")
        prob_map = data.transpose(3, 0, 1, 2).astype(np.float32)  # (C, D, H, W)

        # Ensure probabilities sum to 1
        prob_sum = prob_map.sum(axis=0, keepdims=True)
        prob_map = prob_map / (prob_sum + 1e-8)

    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    return prob_map,loc_prob_map,spacing

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


def evaluate_single_file(pred_path, evaluator, label_names, device, num_classes=7, max_size=200, verbose=True,use_downsample=False):
    """
    Evaluate a single segmentation file.

    Args:
        pred_path: Path to the prediction file
        evaluator: ProtoAtlasEvaluator instance
        label_names: List of label names
        device: torch device
        num_classes: Number of classes (including background)
        max_size: Maximum allowed size for any dimension (will downsample if exceeded)
        verbose: Whether to print detailed results

    Returns:
        scores: Dictionary containing all evaluation scores
        file_name: Name of the evaluated file
    """
    # file_name = os.path.basename(pred_path)
    # 自动提取file_name为带有pred.nii.gz的上层序列名
    # 例：/.../CANDI_1141.nii/CANDI_1141.nii/mri/pred.nii.gz -> "CANDI_1141"
    parts = os.path.normpath(pred_path).split(os.sep)
    file_name = None
    if len(parts) >= 2 and parts[-1] == "pred.nii.gz":
        # 往上查找第一个以".nii"结尾的目录名
        for i in range(len(parts)-2, -1, -1):
            if parts[i].endswith(".nii"):
                file_name = os.path.splitext(parts[i])[0]
                break
        if file_name is None:
            # fallback: 用 pred.nii.gz 的上一级目录
            file_name = parts[-2]
    else:
        file_name = os.path.basename(pred_path)
    # file_name = os.path.basename(pred_path)

    if verbose:
        print("\n" + "="*80)
        print(f"Evaluating: {file_name}")
        print("="*80)

    # Load prediction
    pred_data, loc_pred_data, spacing = load_prediction(pred_path, num_classes=num_classes, max_size=max_size,use_downsample=use_downsample)

    if verbose:
        print(f"Prediction shape: {pred_data.shape}")
        print(f"Prediction for location encoder shape: {loc_pred_data.shape}")

    # Remove background channel (index 0)
    # pred_data = pred_data[1:]  # (L, D, H, W)
    # loc_pred_data = loc_pred_data[1:]

    # Convert to tensor and add batch dimension
    pred_tensor = torch.from_numpy(pred_data).unsqueeze(0).float()  # (1, L, D, H, W)
    loc_pred_tensor = torch.from_numpy(loc_pred_data).unsqueeze(0).float()  # (1, L, D, H, W)

    # Evaluate
    evaluator.load_spacing(spacing)

    scores = evaluator.evaluate(pred=pred_tensor, pred_loc=loc_pred_tensor, label_names=label_names)

    if verbose:
        # Display results
        print("\n--- Per-Label Shape Scores ---")
        for label_name, label_scores in scores['per_label'].items():
            shape_score = label_scores['shape_score']
            print(f"  {label_name:15s}: {shape_score:.4e}")

        print("\n--- Location Score ---")
        print(f"  Global location score: {scores['location']['score']:.4e}")

        print("\n--- Overall Scores ---")
        print(f"  Mean shape score:  {scores['shape_mean']:.4e}")
        print(f"  Overall score:     {scores['overall']:.4e}")

    return scores, file_name


def save_batch_results_to_xlsx(results_list, output_path, label_names):
    """
    Save batch evaluation results to an xlsx file.

    Args:
        results_list: List of (file_name, scores) tuples
        output_path: Path to save the xlsx file
        label_names: List of label names
    """
    # Define region names (7 bilateral brain regions)
    region_names = ['hippocampus', 'amygdala', 'caudate', 'putamen',
                   'pallidum', 'thalamus', 'accumbens']

    # Prepare data for DataFrame
    data = []

    for file_name, scores in results_list:
        row = {'file_name': file_name}

        # Add per-label shape scores
        for label_name in label_names:
            if label_name in scores['per_label']:
                shape_score = scores['per_label'][label_name]['shape_score']
                row[f'{label_name}_shape_score'] = shape_score
            else:
                row[f'{label_name}_shape_score'] = None

        # Add mean shape score
        row['mean_shape_score'] = scores['shape_mean']

        # Add overall location score
        row['location_score'] = scores['location']['score']

        # Add per-region location scores (if available)
        per_region_scores = scores['location'].get('per_region_scores', {})
        for region_name in region_names:
            if region_name in per_region_scores:
                row[f'{region_name}_location'] = per_region_scores[region_name]
            else:
                row[f'{region_name}_location'] = None

        # Add overall score
        row['overall_score'] = scores['overall']

        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Reorder columns: file_name, shape scores, mean_shape, location scores (overall + per-region), overall
    columns = ['file_name']
    for label_name in label_names:
        columns.append(f'{label_name}_shape_score')
    columns.append('mean_shape_score')
    columns.append('location_score')
    for region_name in region_names:
        columns.append(f'{region_name}_location')
    columns.append('overall_score')

    df = df[columns]

    # Save to xlsx
    df.to_excel(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"Total files processed: {len(results_list)}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation quality using ProtoAtlasEvaluator')

    # Input
    parser.add_argument('--pred_path', type=str, required=True,
                       help='Path to predicted segmentation NIfTI file or folder (for batch mode)')
    parser.add_argument('--batch_mode', action='store_true',
                       help='Enable batch processing mode (pred_path should be a folder)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for saving xlsx results (default: current directory)')

    # Checkpoints and priors
    parser.add_argument('--shape_encoder_path', type=str,default="/data0/user/jlliu/git_pull_repos/SAT_MPL/pretrained_encoders/shape_encoder_v2/shape_encoder_best.pth",
                       help='Path to shape encoder checkpoint (.pth)')
    parser.add_argument('--shape_prior_path', type=str, default="/data0/user/jlliu/git_pull_repos/SAT_MPL/pretrained_encoders/shape_encoder_v2/shape_priors.npz",
                       help='Path to shape prior file (.npz)')
    parser.add_argument('--loc_encoder_path', type=str,default="/data0/user/jlliu/git_pull_repos/SAT_MPL/pretrained_encoders/loc_encoder_v2/enhanced_loc_encoder_best.pth",
                       help='Path to location encoder checkpoint (.pth)')
    parser.add_argument('--loc_prior_path', type=str, default="/data0/user/jlliu/git_pull_repos/SAT_MPL/pretrained_encoders/loc_encoder_v2/enhanced_location_prior.npz",
                       help='Path to location prior file (.npz)')

    # Options
    parser.add_argument('--use_enhanced_location', action='store_true',
                       help='Use enhanced location features (7-dim with adjacency)')
    parser.add_argument('--use_se3_shape', action='store_true',
                       help='Use SE(3)-equivariant shape encoder')
    parser.add_argument('--distance_metric', type=str, default='mahalanobis',
                       choices=['mahalanobis', 'cosine'],
                       help='Distance metric: "mahalanobis" (default) or "cosine" (more robust for small samples)')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Scaling for shape Mahalanobis distance (default: 1.0, only used with mahalanobis metric)')
    parser.add_argument('--w', type=float, default=0.5,
                       help='Scaling for location Mahalanobis distance (default: 1.0, only used with mahalanobis metric)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Weight for shape score in overall score (default: 0.5)')
    parser.add_argument('--max_size', type=int, default=200,
                       help='Maximum allowed size for any dimension (will downsample if exceeded to avoid CUDA OOM, default: 200)')
    parser.add_argument('--use_downsample', action='store_true',
                       help='Use downsample for input ')

    # Direct location mode options
    parser.add_argument('--use_direct_location', action='store_true',
                       help='Use direct location feature comparison without encoder (recommended for ground truth)')
    parser.add_argument('--direct_location_method', type=str, default='method_a',
                       choices=['method_a', 'method_c'],
                       help='Direct location scoring method: "method_a" (per-pair standardized, default) or "method_c" (weighted Euclidean)')
    parser.add_argument('--location_mean_path', type=str, default=None,
                       help='Path to location mean prior file (.npz) for direct mode. If not specified, will use loc_prior_path')

    # Labels
    parser.add_argument('--labels', nargs='+',
                       default=["hippocampus", "amygdala", "caudate", "putamen",
                               "pallidum", "thalamus", "accumbens"],
                       help='List of label names (excluding background)')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Number of classes
    # num_classes = len(args.labels) + 1
    num_classes = len(args.labels) 

    # ========================================
    # Load ProtoAtlasEvaluator
    # ========================================
    print("\n" + "="*80)
    print("Loading ProtoAtlasEvaluator")
    print("="*80)

    use_loc = True
    use_shape = False

    # For non-batch mode, we need spacing from the first file
    # For batch mode, we'll load evaluator with default spacing and update per file
    if not args.batch_mode:
        # Single file mode - get spacing from the file
        _, _, spacing = load_prediction(args.pred_path, num_classes=num_classes, max_size=args.max_size, use_downsample=args.use_downsample)
    else:
        # Batch mode - use default spacing, will be updated per file if needed
        spacing = None

    # Determine which location prior to use
    if args.use_direct_location:
        # Use location_mean_path if specified, otherwise use loc_prior_path
        loc_prior_path_to_use = args.location_mean_path if args.location_mean_path is not None else args.loc_prior_path
        print(f"\n✓ Direct location mode enabled")
        print(f"  Method: {args.direct_location_method}")
        print(f"  Using location prior: {loc_prior_path_to_use}")
    else:
        loc_prior_path_to_use = args.loc_prior_path

    evaluator = ProtoAtlasEvaluator.from_checkpoints(
        shape_encoder_path=args.shape_encoder_path,
        shape_prior_path=args.shape_prior_path,
        loc_encoder_path=args.loc_encoder_path,
        loc_prior_path=loc_prior_path_to_use,
        use_enhanced_location=args.use_enhanced_location,
        use_se3_shape=args.use_se3_shape,
        beta=args.beta,
        w=args.w,
        alpha=args.alpha,
        device=device,
        spacing=spacing,
        use_loc=use_loc,
        use_shape=use_shape,
        distance_metric=args.distance_metric,
        use_direct_location=args.use_direct_location,
        direct_location_method=args.direct_location_method
    )

    print(f"\n✓ ProtoAtlasEvaluator loaded successfully")
    print(f"  - Shape encoder: {'SE(3)-equivariant' if args.use_se3_shape else '3D CNN'}")

    if args.use_direct_location:
        print(f"  - Location mode: Direct ({args.direct_location_method})")
        print(f"  - Location features: {'Enhanced (7-dim)' if args.use_enhanced_location else 'Simple (3-dim)'}")
    else:
        print(f"  - Location encoder: {'Enhanced (7-dim)' if args.use_enhanced_location else 'Simple (3-dim)'}")
        print(f"  - Distance metric: {args.distance_metric}")

    if args.distance_metric == 'mahalanobis' or args.use_direct_location:
        print(f"  - Beta (shape scaling): {args.beta}")
        print(f"  - W (location scaling): {args.w}")
    print(f"  - Alpha (combination weight): {args.alpha}")

    # ========================================
    # Process files
    # ========================================
    if args.batch_mode:
        # Batch processing mode
        print("\n" + "="*80)
        print("Batch Processing Mode")
        print("="*80)

        # Get all .nii.gz files in the folder
        pred_folder = args.pred_path
        if not os.path.isdir(pred_folder):
            raise ValueError(f"pred_path should be a folder in batch mode: {pred_folder}")

        # Find all .nii.gz files
        # nii_files = glob(os.path.join(pred_folder, "*.nii.gz"))
        nii_files = glob(os.path.join(pred_folder, "seg_*.nii.gz"))
        # nii_files = glob(os.path.join(pred_folder, "**", "pred.nii.gz"), recursive=True)


        if len(nii_files) == 0:
            print(f"No .nii.gz files found in {pred_folder}")
            return

        print(f"Found {len(nii_files)} .nii.gz files to process")

        # Process each file
        results_list = []
        for i, nii_file in enumerate(nii_files, 1):
            print(f"\n[{i}/{len(nii_files)}] Processing: {os.path.basename(nii_file)}")
            try:
                scores, file_name = evaluate_single_file(
                    nii_file, evaluator, args.labels, device,
                    num_classes=num_classes, max_size=args.max_size, verbose=True,use_downsample=args.use_downsample
                )
                results_list.append((file_name, scores))
                print(f"  ✓ Success - Overall score: {scores['overall']:.4e}")
            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
                traceback.print_exc()
                continue

        # Save results to xlsx
        if len(results_list) > 0:
            output_filename = f"loc_score_{args.distance_metric}.xlsx"
            output_path = os.path.join(args.output_dir, output_filename)

            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)

            save_batch_results_to_xlsx(results_list, output_path, args.labels)

            print("\n" + "="*80)
            print("Batch Processing Complete")
            print("="*80)
        else:
            print("\nNo files were successfully processed.")

    else:
        # Single file mode
        print("\n" + "="*80)
        print("Single File Mode")
        print("="*80)

        scores, file_name = evaluate_single_file(
            args.pred_path, evaluator, args.labels, device,
            num_classes=num_classes, max_size=args.max_size, verbose=True
        )

        # ========================================
        # Interpretation
        # ========================================
        print("\n" + "="*80)
        print("Interpretation")
        print("="*80)

        overall_score = scores['overall']

        if overall_score >= 0.8:
            quality = "Excellent"
            color = "✓"
        elif overall_score >= 0.6:
            quality = "Good"
            color = "✓"
        elif overall_score >= 0.4:
            quality = "Fair"
            color = "⚠"
        elif overall_score >= 0.2:
            quality = "Poor"
            color = "⚠"
        else:
            quality = "Very Poor"
            color = "✗"

        print(f"\n{color} Segmentation Quality: {quality} (Score: {overall_score:.4e})")

        # Identify problematic labels
        print("\n--- Label-Specific Analysis ---")
        poor_labels = []
        good_labels = []

        for label_name, label_scores in scores['per_label'].items():
            shape_score = label_scores['shape_score']
            if shape_score < 0.5:
                poor_labels.append((label_name, shape_score))
            elif shape_score >= 0.7:
                good_labels.append((label_name, shape_score))

        if good_labels:
            print("\nWell-segmented structures:")
            for label_name, score in sorted(good_labels, key=lambda x: x[1], reverse=True):
                print(f"  ✓ {label_name:15s} (score: {score:.4e})")

        if poor_labels:
            print("\nPotentially problematic structures:")
            for label_name, score in sorted(poor_labels, key=lambda x: x[1]):
                print(f"  ⚠ {label_name:15s} (score: {score:.4e}) - may need review")

        # Display per-region location scores if available
        per_region_scores = scores['location'].get('per_region_scores', {})
        if per_region_scores:
            print("\n--- Per-Region Location Analysis ---")
            sorted_regions = sorted(per_region_scores.items(), key=lambda x: x[1], reverse=True)
            for region_name, region_score in sorted_regions:
                if region_score < 0.5:
                    status = "⚠"
                else:
                    status = "✓"
                print(f"  {status} {region_name:15s}: {region_score:.4e}")

        if scores['location']['score'] < 0.5:
            print("\n⚠ Overall location score is low - spatial relationships may be anatomically implausible")

        print("\n" + "="*80)
        print("Evaluation Complete")
        print("="*80)


if __name__ == '__main__':
    main()
