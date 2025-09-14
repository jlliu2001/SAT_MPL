"""
MAPSeg-style Med_SAM_Dataset for patch-based training
Integrates MAPSeg preprocessing while preserving all SAT dataset functionality
"""

import os
import random
import math

from einops import rearrange, repeat, reduce
import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import traceback
from tqdm import tqdm
from monai.transforms import (
    Compose,
    RandShiftIntensityd,
    RandRotate90d,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandScaleIntensityd,
    RandAdjustContrastd
)
import time
import torchio as tio

from train.dist import is_master
from .train_dataset import Med_SAM_Dataset, contains

# MAPSeg preprocessing utilities (from MAPSeg_MAE/data/data_utils.py)
def norm_img_mapseg(img, percentile=100):
    """MAPSeg style percentile normalization to [0,1]"""
    if np.isnan(img).any() or np.isinf(img).any():
        return None

    min_val = np.min(img)
    max_val = np.percentile(img, percentile)
    denom = max_val - min_val
    epsilon = 1e-8

    if denom < epsilon:
        return None

    img = (img - min_val) / denom
    return np.clip(img, 0, 1)

def random_flip_mapseg(img):
    """Random flip along random axes (50% chance each)"""
    # flip at 50% chance for each axis
    if np.random.random_sample() <= 0.5:
        img = np.flip(img, axis=0).copy()
    if np.random.random_sample() <= 0.5:
        img = np.flip(img, axis=1).copy()
    if np.random.random_sample() <= 0.5:
        img = np.flip(img, axis=2).copy()
    
    return img

def get_bounds_mapseg(img):
    """Get bounds of non-zero region (from MAPSeg_MAE/model/utils/util.py)"""
    if isinstance(img, torch.Tensor):
        img = np.squeeze(img.numpy())
    else:
        img = np.squeeze(img)
    
    nz_idx = np.nonzero(img > 1e-6)  # Use small threshold
    if len(nz_idx[0]) == 0:  # All zeros
        return [0, img.shape[0], 0, img.shape[1], 0, img.shape[2]]
    
    idx = []
    for i in nz_idx:
        idx.append(i.min())
        idx.append(i.max() + 1)  # +1 for inclusive indexing
    
    return idx

class Med_SAM_Dataset_MPLSeg(Med_SAM_Dataset):
    def __init__(self, 
                 jsonl_file, 
                 dataset_config,
                 crop_size=[256,128,256],  # Keep SAT's crop size for compatibility
                 max_queries=16, 
                 allow_repeat=True,
                 # MAPSeg specific parameters
                 patch_size=[96, 96, 96],  # MAPSeg patch size for local_patch and global_img
                 norm_perc=100,
                 remove_bg=True,
                 aug_prob=0.5):
        """
        MAPSeg-style Med_SAM_Dataset
        
        Args:
            All original Med_SAM_Dataset args, plus:
            patch_size: Size for both local_patch and global_img (MAPSeg default: 96^3)
            norm_perc: Percentile for MAPSeg normalization
            remove_bg: Whether to prefer foreground region for patch sampling
            aug_prob: Probability for MAPSeg augmentations
        """
        # Initialize parent class with original parameters
        super().__init__(jsonl_file, dataset_config, crop_size, max_queries, allow_repeat)
        
        # MAPSeg specific parameters
        self.patch_size = patch_size
        self.norm_perc = norm_perc
        self.remove_bg = remove_bg 
        self.aug_prob = aug_prob
        
        if is_master():
            print(f"** MAPSeg Dataset ** : Patch size {self.patch_size}")
            print(f"** MAPSeg Dataset ** : Normalization percentile {norm_perc}")
            print(f"** MAPSeg Dataset ** : Remove background: {remove_bg}")
            print(f"** MAPSeg Dataset ** : Aug probability: {aug_prob}")
    
    def _apply_mapseg_image_preprocessing(self, image_tensor):
        """
        Apply MAPSeg style preprocessing to image (like datasets.py mae_dataset.__getitem__)
        
        Args:
            image_tensor: [C, H, W, D] tensor
            
        Returns:
            processed_image: [C, H, W, D] MAPSeg-preprocessed tensor
        """
        # Convert to numpy (remove channel dim for processing)
        image_np = image_tensor.squeeze(0).numpy()  # [H, W, D]
        
        # 1. Negative value clipping (like MAPSeg datasets.py line 123)
        image_np[image_np < 0] = 0
        
        # 2. Random flip with probability (like MAPSeg datasets.py line 124) flip closed, mask don't do
        # if random.random() < self.aug_prob:
        #     image_np = random_flip_mapseg(image_np)
        
        # 3. MAPSeg normalization with random percentile (like datasets.py line 128-134)
        if random.random() <= self.aug_prob:
            perc_diff = 100 - self.norm_perc
            percentile = random.uniform(max(0.5, self.norm_perc - perc_diff), 100)
        else:
            percentile = self.norm_perc
        
        normalized_img = norm_img_mapseg(image_np, percentile)
        if normalized_img is None:
            # Fallback if normalization fails
            normalized_img = image_np
        
        # Convert back to tensor with channel dimension
        processed_tensor = torch.from_numpy(normalized_img).unsqueeze(0).float()
        
        return processed_tensor
    
    def _pad_to_patch_size(self, tensor):
        """Pad tensor to patch_size using 1e-4 (like MAPSeg to avoid patch mismatch)"""
        if tensor.dim() == 4:  # [C, H, W, D]
            _, h, w, d = tensor.shape
        else:  # [H, W, D]
            h, w, d = tensor.shape
            tensor = tensor.unsqueeze(0)
            
        target_h, target_w, target_d = self.patch_size
        
        # Only pad if necessary
        if h >= target_h and w >= target_w and d >= target_d:
            return tensor
            
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        pad_d = max(0, target_d - d)
        
        # Calculate padding for each side
        pad_h_before = pad_h // 2
        pad_h_after = pad_h - pad_h_before
        pad_w_before = pad_w // 2
        pad_w_after = pad_w - pad_w_before
        pad_d_before = pad_d // 2
        pad_d_after = pad_d - pad_d_before
        
        # PyTorch padding format: (left, right, top, bottom, front, back)
        padding = (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after)
        
        # Use 1e-4 like MAPSeg (datasets.py line 147)
        padded_tensor = F.pad(tensor, padding, mode='constant', value=1e-4)
        
        return padded_tensor
    
    def _generate_mapseg_patches(self, mapseg_image, mask_tensor=None):
        """
        Generate MAPSeg-style local_patch, global_img, and coordinates
        (Following MAPSeg datasets.py mae_dataset.__getitem__ logic)
        
        Args:
            mapseg_image: [C, H, W, D] preprocessed image
            mask_tensor: [N, H, W, D] mask (optional, for better sampling)
            
        Returns:
            local_patch: [C, patch_h, patch_w, patch_d] sampled patch
            global_img: [C, patch_h, patch_w, patch_d] downsampled full image
            coordinates: [6] coordinates in downsampled space
        """
        x, y, z = self.patch_size
        _, img_h, img_w, img_d = mapseg_image.shape
        
        # Determine sampling bounds (like MAPSeg datasets.py)
        if self.remove_bg:
            if mask_tensor is not None:
                # Use mask to find foreground
                mask_combined = torch.sum(mask_tensor, dim=0)
                bounds = get_bounds_mapseg(mask_combined)
            else:
                # Use image intensity 
                bounds = get_bounds_mapseg(mapseg_image)
        else:
            # Full image bounds
            bounds = [0, img_h, 0, img_w, 0, img_d]
        
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        # Sample patch location (like datasets.py line 162-184)
        if x_max - x > x_min:
            x_idx = random.randint(x_min, x_max - x)
        else:
            if x_max - x >= 0:
                x_idx = x_max - x
            else:
                x_idx = int((img_h - x) / 2)
                
        if y_max - y > y_min:
            y_idx = random.randint(y_min, y_max - y)
        else:
            if y_max - y >= 0:
                y_idx = y_max - y
            else:
                y_idx = int((img_w - y) / 2)
                
        if z_max - z > z_min:
            z_idx = random.randint(z_min, z_max - z)
        else:
            if z_max - z >= 0:
                z_idx = z_max - z
            else:
                z_idx = int((img_d - z) / 2)
        
        # Extract local patch (like datasets.py line 209)
        local_patch = mapseg_image[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z]
        mask_patch=mask_tensor[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z]
        
        # Create location mask (like datasets.py line 200-201)
        location = torch.zeros_like(mapseg_image[:1])  # [1, H, W, D]
        location[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z] = 1
        
        # Create global image (like datasets.py line 203-207 and test.py line 53-58)
        try:
            # Crop to bounds first
            cropped_img = mapseg_image[:, x_min:x_max, y_min:y_max, z_min:z_max]
            cropped_loc = location[:, x_min:x_max, y_min:y_max, z_min:z_max]
            
            # Use TorchIO for resizing (like test.py)
            img_subject = tio.Subject(
                one_image=tio.ScalarImage(tensor=cropped_img),
                a_segmentation=tio.LabelMap(tensor=cropped_loc)
            )

            
            resize_transform = tio.transforms.Resize(target_shape=(x, y, z))
            resized_subject = resize_transform(img_subject)
            
            global_img = resized_subject['one_image'].data
            resized_location = resized_subject['a_segmentation'].data
            
            # Calculate coordinates (like test.py line 61-68)
            loc_bounds = get_bounds_mapseg(resized_location)
            coordinates = np.array([
                np.floor(loc_bounds[0] / 4),
                np.ceil(loc_bounds[1] / 4),
                np.floor(loc_bounds[2] / 4),
                np.ceil(loc_bounds[3] / 4),
                np.floor(loc_bounds[4] / 4),
                np.ceil(loc_bounds[5] / 4)
            ]).astype(int)
            coordinates = torch.unsqueeze(torch.from_numpy(coordinates), 0)
            
        except Exception as e:
            # Fallback to simple interpolation
            if is_master():
                print(f"TorchIO resize failed: {e}, using simple interpolation")
            global_img = F.interpolate(mapseg_image.unsqueeze(0), size=(x, y, z), 
                                     mode='trilinear', align_corners=False).squeeze(0)
            # Default coordinates
            coordinates = np.array([x//4, 3*x//4, y//4, 3*y//4, z//4, 3*z//4])
        
        return local_patch, global_img, coordinates, mask_patch
    
    def __getitem__(self, idx):
        """
        Enhanced __getitem__ with MAPSeg patch processing
        Preserves all original Med_SAM_Dataset logic except image processing
        
        Returns:
            dict with original fields plus:
            - image: global_img [C, patch_h, patch_w, patch_d] 
            - local_patch: local patch [C, patch_h, patch_w, patch_d]
            - coordinates: patch coordinates [6]
        """
        while True:
            try: 
                # === IDENTICAL TO ORIGINAL Med_SAM_Dataset ===
                sample = random.choices(self.data_3d, weights=self.sample_weight_3d)[0]
                
                # Load image using original method
                image = self._load_image(sample)  # [C, H, W, D]
                
                # Process modality (original logic)
                modality = sample['modality']
                modality = self._merge_modality(modality.lower())
                
                # Pad image (original logic)
                image, _ = self._pad_if_necessary(image, mask=None)
                
                # Crop image (original logic)
                roi_crop_prob = self.dataset_config[sample['dataset']]['foreground_crop_prob']
                label_based_crop_prob = self.dataset_config[sample['dataset']]['label_based_crop_prob']
                uncenter_prob = self.dataset_config[sample['dataset']]['uncenter_prob']
                image, y1x1z1_y2x2z2 = self._crop(image, sample, roi_crop_prob, label_based_crop_prob, uncenter_prob)

                # Label processing (original logic)
                is_pos_in_crop = self._find_pos_labels_in_crop(y1x1z1_y2x2z2, sample['renorm_y1x1z1_y2x2z2'])
                
                pos_label_first_prob = self.dataset_config[sample['dataset']]['pos_label_first_prob']
                neg_label_ratio_threshold = self.dataset_config[sample['dataset']]['neg_label_ratio_threshold']
                all_label_index_ls = [i for i in range(len(is_pos_in_crop))]
                pos_first = random.random() < pos_label_first_prob
                if pos_first:
                    chosen_label_index_ls, is_pos_ls = self._select_pos_labels(all_label_index_ls, is_pos_in_crop, neg_label_ratio_threshold)
                else:
                    chosen_label_index_ls = random.sample(all_label_index_ls, min(self.max_queries, len(all_label_index_ls)))
                
                chosen_label = [sample['label'][i] for i in chosen_label_index_ls]
                chosen_y1x1z1_y2x2z2 = [sample['renorm_y1x1z1_y2x2z2'][i] for i in chosen_label_index_ls]
                
                # Load masks (original logic)
                mask = self._load_mask(sample, chosen_label, chosen_y1x1z1_y2x2z2)
                
                # Process masks (original logic)
                _, mask = self._pad_if_necessary(image=None, mask=mask)
                y1, x1, z1, y2, x2, z2 = y1x1z1_y2x2z2
                mask = mask[:, y1:y2, x1:x2, z1:z2]
                print('mask in precess.shape:',mask.shape)
                print('imagein precess.shape:',image.shape)
                
                # Filter false positives (original logic)
                # if pos_first:
                #     filtered_indexes = []
                #     for i, is_pos in enumerate(is_pos_ls):
                #         if (not is_pos) or (torch.sum(mask[i]) > 0):  # Fixed logic: keep if neg or has positive voxels
                #             filtered_indexes.append(i)
                #     mask = mask[filtered_indexes, :, :, :]  
                #     chosen_label = [chosen_label[i] for i in filtered_indexes]    
                #     if len(filtered_indexes) < len(is_pos_ls):
                #         is_pos_ls = [is_pos_ls[i] for i in filtered_indexes]


                
                # === MAPSeg IMAGE PROCESSING (REPLACES ORIGINAL AUGMENTATION) ===
                # Apply MAPSeg preprocessing to the SAT-processed image
                mapseg_processed_image = self._apply_mapseg_image_preprocessing(image.clone())
                
                # Pad to patch size if needed
                mapseg_processed_image = self._pad_to_patch_size(mapseg_processed_image)
                mask_processed_image = self._pad_to_patch_size(mask)
                
                # Generate MAPSeg patches
                local_patch, global_img, coordinates, mask_patch = self._generate_mapseg_patches(mapseg_processed_image, mask_processed_image)
                
                # Apply original MONAI augmentation to mask only (keep image MAPSeg-style)
                if sample['dataset'] in self.augmentator:
                    # Only augment mask, keep MAPSeg image processing
                    data_dict = {'image': global_img, 'label': mask_patch}
                    aug_data_dict = self.augmentator[sample['dataset']](data_dict)
                    _, mask_patch = aug_data_dict['image'], aug_data_dict['label']  # Only use augmented mask
                
                break
                
            except SystemExit:
                exit()
            except:
                # Record bugs in loading data
                traceback_info = traceback.format_exc()
                print(f'*** {sample["dataset"]} *** {sample["image"]} ***\n')
                print(traceback_info)

        # Return enhanced data with MAPSeg outputs
        return {
            # === MAPSeg Required Outputs ===
            'image': global_img,             # [C, 96, 96, 96] - global downsampled image
            'local_patch': local_patch,      # [C, 96, 96, 96] - sampled local patch  
            'coordinates': coordinates,      # [6] - patch coordinates in global space
            # === Original Med_SAM_Dataset Outputs (Preserved) ===
            'mask': mask_patch,                    # [N, H, W, D] - segmentation masks
            'text': chosen_label,            # List[str] - text labels
            'modality': modality,            # str - image modality
            'image_path': sample['renorm_image'],           # str - image file path
            'mask_path': sample['renorm_segmentation_dir'], # str - mask directory path  
            'dataset': sample['dataset'],                   # str - dataset name
            'y1x1z1_y2x2z2': y1x1z1_y2x2z2,               # List[int] - crop coordinates
            
            # === Additional Debug Info ===
            'patch_size': self.patch_size,   # List[int] - patch size used
        }