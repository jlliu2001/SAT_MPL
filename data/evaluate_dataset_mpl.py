"""
MPL-specific evaluation dataset classes
Handles patch-based preprocessing for MPL model inference, following the infer_single_scan approach
"""
import os
import random
import json
import traceback
import math

from einops import rearrange, repeat, reduce
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
import nibabel as nib
import torchio as tio

from train.dist import is_master

def contains(text, key):
    if isinstance(key, str):
        return key in text
    elif isinstance(key, list):
        for k in key:
            if k in text:
                return True
        return False      

def norm_img_mpl(img, percentile=100):
    """MPL-style normalization (from MAPSeg_MAE/model/utils/util.py)"""
    img = (img - np.min(img)) / (np.percentile(img, percentile) - np.min(img))
    return np.clip(img, 0, 1)

def get_bounds_mpl(img):
    """Get bounds of non-zero region (from MAPSeg_MAE/model/utils/util.py)"""
    try:
        img = np.squeeze(img.numpy())
    except:
        img = np.squeeze(img)
    nz_idx = np.nonzero(img)
    # if len(nz_idx[0]) == 0:  # All zeros
    #     return [0, img.shape[0], 0, img.shape[1], 0, img.shape[2]]
    idx = []
    for i in nz_idx:
        idx.append(i.min())
        idx.append(i.max())
    return idx

def _gen_indices(i1, i2, k, s):
    assert i2 >= k, 'sample size has to be bigger than the patch size'
    for j in range(i1, i2 - k + 1, s):
        yield j
        if j + k < i2:
            yield i2 - k

def patch_slicer_mpl(scan, patch_size, stride, remove_bg=True):
    """
    MPL-style patch slicer (simplified from MAPSeg_MAE/model/utils/util.py)
    Returns patches and their indices for sliding window inference
    
    Args:
        scan: numpy array [H, W, D]
        patch_size: tuple (x, y, z) 
        stride: tuple (s1, s2, s3)
        remove_bg: whether to focus on foreground
    
    Returns:
        scan_patches: list of patches
        patch_idx: list of (x_start, x_end, y_start, y_end, z_start, z_end)
    """
    x, y, z = scan.shape
    scan_patches = []
    patch_idx = []
    
    if remove_bg:
        bound = get_bounds_mpl(torch.from_numpy(scan))
        x1, x2, y1, y2, z1, z2 = bound
    else:
        x1, x2, y1, y2, z1, z2 = 0, x, 0, y, 0, z
    
    p1, p2, p3 = patch_size
    s1, s2, s3 = stride
    
    # Adjust bounds if too small
    if x2 - x1 < p1:
        if x2 - p1 > 0:
            x1 = x2 - p1
        else:
            x1 = 0
            x2=p1
    if y2 - y1 < p2:
        if y2 - p2 > 0:
            y1 = y2 - p2  
        else:
            y1 = 0
            y2=p2
    if z2 - z1 < p3:
        if z2 - p3 > 0:
            z1 = z2 - p3
        else:
            z1 = 0
            z2=p3
    
    # Generate patches
    # for xi in range(x1, min(x2 - p1 + 1, x1 + s1 * ((x2 - x1 - p1) // s1) + 1), s1):
    #     if xi + p1 > x:
    #         xi = x - p1
    #     for yi in range(y1, min(y2 - p2 + 1, y1 + s2 * ((y2 - y1 - p2) // s2) + 1), s2):
    #         if yi + p2 > y:
    #             yi = y - p2
    #         for zi in range(z1, min(z2 - p3 + 1, z1 + s3 * ((z2 - z1 - p3) // s3) + 1), s3):
    #             if zi + p3 > z:
    #                 zi = z - p3
                    
    #             patch = scan[xi:xi+p1, yi:yi+p2, zi:zi+p3]
    #             scan_patches.append(patch)
    #             patch_idx.append((xi, xi+p1, yi, yi+p2, zi, zi+p3))
    
    # return scan_patches, patch_idx

    x_stpes = _gen_indices(x1, x2, p1, s1)
    for x_idx in x_stpes:
        y_steps = _gen_indices(y1, y2, p2, s2)
        for y_idx in y_steps:
            z_steps = _gen_indices(z1, z2, p3, s3)
            for z_idx in z_steps:
                patch = scan[x_idx:x_idx + p1,
                                y_idx:y_idx + p2, z_idx:z_idx + p3]
                # tmp_label = mask[x_idx:x_idx + p1,
                #                  y_idx:y_idx + p2, z_idx:z_idx + p3]
                scan_patches.append(patch)
                # mask_patches.append(tmp_label)
                # if test:
                    # file_path.append(ori_path)
                patch_idx.append(
                    [x_idx, x_idx + p1, y_idx, y_idx + p2, z_idx, z_idx + p3])
    # if not test:
    #     return scan_patches, mask_patches
    # else:
    return scan_patches,  patch_idx

class Evaluate_Dataset_OnlineCrop_MPL(Dataset):
    def __init__(self, jsonl_file, max_queries=256, batch_size=1, 
                 patch_size=[96, 96, 96], norm_perc=100, evaluated_samples=set()):
        """
        MPL-specific online crop dataset for inference
        
        Args:
            jsonl_file: path to JSONL file
            max_queries: maximum number of queries per batch
            batch_size: batch size for patch processing (keep small for MPL)
            patch_size: MPL patch size [96, 96, 96]  
            norm_perc: normalization percentile
            evaluated_samples: samples to skip (for resuming)
        """
        # Load data info
        self.jsonl_file = jsonl_file
        with open(self.jsonl_file, 'r') as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
            
        self.lines = []
        for sample in lines:
            # Skip already evaluated samples
            sample_id = sample['renorm_image'].split('/')[-1][:-4]  # abcd/x.npy --> x
            dataset_name = sample['dataset']
            if f'{dataset_name}_{sample_id}' not in evaluated_samples:
                self.lines.append(sample)
        
        self.max_queries = max_queries
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.norm_perc = norm_perc
        
        # MPL uses overlapping patches for better reconstruction
        stride_reduction = 16  # Like in infer_single_scan  
        self.stride = [patch_size[0] - stride_reduction, 
                      patch_size[1] - stride_reduction, 
                      patch_size[2] - stride_reduction]
        
        if is_master():          
            print(f'** MPL Online Crop DATASET ** : Skip {len(lines)-len(self.lines)} samples, {len(self.lines)} to be evaluated')
            print(f'** MPL Online Crop DATASET ** : Maximum {self.max_queries} queries, patch size {self.patch_size}')
            print(f'** MPL Online Crop DATASET ** : Patch stride {self.stride}, normalization percentile {norm_perc}')
        
    def __len__(self):
        return len(self.lines)
    
    def _split_labels(self, label_list):
        """Split the labels into sub-lists"""
        if len(label_list) < self.max_queries:
            return [label_list], [[0, len(label_list)]]
        else:
            split_idx = []
            split_label = []
            query_num = len(label_list)
            n_crop = (query_num // self.max_queries + 1) if (query_num % self.max_queries != 0) else (query_num // self.max_queries)
            for n in range(n_crop):
                n_s = n*self.max_queries
                n_f = min((n+1)*self.max_queries, query_num)
                split_label.append(label_list[n_s:n_f])
                split_idx.append([n_s, n_f])
            return split_label, split_idx
    
    def _merge_modality(self, mod):
        """Merge modalities following SAT convention"""
        if contains(mod, ['t1', 't2', 'mri', 'mr', 'flair', 'dwi']):
            return 'mri'
        if contains(mod, 'ct'):
            return 'ct'
        if contains(mod, 'pet'):
            return 'pet'
        else:
            return mod
    
    def _preprocess_mpl_image(self, image_np):
        """
        MPL-specific image preprocessing following infer_single_scan
        
        Args:
            image_np: numpy array [H, W, D]
            
        Returns:
            processed_img: preprocessed numpy array
            pad_info: padding information for restoration
        """
        # 1. Normalize image (like infer_single_scan line 18-19)
        normalized_img = norm_img_mpl(image_np, self.norm_perc)
        
        # 2. Pad if necessary (like infer_single_scan line 20-27) 
        x, y, z = self.patch_size
        pad_flag = False
        pad_info = None
        
        if min(normalized_img.shape) < min(x, y, z):
            x_ori_size, y_ori_size, z_ori_size = normalized_img.shape
            pad_flag = True
            x_diff = x - x_ori_size
            y_diff = y - y_ori_size
            z_diff = z - z_ori_size
            
            # Pad with 1e-4 (like MAPSeg, can't pad with 0s)
            normalized_img = np.pad(normalized_img, 
                                  ((max(0, int(x_diff/2)), max(0, x_diff-int(x_diff/2))), 
                                   (max(0, int(y_diff/2)), max(0, y_diff-int(y_diff/2))), 
                                   (max(0, int(z_diff/2)), max(0, z_diff-int(z_diff/2)))), 
                                  constant_values=1e-4)
            
            pad_info = {
                'pad_flag': pad_flag,
                'original_shape': (x_ori_size, y_ori_size, z_ori_size),
                'x_diff': x_diff, 'y_diff': y_diff, 'z_diff': z_diff
            }
        else:
            pad_info = {'pad_flag': False}
            
        return normalized_img, pad_info
    
    def _generate_mpl_patches_and_global_imgs(self, processed_img):
        """
        Generate MPL patches and global images following infer_single_scan approach
        
        Args:
            processed_img: preprocessed image [H, W, D]
            
        Returns:
            patch_data: list of dicts containing local_patch, global_img, coordinates
            patch_indices: list of patch coordinate tuples for reconstruction
        """
        x, y, z = self.patch_size
        
        # 1. Generate sliding window patches (like infer_single_scan line 32-35)
        scan_patches, tmp_idx = patch_slicer_mpl(
            processed_img, 
            self.patch_size,
            self.stride,
            remove_bg=True
        )
        
        # 2. Get image bounds for global image generation 
        bound = get_bounds_mpl(torch.from_numpy(processed_img))
        global_scan = torch.unsqueeze(torch.from_numpy(processed_img).to(dtype=torch.float), dim=0)
        
        patch_data = []
        patch_indices = []
        print('len(scan_patches):',len(scan_patches))
        
        # 3. Process each patch (like infer_single_scan line 43-72)
        for idx, patch in enumerate(scan_patches):
            # Local patch tensor
            local_patch_tensor = torch.from_numpy(patch).to(dtype=torch.float)
            local_patch_tensor = local_patch_tensor.reshape((1, 1,) + local_patch_tensor.shape)
            
            patch_idx = tmp_idx[idx]
            patch_indices.append(patch_idx)
            
            # Create location mask for this patch
            location = torch.zeros_like(torch.from_numpy(processed_img)).float()
            location = torch.unsqueeze(location, 0)
            location[:, patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3], patch_idx[4]:patch_idx[5]] = 1
            
            # Generate global image with TorchIO (like infer_single_scan line 53-58)
            try:
                sbj = tio.Subject(
                    one_image=tio.ScalarImage(
                        tensor=global_scan[:, bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]]
                    ),
                    a_segmentation=tio.LabelMap(
                        tensor=location[:, bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]]
                    )
                )
                transforms = tio.transforms.Resize(target_shape=(x, y, z))
                sbj = transforms(sbj)
                down_scan = sbj['one_image'].data
                loc = sbj['a_segmentation'].data
                
                # Calculate coordinates (like infer_single_scan line 61-70)
                tmp_coor = get_bounds_mpl(loc)
                coordinates_A = np.array([
                    np.floor(tmp_coor[0] / 4), np.ceil(tmp_coor[1] / 4),
                    np.floor(tmp_coor[2] / 4), np.ceil(tmp_coor[3] / 4),
                    np.floor(tmp_coor[4] / 4), np.ceil(tmp_coor[5] / 4)
                ]).astype(int)
                coordinates_A = torch.from_numpy(coordinates_A).unsqueeze(0)
                
            except Exception as e:
                if is_master():
                    print(f"TorchIO processing failed: {e}, using fallback")
                # Fallback: simple resize
                down_scan = F.interpolate(
                    global_scan.unsqueeze(0), size=(x, y, z), 
                    mode='trilinear', align_corners=False
                ).squeeze(0)
                coordinates_A = torch.tensor([[x//4, 3*x//4, y//4, 3*y//4, z//4, 3*z//4]])
            
            # Reshape for model input
            global_img_tensor = down_scan.reshape([1, 1, x, y, z])
            
            patch_data.append({
                'local_patch': local_patch_tensor,  # [1, 1, patch_size]
                'global_img': global_img_tensor,    # [1, 1, patch_size] 
                'coordinates': coordinates_A        # [1, 6]
            })
            
        return patch_data, patch_indices
        
    def __getitem__(self, idx):
        """
        Get item with MPL-specific preprocessing
        
        Returns:
            dict with MPL-specific fields for sliding window inference
        """
        datum = self.lines[idx]
        sample_id = datum['renorm_image'].split('/')[-1][:-4]  # abcd/x.npy --> x
        raw_mask_path=datum['mask']
        patient_id=datum['patient_id'].split('.')[0]
        
        
        # Load original image
        img_np = np.load(datum['renorm_image'])  # [C, H, W, D]
        print('img_np.shape:',img_np.shape)
        if img_np.ndim == 4:
            img_np = img_np.squeeze(0)  # Remove channel dimension -> [H, W, D]
            
        # MPL preprocessing
        processed_img, pad_info = self._preprocess_mpl_image(img_np)
        print('processed_img.shape:',processed_img.shape)
        
        # Generate MPL patches and global images
        patch_data, patch_indices = self._generate_mpl_patches_and_global_imgs(processed_img)
        print('patch_data:',len(patch_data))
        
        # Batch patches for processing
        batch_num = len(patch_data) // self.batch_size if len(patch_data) % self.batch_size == 0 else len(patch_data) // self.batch_size + 1
        batched_patch_data = []
        
        for i in range(batch_num):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(patch_data))
            batch_patches = patch_data[start_idx:end_idx]
            # batched_patch_data.append(batch_patches)
            batched_patch_data.append(patch_data)
        
        
        # Split labels into batches
        labels = datum['label']
        split_labels, split_n1n2 = self._split_labels(labels)
        modality = datum['modality']
        modality = self._merge_modality(modality.lower())
        for i in range(len(split_labels)):
            split_labels[i] = [label.lower() for label in split_labels[i]]
            
        # Load ground truth segmentations 
        c, h, w, d = datum['chwd']
        mask_paths = [f"{datum['renorm_segmentation_dir']}/{label}.npy" for label in labels]
        y1x1z1_y2x2z2_ls = datum['renorm_y1x1z1_y2x2z2']
        
        mc_mask = []
        for mask_path, y1x1z1_y2x2z2 in zip(mask_paths, y1x1z1_y2x2z2_ls):
            mask = torch.zeros((h, w, d))
            # Load mask if not empty
            if y1x1z1_y2x2z2 != False:
                y1, x1, z1, y2, x2, z2 = y1x1z1_y2x2z2
                mask[y1:y2, x1:x2, z1:z2] = torch.tensor(np.load(mask_path))
            mc_mask.append(mask.float())
        mc_mask = torch.stack(mc_mask, dim=0)   # [N, H, W, D]
        print('batched_patch_data:',len(batched_patch_data))

        return {
            'dataset_name': datum['dataset'],
            'sample_id': sample_id,
            'batched_patch_data': batched_patch_data,  # MPL-specific patch batches
            'patch_indices': patch_indices,            # For reconstruction
            'pad_info': pad_info,                      # For result restoration
            'original_shape': img_np.shape,            # Original image shape
            'processed_shape': processed_img.shape,    # Processed image shape  
            'split_labels': split_labels,
            'modality': modality,
            'split_n1n2': split_n1n2,
            'gt_segmentation': mc_mask,
            'labels': labels,
            'image_path': datum['renorm_image'],
            'raw_mask_path': raw_mask_path,
            'patient_id':patient_id
        }

class Evaluate_Dataset_MPL(Dataset):
    def __init__(self, jsonl_file, max_queries=256, batch_size=1, 
                 patch_size=[96, 96, 96], norm_perc=100, evaluated_samples=set()):
        """
        MPL-specific dataset for pre-computed patches (if available)
        Similar to original Evaluate_Dataset but with MPL preprocessing
        """
        # Load data info
        self.jsonl_file = jsonl_file
        with open(self.jsonl_file, 'r') as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        
        self.lines = []
        for sample in lines:
            # Skip already evaluated samples
            sample_id = sample['renorm_image'].split('/')[-1][:-4]  # abcd/x.npy --> x
            dataset_name = sample['dataset']
            if f'{dataset_name}_{sample_id}' not in evaluated_samples:
                self.lines.append(sample)
        
        if is_master():          
            print(f'** MPL DATASET ** : Skip {len(lines)-len(self.lines)} samples, {len(self.lines)} to be evaluated')
        
        self.max_queries = max_queries
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.norm_perc = norm_perc
        
    def __len__(self):
        return len(self.lines)
    
    def _split_labels(self, label_list):
        """Split the labels into sub-lists"""
        if len(label_list) < self.max_queries:
            return [label_list], [[0, len(label_list)]]
        else:
            split_idx = []
            split_label = []
            query_num = len(label_list)
            n_crop = (query_num // self.max_queries + 1) if (query_num % self.max_queries != 0) else (query_num // self.max_queries)
            for n in range(n_crop):
                n_s = n*self.max_queries
                n_f = min((n+1)*self.max_queries, query_num)
                split_label.append(label_list[n_s:n_f])
                split_idx.append([n_s, n_f])
            return split_label, split_idx
    
    def _merge_modality(self, mod):
        """Merge modalities following SAT convention"""
        if contains(mod, ['t1', 't2', 'mri', 'mr', 'flair', 'dwi']):
            return 'mri'
        if contains(mod, 'ct'):
            return 'ct'
        if contains(mod, 'pet'):
            return 'pet'
        else:
            return mod
        
    def __getitem__(self, idx):
        """
        For pre-computed patches, fallback to online processing
        """
        # If patch_path exists, could load pre-computed patches
        # But for MPL we need the special preprocessing, so use online approach
        datum = self.lines[idx]
        
        # Create temporary OnlineCrop instance for this sample
        temp_dataset = Evaluate_Dataset_OnlineCrop_MPL(
            jsonl_file=None,
            max_queries=self.max_queries,
            batch_size=self.batch_size,
            patch_size=self.patch_size,
            norm_perc=self.norm_perc
        )
        temp_dataset.lines = [datum]
        
        return temp_dataset.__getitem__(0)

def collate_fn_mpl(data):
    """Collate function for MPL datasets"""
    return data[0]

# def collect_fn_mplseg(data):
#     """
#     Pad images and masks to the same depth and num of class
    
#     Args:
#         data : [{'text':..., 'image':..., 'mask':..., 'modality':..., 'image_path':..., 'mask_path':..., 'dataset':..., 'y1x1z1_y2x2z2':...}, ...]
#     """
    
#     image = []
#     mask = []
#     text = []
#     modality = []
#     image_path = []
#     mask_path = []
#     dataset = []
#     y1x1z1_y2x2z2 = []
#     local_patch = []
#     # coordinates = []

#     # pad to max depth in the batch
#     # pad to max num of class in the batch
#     max_class = 1
#     for sample in data:
#         class_num = sample['mask'].shape[0]
#         max_class = class_num if class_num > max_class else max_class
        
#     query_mask = torch.zeros((len(data), max_class)) # bn
#     for i, sample in enumerate(data):
#         # for MAE: DO NOT EXPAND THE CHANNEL
#         # if sample['image'].shape[0] == 1:
#         #     sample['image'] = repeat(sample['image'], 'c h w d -> (c r) h w d', r=3)
#         image.append(sample['image'])
#         local_patch.append(sample['local_patch'])
#         coordinates=sample['coordinates']
        
        
#         class_num = sample['mask'].shape[0]
#         pad = (0, 0, 0, 0, 0, 0, 0, max_class-class_num)
#         padded_mask = F.pad(sample['mask'], pad, 'constant', 0)   # nhwd
#         mask.append(padded_mask)
#         sample['text'] += ['none'] * (max_class-class_num)
#         query_mask[i, :class_num] = 1.0
        
#         text.append(sample['text'])   
#         modality.append(sample['modality'])
#         image_path.append(sample['image_path'])
#         mask_path.append(sample['mask_path'])
#         dataset.append(sample['dataset'])
#         y1x1z1_y2x2z2.append(sample['y1x1z1_y2x2z2'])
    
#     image = torch.stack(image, dim=0)
#     mask = torch.stack(mask, dim=0).float()
#     local_patch=torch.stack(local_patch,dim=0)
#     print(f"Sample {i}: ,image.shape={image.shape},mask.shape={sample['mask'].shape},text={sample['text']},class_num={class_num},coordinates={coordinates},local_patch.shape={local_patch.shape}")
#     return {'image':image, 'mask':mask, 'text':text, 'modality':modality, 'image_path':image_path, 'mask_path':mask_path, 'dataset':dataset, 'query_mask':query_mask, 'y1x1z1_y2x2z2':y1x1z1_y2x2z2, 'local_patch':local_patch, 'coordinates':coordinates}
