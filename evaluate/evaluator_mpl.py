"""
MPL-specific evaluator for handling patch-based inference
Adapted from original evaluator.py to work with MPL preprocessing and reconstruction
"""
import os
import time

import torch
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
from einops import rearrange, repeat, reduce
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
import shutil
import pickle
from scipy.ndimage import gaussian_filter
import torch.distributed as dist

from evaluate.metric import calculate_metric_percase
from evaluate.merge_after_evaluate import merge
from train.dist import is_master

def compute_gaussian(tile_size, sigma_scale: float = 1. / 8, value_scaling_factor: float = 10, dtype=np.float16):
    """Compute gaussian weights for patch aggregation"""
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.astype(dtype)
    
    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])
    
    return gaussian_importance_map

def reconstruct_prediction_mpl(patch_predictions, patch_indices, pad_info, processed_shape, cls_num, device='cpu'):
    """
    Reconstruct full volume prediction from MPL patch predictions
    Following infer_single_scan logic for aggregation
    
    Args:
        patch_predictions: list of prediction tensors from model
        patch_indices: list of patch coordinate tuples
        pad_info: padding information from dataset
        cls_num: number of classes
        device: computation device
        
    Returns:
        pred_vol: reconstructed prediction volume [H, W, D]
    """
    if pad_info['pad_flag']:
        # Use padded shape for reconstruction
        pred_shape = pad_info.get('padded_shape', None)
        if pred_shape is None:
            # Calculate padded shape from original + diffs
            orig_h, orig_w, orig_d = pad_info['original_shape']
            x_diff, y_diff, z_diff = pad_info['x_diff'], pad_info['y_diff'], pad_info['z_diff']
            pred_shape = (
                orig_h + x_diff,
                orig_w + y_diff, 
                orig_d + z_diff
            )
    else:
        # Use original shape
        # pred_shape = pad_info.get('processed_shape', patch_indices[0][:2] + patch_indices[0][2:4] + patch_indices[0][4:])
        # if len(pred_shape) != 3:
        #     # Fallback: estimate from patch indices
        #     max_coords = [0, 0, 0]
        #     for patch_idx in patch_indices:
        #         x1, x2, y1, y2, z1, z2 = patch_idx
        #         max_coords[0] = max(max_coords[0], x2)
        #         max_coords[1] = max(max_coords[1], y2) 
        #         max_coords[2] = max(max_coords[2], z2)
        #     pred_shape = tuple(max_coords)
        pred_shape=processed_shape
    
    # Initialize prediction and normalization arrays (like infer_single_scan line 29-30)
    pred = np.zeros((cls_num,) + pred_shape)
    tmp_norm = np.zeros((cls_num,) + pred_shape)
    
    # Aggregate predictions from all patches (like infer_single_scan line 78-80)
    for i, (patch_pred, patch_idx) in enumerate(zip(patch_predictions, patch_indices)):
        x1, x2, y1, y2, z1, z2 = patch_idx
        
        # Convert prediction to numpy if needed
        if isinstance(patch_pred, torch.Tensor):
            patch_pred = patch_pred.detach().cpu().numpy()
        print('patch_pred.shape:',patch_pred.shape)
        
        # Ensure correct shape [cls_num, H, W, D]
        if patch_pred.ndim == 5:  # [B, cls_num, H, W, D]
            patch_pred = np.squeeze(patch_pred, axis=0)
        if patch_pred.ndim == 4 and patch_pred.shape[0] != cls_num:
            # Assume first dim is batch, squeeze it
            patch_pred = np.squeeze(patch_pred, axis=0)
            
        # Create patch index tuple for prediction array
        patch_slice = (slice(0, cls_num), slice(x1, x2), slice(y1, y2), slice(z1, z2))
        
        try:
            pred[patch_slice] += patch_pred
            tmp_norm[patch_slice] += 1
        except Exception as e:
            if is_master():
                print(f"Error aggregating patch {i}: {e}")
                print(f"Patch pred shape: {patch_pred.shape}, Patch slice: {patch_slice}")
                print(f"Pred shape: {pred.shape}, Expected shape: {(cls_num,) + pred_shape}")
            continue
    
    # Normalize predictions (like infer_single_scan line 82-83)
    pred[tmp_norm > 0] = pred[tmp_norm > 0] / tmp_norm[tmp_norm > 0]
    
    # Apply softmax and get final prediction (like infer_single_scan line 84-86)
    # sf = torch.nn.Softmax(dim=0)
    # pred_vol = sf(torch.from_numpy(pred)).numpy()
    # pred_vol = np.argmax(pred_vol, axis=0)
    pred_ = np.where(pred>0.5, 1.0, 0.0)
    pred_vol=np.zeros(pred_shape)

    for i_ in range(pred_.shape[0]):
        pred_vol[pred_[i_]>0]=i_+1
    print('np.unique(pred_vol):',np.unique(pred_vol))
    

    
    # Unpad if necessary (like infer_single_scan line 87-92)
    if pad_info['pad_flag']:
        orig_h, orig_w, orig_d = pad_info['original_shape']
        x_diff, y_diff, z_diff = pad_info['x_diff'], pad_info['y_diff'], pad_info['z_diff']
        
        pred_vol = pred_vol[
            max(0, int(x_diff/2)): max(0, int(x_diff/2)) + orig_h,
            max(0, int(y_diff/2)): max(0, int(y_diff/2)) + orig_w,
            max(0, int(z_diff/2)): max(0, int(z_diff/2)) + orig_d
        ]
        
        assert pred_vol.shape == (orig_h, orig_w, orig_d), \
            f'pred_vol shape {pred_vol.shape} must match original {(orig_h, orig_w, orig_d)}'
    
    return pred_vol,pred_

def evaluate_mpl(model, 
                 text_encoder, 
                 device, 
                 testset, 
                 testloader, 
                 dice_score,
                 nsd_score,
                 csv_path, 
                 resume,
                 save_interval,
                 visualization):
    """
    MPL-specific evaluation function
    Handles patch-based inference and reconstruction
    """
    
    # if to store pred、gt、img (as nii.gz)
    if visualization:
        nib_dir = csv_path.replace('.csv', '')
        
    # collate in master process
    if is_master():
        shutil.copy(testset.jsonl_file if hasattr(testset, 'jsonl_file') else '/dev/null', 
                   csv_path.replace('.csv', '.jsonl'))
        
        # datasets --> labels --> metrics
        datasets_labels_metrics = {}
        # datasets --> samples --> labels --> metrics
        samples_labels_metrics = {}
        # datsets --> labels
        datasets_labels_sets = {}
        
    # accumulate scores of each sample in each process
    results_of_samples = []
    
    # load results from an interrupted eval (only in master process)
    if resume and is_master():
        root_dir = os.path.dirname(csv_path)
        prefix = os.path.basename(csv_path).replace('.csv', '_tmp_rank')
        pkl_to_del = []
        for f in os.listdir(root_dir):
            if prefix in f:
                pkl_path = f'{root_dir}/{f}'
                with open(pkl_path, 'rb') as f:
                    results_of_samples += pickle.load(f)
                print(f'Load results from {pkl_path}')   
                pkl_to_del.append(pkl_path)
                
        # merge all the loaded samples, del the tmp pickle files
        for pkl_path in pkl_to_del:
            os.remove(pkl_path)
            print(f'Del {pkl_path}')
        merge_pkl = csv_path.replace('.csv', f'_tmp_rank0.pkl')
        with open(merge_pkl, 'wb') as f:
            pickle.dump(results_of_samples, f)  
        print(f'Load results of {len(results_of_samples)} samples, Merge into {merge_pkl}')
                        
    model.eval()
    text_encoder.eval()
        
    with torch.no_grad():
        
        data_time = 0
        pred_time = 0
        metric_time = 0
        
        avg_patch_batch_num = 0
        avg_query_batch_num = 0
        
        # in ddp, only master process display the progress bar
        if is_master():
            testloader = tqdm(testloader, disable=False)
        else:
            testloader = tqdm(testloader, disable=True)  

        end_time = time.time()
        for sample in testloader:
            # data loading - MPL specific fields
            dataset_name = sample['dataset_name']
            sample_id = sample['sample_id'] 
            batched_patch_data = sample['batched_patch_data']  # MPL patch batches
            patch_indices = sample['patch_indices']           # For reconstruction
            pad_info = sample['pad_info']                     # For result restoration
            original_shape = sample['original_shape']         # Original image shape
            processed_shape = sample['processed_shape']       # Processed image shape
            split_labels = sample['split_labels'] 
            split_n1n2 = sample['split_n1n2']
            gt_segmentation = sample['gt_segmentation'].numpy()  # n h w d
            labels = sample['labels']
            modality = sample['modality']
            image_path = sample['image_path']
            raw_mask_path=sample['raw_mask_path']
            raw_mask_info=nib.load(raw_mask_path)
            patient_id=sample['patient_id']
            # print('image_path:',image_path)
            for key_,value_ in pad_info.items():
                print(f'key:{key_},value:{value_}')

            n, h, w, d = gt_segmentation.shape
            print('n, h, w, d:',n, h, w, d)
            
            data_time += (time.time()-end_time) 
            end_time = time.time()
            
            avg_patch_batch_num += len(batched_patch_data)
            avg_query_batch_num += len(split_labels)
            
            with autocast():
                # Convert list of texts to list of embeds
                queries_ls = []
                for labels_ls, n1n2 in zip(split_labels, split_n1n2):
                    queries_ls.append(text_encoder(labels_ls, modality))
                
                # Process each batch of MPL patches
                patch_predictions = []
                for batch_patches in batched_patch_data:
                    batch_preds = []
                    
                    for num_,patch_data in enumerate(batch_patches):
                        # img_cpu=patch_data['local_patch'].detach().cpu().numpy()
                        # img_cpu=img_cpu[0,0]
                        # nib.save(nib.nifti2.Nifti1Image(img_cpu, np.eye(4)),f"/data0/user/jlliu/git_pull_repos/SAT/demo/inference_demo/mapseg_results/train_latest_eval/CANDI13/img_{num_}_3.nii.gz")
                        local_patch = patch_data['local_patch'].to(device)      # [1, 1, 96, 96, 96]
                        global_img = patch_data['global_img'].to(device)        # [1, 1, 96, 96, 96]
                        coordinates = patch_data['coordinates'].to(device)      # [1, 6]
                        
                        # Call MPL model with local_patch and global_img (like test.py line 71)
                        # pred, _ = model(
                        #     local_patch=local_patch,
                        #     global_img=global_img,
                        #     coordinates=coordinates
                        # )
                        pred, local_mask = model(queries=queries_ls, image_input=global_img, local_patch=local_patch, coordinates=coordinates)   # bnhwd or list of bnhwd (deep supervision)
            
                        
                        # For SAT integration, we also need to handle text queries
                        # This requires checking the actual model interface
                        # For now, assume the model handles both MPL and text inputs
                        # if hasattr(model, 'forward_with_queries'):
                        #     # If model supports text queries, use them
                        #     pred = model.forward_with_queries(
                        #         queries=queries_ls,
                        #         local_patch=local_patch,
                        #         global_img=global_img,
                        #         coordinates=coordinates
                        #     )
                        
                        pred = torch.sigmoid(pred) if not torch.is_tensor(pred) or not pred.dtype == torch.bool else pred
                        # mask_cpu_=pred[0].detach().cpu().numpy()
                        # mask_cpu=np.zeros_like(img_cpu)
                        # for i in range(mask_cpu_.shape[0]):
                        #     tmp_=mask_cpu_[i]
                        #     mask_cpu[tmp_>0.5]=i+1
                        # nib.save(nib.nifti2.Nifti1Image(mask_cpu, np.eye(4)),f"/data0/user/jlliu/git_pull_repos/SAT/demo/inference_demo/mapseg_results/train_latest_eval/CANDI13/gt_{num_}_3.nii.gz")

                        # local_mask_cpu=local_mask.detach().cpu().numpy()
                        # local_mask_cpu=local_mask_cpu[0]
                        # local_mask_cpu=np.argmax(local_mask_cpu,axis=0)

                        # nib.save(nib.nifti2.Nifti1Image(local_mask_cpu.astype(np.float32), np.eye(4)),f"/data0/user/jlliu/git_pull_repos/SAT/demo/inference_demo/mapseg_results/train_latest_eval/CANDI13/local_mask_{num_}_3.nii.gz")
                        
                        batch_preds.append(pred.detach().cpu())
                    
                    patch_predictions.extend(batch_preds)
            
            pred_time += (time.time()-end_time)
            end_time = time.time()
            
            # Reconstruct full volume prediction from patches
            try:
                # Determine number of classes from model output
                if len(patch_predictions) > 0:
                    sample_pred = patch_predictions[0]
                    print('sample_pred.shape:',sample_pred.shape)
                    if sample_pred.dim() == 5:  # [B, C, H, W, D]
                        cls_num = sample_pred.shape[1]
                    elif sample_pred.dim() == 4:  # [C, H, W, D]
                        cls_num = sample_pred.shape[0]
                    else:
                        cls_num = len(labels)  # Fallback
                else:
                    cls_num = len(labels)
                
                # Reconstruct prediction using MPL approach
                reconstructed_pred,recon_bin = reconstruct_prediction_mpl(
                    patch_predictions, 
                    patch_indices, 
                    pad_info, 
                    processed_shape,
                    cls_num, 
                    device
                )
                print('reconstructed_pred.shape:',reconstructed_pred.shape)
                
                prediction = recon_bin
                # Convert to binary prediction for each label
                # prediction = np.zeros((n, h, w, d))
                # if reconstructed_pred.ndim == 3:  # Single class output
                #     prediction[0] = (reconstructed_pred > 0).astype(float)
                # else:
                #     # Multi-class: create binary masks for each label
                #     for j in range(min(n, reconstructed_pred.max() + 1)):
                #         prediction[j] = (reconstructed_pred == (j + 1)).astype(float)
                        
            except Exception as e:
                if is_master():
                    print(f"Error in reconstruction for sample {sample_id}: {e}")
                # Fallback: zero prediction
                prediction = np.zeros((n, h, w, d))
                            
            # Calculate metrics
            scores = []
            for j in range(len(labels)):
                scores.append(calculate_metric_percase(
                    prediction[j, :, :, :], 
                    gt_segmentation[j, :, :, :], 
                    dice_score, 
                    nsd_score
                ))
            
            # visualization  
            if visualization:
                Path(f'{nib_dir}/{dataset_name}').mkdir(exist_ok=True, parents=True)
                results = np.zeros((h, w, d))
                for j, label in enumerate(labels):
                    results += prediction[j, :, :, :] * (j+1)
                    Path(f'{nib_dir}/{dataset_name}/seg_{patient_id}').mkdir(exist_ok=True, parents=True)
                    # segobj = nib.nifti2.Nifti1Image(prediction[j, :, :, :], np.eye(4))
                    segobj = nib.Nifti1Image(prediction[j, :, :, :], raw_mask_info.affine, raw_mask_info.header)
                    nib.save(segobj, f'{nib_dir}/{dataset_name}/seg_{patient_id}/{label}.nii.gz')
                # segobj = nib.nifti2.Nifti1Image(results, np.eye(4))
                segobj = nib.Nifti1Image(results, raw_mask_info.affine, raw_mask_info.header)
                
                nib.save(segobj, f'{nib_dir}/{dataset_name}/seg_{patient_id}.nii.gz')
                
                # Save original image
                if hasattr(testset, 'load_image'):
                    image = testset.load_image(image_path)
                    image = np.squeeze(image)
                    # imgobj = nib.nifti2.Nifti1Image(image, np.eye(4))
                    imgobj = nib.Nifti1Image(image, raw_mask_info.affine, raw_mask_info.header)
                    nib.save(imgobj, f'{nib_dir}/{dataset_name}/img_{patient_id}.nii.gz')
                
                gt = np.zeros((h, w, d))
                for j, label in enumerate(labels):
                    gt += gt_segmentation[j, :, :, :] * (j+1)
                    Path(f'{nib_dir}/{dataset_name}/gt_{patient_id}').mkdir(exist_ok=True, parents=True)
                    # gtobj = nib.nifti2.Nifti1Image(gt_segmentation[j, :, :, :], np.eye(4))
                    gtobj = nib.Nifti1Image(gt_segmentation[j, :, :, :], raw_mask_info.affine, raw_mask_info.header)
                    nib.save(gtobj, f'{nib_dir}/{dataset_name}/gt_{patient_id}/{label}.nii.gz')
                # gtobj = nib.nifti2.Nifti1Image(gt, np.eye(4))
                gtobj = nib.Nifti1Image(gt, raw_mask_info.affine, raw_mask_info.header)
                nib.save(gtobj, f'{nib_dir}/{dataset_name}/gt_{patient_id}.nii.gz')
                
            # metric_time += (time.time()-end_time)
            # end_time = time.time()
            
            # # Store results
            # results_of_samples.append([dataset_name, modality, sample_id, scores, labels])
                
            # # Save intermediate results periodically
            # if is_master() and len(results_of_samples) % save_interval == 0:
            #     rank = dist.get_rank() if dist.is_initialized() else 0
            #     tmp_pkl = csv_path.replace('.csv', f'_tmp_rank{rank}.pkl')
            #     with open(tmp_pkl, 'wb') as f:
            #         pickle.dump(results_of_samples, f)
            #     if is_master():
            #         print(f'Saved {len(results_of_samples)} results to {tmp_pkl}')
            

            metric_time += (time.time()-end_time)
            end_time = time.time()
            
            # accumulate
            results_of_samples.append([dataset_name, modality, sample_id, scores, labels])
            
            # save in each process regularly in case of interruption
            if len(results_of_samples) % save_interval == 0:
                with open(csv_path.replace('.csv', f'_tmp_rank{dist.get_rank()}.pkl'), 'wb') as f:
                    pickle.dump(results_of_samples, f)
            
        """
        # gather results from all device to rank-0 (solution 1)
        gather_results = [None for i in range(dist.get_world_size())]
        dist.gather_object(
            results_of_samples, 
            gather_results if dist.get_rank() == 0 else None,
            dst = 0
            )
        
        if int(dist.get_rank()) == 0:
            results_of_samples = [tmp for ls in results_of_samples for tmp in ls]
        """         
                
        avg_patch_batch_num /= len(testloader)
        avg_query_batch_num /= len(testloader)
        data_time /= len(testloader)
        pred_time /= len(testloader)
        metric_time /= len(testloader)
        print(f'On Rank {dist.get_rank()}, each sample has {avg_patch_batch_num} batch of patches and {avg_query_batch_num} batch of queries, Data Time: {data_time}, Pred Time: {pred_time}, Dice Time: {metric_time}')
          
        torch.cuda.empty_cache()
        
        # save in each process (to a fnl pickle, also denoting this process ends)
        with open(csv_path.replace('.csv', f'_fnl_rank{dist.get_rank()}.pkl'), 'wb') as f:
            pickle.dump(results_of_samples, f)
                    
        # gather and record in rank 0 (solution 2)
        if is_master():
            
            # detect the finish of each process
            while True:
                all_process_finished = True
                for rank_id in range(torch.distributed.get_world_size()):
                    if not os.path.exists(csv_path.replace('.csv', f'_fnl_rank{rank_id}.pkl')): # xxx_tmp_rankx.pkl
                        all_process_finished = False
                        break
                if all_process_finished:
                    break
                else:
                    time.sleep(10)
            
            # read results of each process (samples may be duplicated due to the even distribution of ddp, check)
            results_of_samples = []    
            for rank_id in range(torch.distributed.get_world_size()):
                fnl_results_file = csv_path.replace('.csv', f'_fnl_rank{rank_id}.pkl')
                tmp_results_file = csv_path.replace('.csv', f'_tmp_rank{rank_id}.pkl')
                with open(fnl_results_file, 'rb') as f:
                    results_of_samples += pickle.load(f)
                os.remove(fnl_results_file)
                if os.path.exists(tmp_results_file):
                    os.remove(tmp_results_file)
                
            # check duplication
            unique_set = set()
            deduplicated_results_of_samples = []
            for dataset_name, modality, sample_id, scores, labels in results_of_samples:
                if f'{dataset_name}/{sample_id}' not in unique_set:
                    unique_set.add(f'{dataset_name}/{sample_id}')
                    deduplicated_results_of_samples.append([dataset_name, modality, sample_id, scores, labels])
            results_of_samples = deduplicated_results_of_samples
            
            # save for tmp
            with open(csv_path.replace('.csv', '.pkl'), 'wb') as f:
                pickle.dump(results_of_samples, f)

            # collate results
            for dataset_name, modality, sample_id, scores, labels in results_of_samples:    #  [[dataset_name, modality, sample_id, scores_of_labels(dict), label_names], ...]
                dataset_name = f'{dataset_name}({modality})'
                
                if dataset_name not in datasets_labels_metrics:
                    datasets_labels_metrics[dataset_name] = {}  # {'COVID19(CT)':{}}
                if dataset_name not in datasets_labels_sets:
                    datasets_labels_sets[dataset_name] = set()  # {'COVID19(CT)':set()}
                if dataset_name not in samples_labels_metrics:
                    samples_labels_metrics[dataset_name] = {}
                samples_labels_metrics[dataset_name][sample_id] = {}   # {'COVID19(CT)':{'0':{}}}
                
                for metric_dict, label in zip(scores, labels):
                    # accumulate metrics （for per dataset per class
                    # {'COVID19(CT)':{'covid19_infection':{'dice':[0.8, 0.9, ...], 'nsd':[0.8, 0.9, ...], ...} ...}, ...}
                    if label not in datasets_labels_metrics[dataset_name]:
                        datasets_labels_metrics[dataset_name][label] = {k:[v] for k,v in metric_dict.items()}
                    else:
                        for k,v in metric_dict.items():
                            datasets_labels_metrics[dataset_name][label][k].append(v)
                    
                    # statistic labels
                    # {'COVID19(CT)':set('covid19_infection', ...)}
                    if label not in datasets_labels_sets[dataset_name]:
                        datasets_labels_sets[dataset_name].add(label)
                    
                    # record metrics （for per dataset per sample per class
                    # {'COVID19':{'0.npy':{'covid19_infection':{'dice':0.8, 'nsd':0.9, ...} ...}, ...}
                    samples_labels_metrics[dataset_name][sample_id][label] = {k:v for k,v in metric_dict.items()}
                        
            # average and log (列为metrics，例如dice，nsd...)
            # create a df like:
            # {
            #   'TotalSegmentator': [0.xx, 0.xx, ...]    # 在T之前，这是一列
            #   'TotalSegmentator, Lung': [0.68, 0.72, ...]
            # }
            # by defult, print the dice (1st metric) of each dataset 
            info = 'Metrics of Each Dataset:\n'
            avg_df = {}
            for dataset in datasets_labels_metrics.keys():
                avg_df[dataset] = {k:[] for k in metric_dict.keys()}    # 'TotalSegmentator(CT)': {'dice':[0.8, ...] 'nsd':[0.5, ...], ...}
                for label in datasets_labels_metrics[dataset].keys():
                    avg_df[f'{dataset}, {label}'] = []
                    for metric in datasets_labels_metrics[dataset][label].keys():
                        label_metric = np.average(datasets_labels_metrics[dataset][label][metric])
                        avg_df[f'{dataset}, {label}'].append(label_metric)  # 'TotalSegmentator, Lung': [0.68, 0.72, ...] list of num_metrics
                        avg_df[dataset][metric].append(label_metric)
                avg_df[dataset] = {k:np.average(v) for k,v in avg_df[dataset].items()} # 'TotalSegmentator': {'dice':[0.8, ...] 'nsd':[0.5, ...], ...} --> 'TotalSegmentator': {'dice':0.x, 'nsd':0.x, ...}
                info += f'{dataset}  |  ' 
                for k ,v in avg_df[dataset].items():
                    info += f'{v}({k})  |  '
                info += '\n'
                avg_df[dataset] = list(avg_df[dataset].values())
            avg_df = pd.DataFrame(avg_df).T
            avg_df.columns = list(metric_dict.keys())   # ['dice', 'nsd']
            avg_df.to_csv(csv_path)        
            print(info)
            
            # detailed log （nsd和dice，列为class label
            # multi-sheet, two for each dataset
            df_list = [['summary', avg_df]]
            for dataset, label_set in datasets_labels_sets.items():
                metric_df ={}
                if dice_score:
                    metric_df['dice'] = {}
                if nsd_score:
                    metric_df['nsd'] = {}

                # create dfs like:
                # {
                #   '0.npy': [0.xx, 0.xx, ...]
                #   ......
                # }
                
                # {'COVID19':{'0.npy':{'covid19_infection':{'dice':0.8, ...} ...}, ...}
                for image_id, label_dict in samples_labels_metrics[dataset].items():
                    for metric in metric_df:
                        tmp = []    # one dice for each label in this dataset
                        for label in label_set:
                            score = label_dict[label][metric] if label in label_dict else -1
                            tmp.append(score)
                        metric_df[metric][image_id] = tmp   
                
                for metric, metric_df in metric_df.items():
                    metric_df = pd.DataFrame(metric_df).T
                    metric_df.columns = list(label_set)
                    df_list.append([dataset+f'({metric})', metric_df])
                
            xlsx_path = csv_path.replace('.csv', '.xlsx')
            with pd.ExcelWriter(xlsx_path) as writer:
                for name, df in df_list:
                    # 将每个 DataFrame 写入一个 sheet(sheet name must be < 31)
                    if len(name) > 31:
                        name = name[len(name)-31:]
                    df.to_excel(writer, sheet_name=name, index=True)    
                    
            # avg_dice_over_merged_labels, avg_nsd_over_merged_labels = merge(region_split_json, label_statistic_json, xlsx_path, xlsx_path)  
            
            os.remove(csv_path.replace('.csv', '.pkl'))
            
        else:
            
            pass
        
            # avg_dice_over_merged_labels = avg_nsd_over_merged_labels = 0
            
        return # avg_dice_over_merged_labels, avg_nsd_over_merged_labels
                    
        # Final save
    #     if is_master():
    #         rank = dist.get_rank() if dist.is_initialized() else 0
    #         tmp_pkl = csv_path.replace('.csv', f'_tmp_rank{rank}.pkl')
    #         with open(tmp_pkl, 'wb') as f:
    #             pickle.dump(results_of_samples, f)
    #         print(f'Final save: {len(results_of_samples)} results to {tmp_pkl}')
            
    #         # Print timing stats
    #         avg_patch_batch_num /= len(testloader)
    #         avg_query_batch_num /= len(testloader)
    #         print(f'Avg patch batch num: {avg_patch_batch_num:.2f}')
    #         print(f'Avg query batch num: {avg_query_batch_num:.2f}')
    #         print(f'Data loading time: {data_time:.2f}s')
    #         print(f'Prediction time: {pred_time:.2f}s') 
    #         print(f'Metric calculation time: {metric_time:.2f}s')
            
    # # Wait for all processes to finish
    # if dist.is_initialized():
    #     dist.barrier()
        
    # # Merge results in master process
    # if is_master():
    #     merge(csv_path)