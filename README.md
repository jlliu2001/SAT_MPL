# SAT-MPL

# 1. Data preprocessing

## 1.1 extract subcortical regions from original labels
python /SAT_MPL/MPL_extract_subcortical_label.py --data_dir $dataset_dir --text_path "/SAT_MPL/data/1131_1_seg_Label.txt"

## 1.2 data transfer and preprocess
python /SAT_MPL/preprocess_private_brain_data.py --root_path $dataset_dir --output_jsonl 'your_output_jsonl_file' --dataset_name 'PrivateBrainData' --modality MRI

python /SAT_MPL/convert_to_npy.py --jsonl2load 'your_output_jsonl_file' --jsonl2save 'your_output_npy_jsonl_file' --image_npy_dir 'your_processed_npy_files_dir'

# 2. training
## 2.1 add the pretrained model
The download link of text encoder and the unet encoder(for the pretrained transformer decoder):https://huggingface.co/zzh99/SAT/tree/main/Others/UNET-Ours

## 2.2 (Optional) Pretrain Shape Encoder and Location Encoder for Proto-Atlas

After data preprocessing, you can optionally pretrain shape and location encoders to enable anatomical prior learning.

### 2.2.1 Pretrain Shape Encoder

The shape encoder learns SE(3)-equivariant shape representations through self-supervised reconstruction.

```bash
python pretrain_shape_encoder.py \
    --data_dir 'path_to_annotation_folder' \
    --output_dir './pretrained_encoders/shape_encoder' \
    --emb_dim 128 \
    --training_mode binary \
    --use_se3 \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --crop_size 96 96 96
```

**Key Parameters:**
- `--data_dir`: Folder containing subject folders with `*_seg.nii.gz` annotation files
- `--training_mode`: Training mode - `binary` (binary masks + noise), `soft` (Gaussian-blurred), or `mixed` (50-50, **recommended**)
- `--use_se3`: Use SE(3)-equivariant encoder (recommended for better rotation invariance)
- `--emb_dim`: Embedding dimension (default: 128)

**Output:** `shape_encoder_best.pth` (trained shape encoder checkpoint)

### 2.3.2 Pretrain Location Encoder

The location encoder learns anatomical spatial relationships through protocol-based reconstruction.

```bash
python pretrain_enhanced_loc_encoder.py \
    --data_dir 'path_to_annotation_folder' \
    --output_dir 'pretrained_encoders/loc_encoder' \
    --emb_dim 64 \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-3 \
    --use_enhanced_location
```

**Key Parameters:**
- `--data_dir`: Folder containing subject folders with `*_seg.nii.gz` annotation files
- `--use_enhanced_location`: Use enhanced 7-dimensional location features (includes adjacency information)
- `--emb_dim`: Embedding dimension (default: 64)

**Output:** `loc_encoder_best.pth` (trained location encoder checkpoint)

### Pretrained Encoders link
https://drive.google.com/drive/folders/1mZy_Ilh0f0NtQ2ytc2L4BmG-gfQx4FHK?usp=sharing
contains of `shape_encoder_best.pth`, `loc_encoder_best.pth`, `location_mean_prior.npz`, `shape_priors.npz`.


## 2.3 start training without Proto-Atlas Loss
the details can be seen in the "/SAT_MPL/sh/train_private_brain_mplseg.sh" file

bash "/SAT_MPL/sh/train_private_brain_mplseg.sh"
```
torchrun --nproc_per_node=${NPROC_PER_NODE} /SAT_MPL/train_mplseg.py \
    --datasets_jsonl "your_output_npy_jsonl_file" \
    --dataset_config "./SAT_MPL/data/72.json" \
    --vision_backbone "MAPSeg_MAE" \
    --text_encoder "ours" \
    --text_encoder_checkpoint "text_encoder_checkpoint_pretrained_by_SAT" \
    --checkpoint "mapseg_mpl_best_model.pth" \
    --mapseg_embed_dim 512 \
    --unet_checkpoint "unet_checkpoint_pretrained_by_SAT" \
    --freeze_encoder true \
    --freeze_text_encoder true \
    --step_num 10000 \ # the_number_of_training_epoches
    --warmup ${WARMUP} \
    --lr ${LEARNING_RATE} \
    --batchsize_3d ${BATCHSIZE_3D} \
    --accumulate_grad_interval 1 \
    --max_queries ${MAX_QUERIES} \
    --crop_size 256 128 156 \
    --patch_size 96 96 96 \
    --gpu 1 \ # the 
    --save_large_interval 20000 \ # Save the model parameters to step_{epoch}.pth every save_large_interval epoches
    --save_small_interval 10 \ # Save the model parameters to latest_step.pth every save_small_interval epoches
    --log_step_interval 500 \ # Deprecated: Print the current metric to the Log file every log_step_interval epoches(now every epoch will be printed)
    --log_dir "${LOG_DIR}" \
    --name "${EXPERIMENT_NAME}" \ # EXPERIMENT_NAME: the dir name of output checkpoints and log files
    --num_workers 8 \
    --allow_repeat true \
    --deep_supervision false \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --weight_decay 0.01 \
    --partial_load false \
    --resume false \
    --pin_memory False
```

## 2.4 start training with Proto-Atlas Loss
the details can be seen in the "/SAT_MPL/sh/train_private_brain_mplseg_newloss.sh" file

bash "/SAT_MPL/sh/train_private_brain_mplseg_newloss.sh"
```
torchrun --master_port=$MASTER_PORT  /data0/user/jlliu/git_pull_repos/SAT_MPL/train_mplseg.py \
    --datasets_jsonl "your_output_npy_jsonl_file" \
    --dataset_config "./SAT_MPL/data/72.json" \
    --vision_backbone "MAPSeg_MAE" \
    --text_encoder "ours" \
    --text_encoder_checkpoint "text_encoder_checkpoint_pretrained_by_SAT" \
    --checkpoint "mapseg_mpl_best_model.pth" \
    --mapseg_embed_dim 512 \
    --unet_checkpoint "unet_checkpoint_pretrained_by_SAT" \
    --freeze_encoder false \
    --freeze_text_encoder true \
    --step_num ${STEP_NUM} \
    --warmup ${WARMUP} \
    --lr ${LEARNING_RATE} \
    --batchsize_3d ${BATCHSIZE_3D} \
    --accumulate_grad_interval ${ACCUMULATE_GRAD} \
    --max_queries ${MAX_QUERIES} \
    --crop_size 256 256 256 \
    --patch_size 96 96 96 \
    --gpu 4 \
    --save_large_interval 100 \
    --save_small_interval 10 \
    --log_step_interval 500 \
    --log_dir "${LOG_DIR}" \
    --name "${EXPERIMENT_NAME}" \
    --num_workers 8 \
    --allow_repeat true \
    --deep_supervision false \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --weight_decay 0.01 \
    --partial_load false \
    --resume false \
    --cfg_file "./SAT_MPL/cfg/example_yaml/X_site_finetune70_testtime.yaml" \
    --pin_memory False \
    --use_proto_atlas \
    --shape_encoder_path "./SAT_MPL/pretrained_encoders/shape_encoder/shape_encoder_best.pth" \
    --loc_encoder_path "./SAT_MPL/pretrained_encoders/loc_encoder/enhanced_loc_encoder_best.pth" \
    --lambda_atlas 1 \
    --lambda_shape 1 \
    --lambda_loc 1 \
    --loc_emb_dim 64 \
    --use_e3nn True
```



# 3. evaluation
## 3.1 Standard Segmentation Evaluation
the details can be seen in the "/SAT_MPL/sh/evaluate_sat_mplseg.sh" file

bash "/SAT_MPL/sh/evaluate_sat_mplseg.sh"
```
torchrun \
--nproc_per_node 1 \
 "/data0/user/jlliu/git_pull_repos/SAT/evaluate_mplseg.py" \
--rcd_dir 'your_result_dir' \
--rcd_file 'save_name' \
--mapseg_embed_dim 512 \
--gpu 0 \
--resume False \
--visualization False \
--deep_supervision False \
--datasets_jsonl "your_output_npy_jsonl_file" \
--crop_size 256 128 256 \
--patch_size 96 96 96 \
--online_crop True \
--vision_backbone 'MAPSeg_MAE' \
--checkpoint "your_trained_checkpoint_pth_file" \
--partial_load True \
--text_encoder 'ours' \
--text_encoder_checkpoint "your_trained_text_encoder_checkpoint_pth_file" \
--batchsize_3d 1 \
--max_queries 256 \
--pin_memory False \
--num_workers 4 \
--dice True \
--nsd True
```
## 3.2 Proto-Atlas Quality Scoring (Post-Inference)
After segmentation inference, you can use the Proto-Atlas evaluator to compute anatomical plausibility scores for the segmentation results.

Batch evaluation (multiple files)
```
python example_proto_atlas_evaluator.py \
    --pred_path 'folder_with_predictions' \
    --batch_mode \
    --output_dir 'evaluation_results' \
    --shape_encoder_path 'pretrained_encoders/shape_encoder/shape_encoder_best.pth' \
    --shape_prior_path 'pretrained_encoders/shape_encoder/shape_priors.npz' \
    --location_mean_path 'pretrained_encoders/loc_encoder/location_mean_prior.npz' \
    --use_se3_shape \
    --use_enhanced_location \
    --use_direct_location \
    --direct_location_method method_a \
    --w 0.15 \
    --distance_metric cosine
```
**Output (Batch Mode):**
- `score_cosine.xlsx`: Excel file containing:
  - Per-file per-label shape scores
  - Per-file per-region location scores
  - Mean shape score
  - Overall location score
  - Overall quality score

**Use Cases:**
- Quality control: Identify potentially problematic segmentations
- Model comparison: Compare different models' anatomical plausibility
- Active learning: Select uncertain samples for manual review
- Test-time adaptation: Detect distribution shift in new datasets
