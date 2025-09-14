# SAT-MPL

## 1. Data preprocessing

# 1.1 extract subcortical regions from original labels
python /SAT_MPL/MPL_extract_subcortical_label.py --data_dir $dataset_dir --text_path "/SAT_MPL/data/1131_1_seg_Label.txt"

# 1.2 data transfer and preprocess
python /SAT_MPL/preprocess_private_brain_data.py --root_path $dataset_dir --output_jsonl 'your_output_jsonl_file' --dataset_name 'PrivateBrainData' --modality MRI
python /SAT_MPL/convert_to_npy.py --jsonl2load 'your_output_jsonl_file' --jsonl2save 'your_output_npy_jsonl_file' --image_npy_dir 'your_processed_npy_files_dir'

## 2. training
# 2.1 add the pretrained model
The download link of text encoder and the unet encoder(for the pretrained transformer decoder):https://huggingface.co/zzh99/SAT/tree/main/Others/UNET-Ours

# 2.2 start training
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
# 3. evaluation
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