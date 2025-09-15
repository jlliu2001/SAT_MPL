#!/bin/bash

# Private Brain Data MAPSeg Training Script
# This script trains MAPSeg_MAE model on private brain segmentation data
# with frozen encoder and text encoder for semantic alignment fine-tuning

# =====================================================
# CONFIGURATION PARAMETERS
# =====================================================

# Data paths
DATA_JSONL="/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/CANDI13_data_v4_npy.jsonl"  # UPDATE THIS PATH
DATASET_CONFIG="/data0/user/jlliu/git_pull_repos/SAT/data/dataset_config/72.json"

# Model configuration
VISION_BACKBONE="MAPSeg_MAE"
TEXT_ENCODER="ours"
TEXT_ENCODER_CHECKPOINT="/data0/user/jlliu/git_pull_repos/SAT/weights/text_encoder.pth"  # UPDATE THIS PATH

# MAPSeg specific paths
MAPSEG_PRETRAINED_PATH="/data0/user/jlliu/git_pull_repos/SAT/weights/mapseg_mpl_best_model.pth"  # UPDATE THIS PATH
UNET_CHECKPOINT="/data0/user/jlliu/git_pull_repos/SAT/weights/SAT_Nano_UNET.pth"  # Optional: for transformer decoder

# Training parameters
STEP_NUM=10000  # Reduced for fine-tuning
WARMUP=5000
LEARNING_RATE=1e-4  # Lower learning rate for fine-tuning
BATCHSIZE_3D=1  # Adjust based on GPU memory
ACCUMULATE_GRAD=1  # Effective batch size = BATCHSIZE_3D * ACCUMULATE_GRAD
MAX_QUERIES=32  # Adjust based on number of brain regions

# GPU configuration
NPROC_PER_NODE=1  # Number of GPUs to use
MASTER_PORT=12345

# Output configuration
EXPERIMENT_NAME="private_brain_mapseg_finetune"
LOG_DIR="/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/log/${EXPERIMENT_NAME}"

# =====================================================
# TRAINING COMMAND
# =====================================================

echo "Starting MAPSeg training on private brain data..."
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Vision backbone: ${VISION_BACKBONE}"
echo "Data: ${DATA_JSONL}"
echo "Batch size: ${BATCHSIZE_3D} x ${ACCUMULATE_GRAD} = $(($BATCHSIZE_3D * $ACCUMULATE_GRAD))"
echo "Steps: ${STEP_NUM}"
echo "Learning rate: ${LEARNING_RATE}"
echo ""

torchrun --nproc_per_node=${NPROC_PER_NODE} /data0/user/jlliu/git_pull_repos/SAT_MPL/train_mplseg.py \
    --datasets_jsonl "${DATA_JSONL}" \
    --dataset_config "${DATASET_CONFIG}" \
    --vision_backbone "${VISION_BACKBONE}" \
    --text_encoder "${TEXT_ENCODER}" \
    --text_encoder_checkpoint "${TEXT_ENCODER_CHECKPOINT}" \
    --checkpoint "${MAPSEG_PRETRAINED_PATH}" \
    --mapseg_embed_dim 512 \
    --unet_checkpoint "${UNET_CHECKPOINT}" \
    --freeze_encoder true \
    --freeze_text_encoder true \
    --step_num ${STEP_NUM} \
    --warmup ${WARMUP} \
    --lr ${LEARNING_RATE} \
    --batchsize_3d ${BATCHSIZE_3D} \
    --accumulate_grad_interval ${ACCUMULATE_GRAD} \
    --max_queries ${MAX_QUERIES} \
    --crop_size 256 128 156 \
    --patch_size 96 96 96 \
    --gpu 1 \
    --save_large_interval 20000 \
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
    --cfg_file "/data0/user/jlliu/git_pull_repos/SAT_MPL/cfg/example_yaml/X_site_finetune70_testtime.yaml" \
    --pin_memory False

echo ""
echo "Training completed!"
echo "Results saved in: ${LOG_DIR}"
echo ""
echo "To monitor training progress:"
echo "  tensorboard --logdir=${LOG_DIR}"
echo ""
echo "To resume training (if interrupted):"
echo "  Change --resume to true and specify --checkpoint path"