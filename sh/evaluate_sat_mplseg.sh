#!/bin/bash
#SBATCH --job-name=eval_mapseg
#SBATCH --quotatype=auto
#SBATCH --partition=medai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=128G
#SBATCH --chdir=logs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.error
###SBATCH -w SH-IDC1-10-140-0-[...], SH-IDC1-10-140-1-[...]
###SBATCH -x SH-IDC1-10-140-0-[...], SH-IDC1-10-140-1-[...]

# export NCCL_DEBUG=INFO
# export NCCL_IBEXT_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# echo NODELIST=${SLURM_NODELIST}
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# MASTER_PORT=$((RANDOM % 101 + 20000))
# echo "MASTER_ADDR="$MASTER_ADDR

# srun torchrun \
# --nnodes 1 \
# --nproc_per_node 1 \
# --rdzv_id 100 \
# --rdzv_backend c10d \
# --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT evaluate_mapseg.py \
# --rcd_dir 'evaluation_results_mapseg' \
# --rcd_file 'mapseg_evaluation.pkl' \
# --resume False \
# --visualization False \
# --deep_supervision False \
# --datasets_jsonl 'testset.jsonl' \
# --crop_size 288 288 96 \
# --online_crop True \
# --vision_backbone 'MAPSeg_MAE' \
# --mapseg_embed_dim 512 \
# --checkpoint 'weights/SAT_MAPSeg_MAE.pth' \
# --partial_load True \
# --text_encoder 'ours' \
# --text_encoder_checkpoint 'weights/text_encoder.pth' \
# --batchsize_3d 2 \
# --max_queries 256 \
# --pin_memory False \
# --num_workers 4 \
# --dice True \
# --nsd True

MASTER_PORT=12345

torchrun \
--nproc_per_node 1 \
 "/data0/user/jlliu/git_pull_repos/SAT_MPL/evaluate_mplseg.py" \
--rcd_dir "/data0/user/jlliu/git_pull_repos/SAT/demo/inference_demo/mapseg_results" \
--rcd_file 'train_latest_eval' \
--mapseg_embed_dim 512 \
--gpu 0 \
--resume False \
--visualization False \
--deep_supervision False \
--datasets_jsonl "/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/CANDI13_data_v3_npy.jsonl" \
--crop_size 256 128 256 \
--patch_size 96 96 96 \
--online_crop True \
--vision_backbone 'MAPSeg_MAE' \
--checkpoint "/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/log/private_brain_mapseg_finetune/private_brain_mapseg_finetune/checkpoint/latest_step.pth" \
--partial_load True \
--text_encoder 'ours' \
--text_encoder_checkpoint "/data0/user/jlliu/git_pull_repos/SAT/demo/train_demo/log/private_brain_mapseg_finetune/private_brain_mapseg_finetune/checkpoint/text_encoder_latest_step.pth" \
--batchsize_3d 1 \
--max_queries 256 \
--pin_memory False \
--num_workers 4 \
--cfg_file "/data0/user/jlliu/git_pull_repos/SAT_MPL/cfg/example_yaml/X_site_finetune70_testtime.yaml" \
--dice True \
--nsd True