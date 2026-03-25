#!/bin/bash
# ==============================================================================
# Script for training BRNet
# Usage: bash train.sh
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables (Modify these if you want to change the dataset or model name)
DATASET="arcade"
ARCH="BiFormer_resnet18"
NAME="${DATASET}_${ARCH}_woDS"  # 您的专属实验名称

echo "=> Starting Training for ${NAME}..."

# Run the training script in the background using nohup
# CUDA_VISIBLE_DEVICES=0 means it will only use the first GPU
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
    --dataset ${DATASET} \
    --arch ${ARCH} \
    --name ${NAME} \
    --epochs 100 \
    --batch_size 4 \
    --input_w 512 \
    --input_h 512 \
    --optimizer "Adam" \
    --lr 1e-4 \
    --scheduler "CosineAnnealingLR" \
    --early_stopping 20 \
    > train_${NAME}.log 2>&1 &

echo "=> Training started in the background!"
echo "=> You can check the progress by running: tail -f train_${NAME}.log"