#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

# MODEL=/data/Qwen/Qwen3-Embedding-0.6B
MODEL=/data/google/embeddinggemma-300m
METHOD=hero
BATCH_SIZE=32


for DATASET in bird spider ehr
do
    # python src/infer_aug.py \
    python src/inference.py \
        --seed=2025 \
        --model.model_name="$METHOD" \
        --dataset.dataset_name="$DATASET" \
        --dataset.batch_size=${BATCH_SIZE} \
        --model.embedding_model_name="$MODEL"
done