#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,3,5,6

# MODEL=/data/Qwen/Qwen3-Embedding-0.6B
MODEL=/data/google/embeddinggemma-300m
METHOD=hero
BATCH_SIZE=64

python src/main.py \
    --seed=2025 \
    --model.model_name="$METHOD" \
    --dataset.dataset_name="aug" \
    --dataset.batch_size=${BATCH_SIZE} \
    --trainer.max_epochs=50 \
    --trainer.check_val_every_n_epoch=1 \
    --model.embedding_model_name="$MODEL"