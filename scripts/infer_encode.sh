#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# MODEL=BAAI/bge-large-en
MODEL=thenlper/gte-large

BATCH_SIZE=512


for DATASET in bird spider ehr
do
    python src/infer_aug.py \
        --seed=2025 \
        --model.model_name="encode" \
        --dataset.dataset_name="$DATASET" \
        --dataset.batch_size=${BATCH_SIZE} \
        --model.embedding_model_name="$MODEL"
done