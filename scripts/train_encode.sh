#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3

MODEL=BAAI/bge-large-en
# MODEL=thenlper/gte-large

BATCH_SIZE=512

python src/main.py \
    --seed=2025 \
    --model.model_name="encode" \
    --dataset.dataset_name="aug" \
    --dataset.batch_size=${BATCH_SIZE} \
    --trainer.max_epochs=50 \
    --trainer.check_val_every_n_epoch=1 \
    --model.embedding_model_name="$MODEL"