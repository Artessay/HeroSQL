#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3,7

# MODEL=/data/Qwen/Qwen3-0.6B
MODEL=/data/google/gemma-3-270m-it

for DATASET in bird spider ehr
do
  for METHOD in llm cot confidence cove
  do
    for SEED in 2025 # 2026 2027 2028 2029
    do
      echo "Running: python src/baseline/main.py -m $METHOD -d $DATASET -s $SEED -l $MODEL"
      python src/baseline/main.py -m "$METHOD" -d "$DATASET" -s "$SEED" -l "$MODEL"
    done
  done
done