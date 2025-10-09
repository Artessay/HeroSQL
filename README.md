# HereSQL

Setup environment:

```bash
conda create -n hero python=3.12 -y
conda activate hero

pip install -e .
```

Train:

```bash
python src/main.py \
    --seed=2025 \
    --model.model_name="hero" \
    --dataset.dataset_name="all" \
    --dataset.batch_size=32 \
    --trainer.max_epochs=50 \
    --trainer.check_val_every_n_epoch=1 \
    --model.embedding_model_name="Qwen/Qwen3-Embedding-0.6B"
```

Inference:

```bash
python src/inference.py \
    --seed=2025 \
    --model.model_name="plain" \
    --dataset.dataset_name="bird" \
    --dataset.batch_size=48 \
    --trainer.devices=1

python src/inference.py \
    --seed=2025 \
    --model.model_name="lope" \
    --dataset.dataset_name="bird" \
    --dataset.batch_size=32 \
    --trainer.devices=1
```

To view the Tensorboard, run

```bash
tensorboard --logdir=lightning_logs
```

## Download Model from ModelScope

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-0.6B  --local_dir /data/Qwen/Qwen3-0.6B
modelscope download --model Qwen/Qwen3-Embedding-0.6B  --local_dir /data/Qwen/Qwen3-Embedding-0.6B
modelscope download --model google/embeddinggemma-300m  --local_dir /data/google/embeddinggemma-300m
modelscope download --model google/gemma-3-270m-it  --local_dir /data/google/gemma-3-270m-it
modelscope download --model Qwen/Qwen3-4B  --local_dir /data/Qwen/Qwen3-4B
modelscope download --model Qwen/Qwen3-Embedding-4B  --local_dir /data/Qwen/Qwen3-Embedding-4B
```