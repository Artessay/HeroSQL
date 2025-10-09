import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.dataset import TripleStreamDataModule
from src.model import TripleFusionLightningModule
from src.utils.lightning_utils import print_only_on_rank_zero

def run(args):
    seed = args.seed
    seed_everything(seed, workers=True)

    # prepare dataset and model
    model = TripleFusionLightningModule(**args.model)

    method_name = model.hparams.model_name
    dataset_name = args.dataset.dataset_name
    embedding_model_name = model.hparams.embedding_model_name
    embedding_model_abbreviation = 'qwen' if 'qwen' in embedding_model_name.lower() else 'gemma'
    checkpoint_path = f"./checkpoints/{method_name}/{dataset_name}_{embedding_model_abbreviation}_{seed}.ckpt"
    # if os.path.exists(checkpoint_path):
    #     print_only_on_rank_zero(f"[warning] Checkpoint {checkpoint_path} already exists, skipping training.")
    #     return

    args.dataset['method_name'] = method_name
    args.dataset['embedding_model_name'] = embedding_model_name
    dm = TripleStreamDataModule(**args.dataset)

    logger = TensorBoardLogger("./", default_hp_metric=False)

    # define callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_auc_roc",
        mode="max",
        patience=5
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{logger.log_dir}/checkpoints",
        filename='{epoch:02d}-{val_auc_roc:.2f}',
        monitor='val_auc_roc',
        mode='max',
        save_top_k=1,  # save best checkpoints
    )

    args.trainer['logger'] = logger
    args.trainer['callbacks'] = [early_stop_callback, checkpoint_callback]
    trainer = Trainer(**args.trainer)

    trainer.fit(model, dm)
    trainer.test(model, dm)

    
if __name__ == '__main__':
    import torch
    torch.set_float32_matmul_precision('high')

    from src.utils.lightning_params import get_lightning_args
    args = get_lightning_args()

    run(args)