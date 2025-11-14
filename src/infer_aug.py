import os
from pytorch_lightning import Trainer, seed_everything

from src.dataset import TripleStreamDataModule
from src.model import TripleFusionLightningModule
from src.utils.lightning_utils import print_only_on_rank_zero, evaluate_and_save_on_rank_zero


def run(args):
    seed = args.seed
    seed_everything(seed, workers=True)

    method_name = args.model.model_name
    embedding_model_name = args.model.embedding_model_name.split('/')[-1]
    
    checkpoint_path = f"./checkpoints/{method_name}/aug_{embedding_model_name}_{seed}.ckpt"
    save_path = f'results/{args.dataset.dataset_name}/{embedding_model_name}/aug-{method_name}-{seed}.json'

    # if os.path.exists(save_path):
    #     print_only_on_rank_zero(f"[warning] Result file {save_path} already exists, skipping inference.")
    #     return

    # prepare dataset and model
    if os.path.exists(checkpoint_path):
        print_only_on_rank_zero(f"Loading model from checkpoint: {checkpoint_path}")
        model = TripleFusionLightningModule.load_from_checkpoint(checkpoint_path)
    else:
        assert False, f"[error] No checkpoint found for {method_name} on {checkpoint_path}."
        print_only_on_rank_zero("[warning] No checkpoint found, initializing a new model.")
        model = TripleFusionLightningModule(**args.model)
    
    args.dataset['method_name'] = method_name
    args.dataset['embedding_model_name'] = model.hparams.embedding_model_name
    dm = TripleStreamDataModule(**args.dataset)

    # args.trainer.devices, args.trainer.num_nodes = 1, 1
    trainer = Trainer(**args.trainer)
    trainer.test(model, dm)
    
    # save evaluation result
    score_list, label_list = model.test_probs, model.test_labels
    evaluate_and_save_on_rank_zero(score_list, label_list, save_path)
    
if __name__ == '__main__':
    import torch
    torch.set_float32_matmul_precision('high')

    from src.utils.lightning_params import get_lightning_args
    args = get_lightning_args()
    run(args)