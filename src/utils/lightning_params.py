
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning import Trainer

from src.dataset import TripleStreamDataModule
from src.model import TripleFusionLightningModule
from src.utils.lightning_utils import print_only_on_rank_zero

def get_lightning_args():
    parser = LightningArgumentParser()

    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--version", type=int, default=None)

    parser.add_lightning_class_args(Trainer, "trainer")
    parser.add_lightning_class_args(TripleStreamDataModule, "dataset")
    parser.add_lightning_class_args(TripleFusionLightningModule, "model")

    args = parser.parse_args()
    print_only_on_rank_zero(args)

    return args