import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.dataset.triple_stream_dataset import TripleStreamDataset

class TripleStreamDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            dataset_name: str = "bird", 
            batch_size=4, 
            num_workers=64,
            method_name: str = None,
            embedding_model_name: str = None,
        ):
        super().__init__()
        
        self.batch_size = batch_size  
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.embedding_model_name = embedding_model_name

    def setup(self, stage):
        if stage == 'fit':
            # load dataset
            train_dataset = TripleStreamDataset(self.dataset_name, "train", self.method_name, self.embedding_model_name)

            # split train dataset into train and val
            self.train_dataset, self.val_dataset = random_split(train_dataset, [0.8, 0.2])
        elif stage == 'test':
            self.test_dataset = TripleStreamDataset(self.dataset_name, "dev", self.method_name, self.embedding_model_name)
        else:
            raise NotImplementedError(f"Stage `{stage}` not supported")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.train_dataset.dataset.multimodal_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.val_dataset.dataset.multimodal_collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.test_dataset.multimodal_collate_fn
        )
