import pytorch_lightning as pl
from torch.utils.data import DataLoader
from wrapper.custom_dataset import Dataset_ETT
import ml_collections
import torch
from typing import Dict, Any, Tuple


class DataModule(pl.LightningDataModule):

    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__()
        self.config = config
        self.root_path = config.root_path
        self.data_path = config.data_path
        self.features = config.features
        self.target = config.target
        self.freq = config.freq
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.setup_datasets()

    def setup_datasets(self) -> None:

        self.train_dataset = Dataset_ETT(
            root_path=self.root_path,
            data_filename=self.data_path,
            flag="train",
            size=[self.config.seq_len, self.config.label_len, self.config.pred_len],
            features=self.features,
            target=self.target,
            scale=True,
            timeenc=1,
            freq=self.freq,
        )
        self.val_dataset = Dataset_ETT(
            root_path=self.root_path,
            data_filename=self.data_path,
            flag="val",
            size=[self.config.seq_len, self.config.label_len, self.config.pred_len],
            features=self.features,
            target=self.target,
            scale=True,
            timeenc=1,
            freq=self.freq,
        )

        self.test_dataset = Dataset_ETT(
            root_path=self.root_path,
            data_filename=self.data_path,
            flag="test",
            size=[self.config.seq_len, self.config.label_len, self.config.pred_len],
            features=self.features,
            target=self.target,
            scale=True,
            timeenc=1,
            freq=self.freq,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
