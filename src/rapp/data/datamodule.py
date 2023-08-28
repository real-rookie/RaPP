from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms as T
import pytorch_lightning as pl

from .dataset import CustomDataset, _flatten, _normalize


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_normal: str,
        dataset_novel: str,
        setting: str, # options:["SIMO", "inter_set", "set_to_set"]
        data_dir: str = "./data",
        num_workers: int = 8,
        seed: int = 42,
        batch_size: int = 256,
        normal_label: int = 0,
    ):
        super().__init__()
        self.class_dict = {
            "MNIST": MNIST,
            "FashionMNIST": FashionMNIST,
            "CIFAR": CIFAR10,
        }
        assert dataset_normal in self.class_dict.keys()
        assert dataset_novel in self.class_dict.keys()
        assert dataset_normal != dataset_novel
        self.dataset_normal = dataset_normal
        self.dataset_novel = dataset_novel
        if dataset_normal in ["MNIST", "FashionMNIST"]:
            self.image_size = 1 * 28 * 28
        elif dataset_normal == "CIFAR10":
            self.image_size = 3 * 32 * 32
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.seed = seed
        self.batch_size = batch_size
        self.normal_label = normal_label
        self.setting = setting
        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ...
        self.test_transforms = self.default_transforms

    def prepare_data(self):
        if self.setting == "SIMO":
            self.prepare_data_SIMO()
        elif self.setting == "inter_set":
            self.prepare_data_inter_set()
        elif self.setting == "set_to_set":
            self.prepare_data_set_to_set()

    def prepare_data_SIMO(self):
        """Saves files to `data_dir`"""
        self.class_dict[self.dataset_normal](self.data_dir, train=True, download=True)
        self.class_dict[self.dataset_normal](self.data_dir, train=False, download=True)
    
    def prepare_data_inter_set(self):
        self.prepare_data_SIMO()

    def prepare_data_set_to_set(self):
        # normal dataset
        self.class_dict[self.dataset_normal](self.data_dir, train=True, download=True)
        self.class_dict[self.dataset_normal](self.data_dir, train=False, download=True)
        # novel dataset
        self.class_dict[self.dataset_novel](self.data_dir, train=True, download=True)
        self.class_dict[self.dataset_novel](self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if self.setting == "SIMO":
            self.setup_SIMO()
        elif self.setting == "inter_set":
            self.setup_inter_set()
        elif self.setting == "set_to_set":
            self.setup_set_to_set()

    def setup_SIMO(self):
        train_dataset = self.class_dict[self.dataset_normal](self.data_dir, train=True, download=False)
        test_dataset = self.class_dict[self.dataset_normal](self.data_dir, train=False, download=False)

        train_data, train_label = torch.tensor(train_dataset.data), torch.tensor(train_dataset.targets)
        test_data, test_label = torch.tensor(test_dataset.data), torch.tensor(test_dataset.targets)
        data = torch.cat([train_data, test_data])
        labels = torch.cat([train_label, test_label])

        # split data with seen labels and unseen labels
        seen_idx, unseen_idx = None, None
        if self.setting == "SIMO":
            seen_idx = labels == self.normal_label
            unseen_idx = labels != self.normal_label
        elif self.setting == "inter_set":
            seen_idx = labels < 5
            unseen_idx = labels >= 5

        assert seen_idx is not None and unseen_idx is not None
        seen_data = data[seen_idx]
        unseen_data = data[unseen_idx]

        self.split_dataset(seen_data, unseen_data)

    def setup_inter_set(self):
        self.setup_SIMO()

    def setup_set_to_set(self):
        # normal dataset
        train_dataset_normal = self.class_dict[self.dataset_normal](self.data_dir, train=True, download=False)
        test_dataset_normal = self.class_dict[self.dataset_normal](self.data_dir, train=False, download=False)
        train_data_normal = torch.tensor(train_dataset_normal.data)
        test_data_normal = torch.tensor(test_dataset_normal.data)
        seen_data = torch.cat([train_data_normal, test_data_normal])
        # novel dataset
        train_dataset_novel = self.class_dict[self.dataset_novel](self.data_dir, train=True, download=False)
        test_dataset_novel = self.class_dict[self.dataset_novel](self.data_dir, train=False, download=False)
        train_data_novel = torch.tensor(train_dataset_novel.data)
        test_data_novel = torch.tensor(test_dataset_novel.data)
        unseen_data = torch.cat([train_data_novel, test_data_novel])

        self.split_dataset(seen_data, unseen_data)

    def split_dataset(self, seen_data, unseen_data):
        # TODO transforms for unseen_dataset

        train_size = int(seen_data.size(0) * 0.6)
        valid_size = int(seen_data.size(0) * 0.2)
        test_size = len(seen_data) - train_size - valid_size

        sample_idx = np.random.choice(len(unseen_data), test_size, replace=False)
        # auprc baseline is always 0.5
        sample_idx.sort()
        unseen_data = unseen_data[sample_idx]

        seen_dataset = CustomDataset(
            seen_data, torch.Tensor([0] * len(seen_data)), self.default_transforms
        )
        unseen_dataset = CustomDataset(
            unseen_data, torch.Tensor([1] * len(unseen_data)), # FIXME
        )

        self.dataset_train, self.dataset_val, test_data = random_split(
            seen_dataset, [train_size, valid_size, test_size]
        )
        # make test data with seen data and unseen data
        self.dataset_test = ConcatDataset([test_data, unseen_dataset])

    def train_dataloader(self):
        """MNIST train set removes a subset to use for validation"""
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        """MNIST val set uses a subset of the training set for validation"""
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        """MNIST test set uses the test split"""
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    @property
    def default_transforms(self):
        #TODO different datasets have different sizes
        transforms = []
        transforms.append(T.Lambda(_normalize))
        transforms.append(T.Lambda(_flatten))
        transforms = T.Compose(transforms)
        return transforms
