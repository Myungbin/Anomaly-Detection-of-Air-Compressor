"""
Deep Learning
"""
import torch
from torch.utils.data import Dataset, DataLoader

from src.config.config import cfg


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class DatasetLoader:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def _dataset_init(self):
        train_dataset = CustomDataset(self.train_data)
        test_dataset = CustomDataset(self.test_data)

        return train_dataset, test_dataset

    def _dataloader_init(self, train_dataset, test_dataset):
        train_loader = DataLoader(train_dataset, shuffle=False, batch_size=cfg.BATCH_SIZE)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=cfg.BATCH_SIZE)

        return train_loader, test_loader

    @property
    def load(self):
        train_dataset, test_dataset = self._dataset_init()
        train_loader, test_loader = self._dataloader_init(train_dataset, test_dataset)
        return train_loader, test_loader
