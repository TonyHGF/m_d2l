from dataloader.dataset import MyDataset as MyDataset
from torch.utils.data import DataLoader
import argparse

source_arr = [1, 2, 3, 4, 5]
target_arr = [1, 2, 3, 4, 5]

dataloader_args = {
    'batch_size': 1,
    'shuffle': True,
    'num_workers': 8
}


def main():
    train_dataset = MyDataset(source_arr, target_arr)
