import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from regGPT import *


class RegDataset(Dataset):
    def __init__(self, file, train=True, test_size=0.2, random_state=42):
        super().__init__()

        try:
            data = pd.read_csv(file)
            train_data, val_data = train_test_split(
                data, test_size=test_size, random_state=random_state)
            self.data = train_data if train else val_data

            # Convert data to tensors and reshape to [N, 1]
            self.inputs = torch.tensor(
                self.data.iloc[:, :-1].values, dtype=torch.float32).view(-1, 1)
            self.target = torch.tensor(
                self.data.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)
        except:
            raise FileNotFoundError("File not found")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.inputs[i], self.target[i]

    def get_dataloader(self, batch_size=1000, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
