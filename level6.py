from torch.utils.data import Dataset, DataLoader
import torch

class ToyDataset(Dataset):
    def __init__(self):
        self.x = torch.arange(1, 11, dtype=torch.float32).reshape(-1, 1)
        self.y = 2 