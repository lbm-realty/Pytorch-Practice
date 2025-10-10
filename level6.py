from torch.utils.data import Dataset, DataLoader
import torch

class ToyDataset(Dataset):
    def __init__(self):
        self.x = torch.arange(1, 11, dtype=torch.float32).reshape(-1, 1)
        self.y = 2 * self.x + 3

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
dataset = ToyDataset()
loader = DataLoader(dataset, batch_size=5, shuffle=True)

for batch in loader:
    x_batch, y_batch = batch
    print(f"X: {x_batch}")
    print(f"Y: {y_batch}")
