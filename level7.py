import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))        
        x = self.fc3(x)
        return x
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

raw_data = datasets.MNIST(
    root="data",
    train=True,
    download=True
)

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self):
        self.model.train()
        curr_loss = 0
        for images, labels in self.train_loader:
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            curr_loss += loss.item()
        return curr_loss / len(self.train_loader)
                
    def test(self):
        self.model.eval()
        correct, total = 0, 0
        pred = torch.tensor([])
        label = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                unknown, predicted = torch.max(outputs, 1)
                pred = unknown
                total += labels.size(0)
                label = labels
                print(labels.shape, predicted.shape)
                correct += (predicted == labels).sum().item()
        print(pred)
        return correct / total


if __name__ == "__main__":
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer)
    epochs = 1

    for epoch in range(epochs):
        train_loss = trainer.train()
        test_accuracy = trainer.test()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Test: {test_accuracy:.4f}")
