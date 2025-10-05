import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
def train(x_train, y_train, epochs, loss_fn, optimizer, model):
    for iter in range(epochs):
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (iter+1) % 20 == 0:
            print(f"Epoch [{iter+1}/{epochs}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    x = torch.randn(100, 1) * 10
    y = 2 * x + 3 + torch.randn(100, 1) * 10
    model = NeuralNet(input_size=1, hidden_size=10, output_size=1)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(x, y, 200, loss_fn, optimizer, model)
