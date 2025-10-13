import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input, hidden_1, hidden_2, output):
        super().__init__()
        self.fc1 = nn.Linear(input, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, output)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def train(x, y, model, epochs, optimizer, loss_fn):
    for epoch in range(epochs):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 5000 == 0:
            print(f"For {epoch}th epoch, loss: {loss}")

def test(x, y, model):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        print(prediction)
        print(y)
        

if __name__ == "__main__":
    x = torch.randn(10, 1) 
    y = 2 * x + 3 * torch.randn(10, 1)
    model = NeuralNetwork(input=1, hidden_1=10, hidden_2=10, output=1)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(x, y, model, 50000, optimizer, loss_fn)
    test(x, y, model)
