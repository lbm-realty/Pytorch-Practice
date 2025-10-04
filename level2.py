import torch

class LinearRegression:
    # Simple Model: y = wx + b

    def __init__(self):
        self.w = torch.tensor(0.0, requires_grad=True)
        self.b = torch.tensor(0.0, requires_grad=True)

    # Forward pass -> prediction     
    def forward(self, x):
        return self.w * x + self.b
    
    # MSE
    def compute_loss(self, y_pred, y):
        return ((y_pred - y)**2).mean()
    
    def parameters(self):
        return [self.w, self.b]
    

def train(model, x_train, y_train, alpha, epochs, verbose):
    for iter in range(epochs):
        y_pred = model.forward(x_train)
        loss = model.compute_loss(y_pred, y_train)
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= alpha * param.grad
                param.grad.zero_()
        
        if verbose and iter % 10 == 0:
            print(f"Step {iter}: w={model.w:.2f}, b={model.b:.2f}, loss={loss:.2f}")

def learning():
    x = torch.tensor(2.0, requires_grad=True)
    y = x**2 + 3*x + 1
    z = y**2 + 1
    y.backward()

    # with torch.no_grad():
    #     y = x**2 + 3*x + 2

    x_detached = x.detach()
    a = y.grad_fn
    b = a.next_functions

    f1 = torch.tensor(2.0, requires_grad=True)
    f2 = torch.tensor(3.0, requires_grad=True)
    y1 = 8 * f1**2 + 4 * f2
    y1.backward()
    y1.backward()
    print(f1.grad, f2.grad)


if __name__ == "__main__":
    x_train = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_train = torch.tensor([3.0, 5.0, 7.0, 9.0]) 

    model = LinearRegression()

    # Train model
    train(model, x_train, y_train, alpha=0.01, epochs=100, verbose=True)