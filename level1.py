import torch

x = torch.tensor([[10, 20, 30], 
                  [40, 50, 60], 
                  [70, 80, 90],
                  [100, 110, 120]])
x1 = x[1, 2]
x2 = x[:, 1:]
x3 = x[:, :1]
x4 = x[:2, :-1]
x5 = x[:2, -1]      # Prints elements as a row vector
x6 = x[:2, -1:]     # Prints elements as a column vector
x7 = x[[0, 2]]
x8 = x[:, [1, 2]]
x9 = x > 50
x10 = x[x9]
x11 = x.shape
x12 = x.view(2, 6)
x13 = x.reshape(3, 4)
x14 = x.reshape(-1)     # Flattens to 1-D
x15 = x.t()

y = torch.randn(1, 2, 4)
x17 = y.permute(1, 0, 2)

a = torch.tensor([[1, 2],
                  [3, 4]])
b = torch.tensor([[5, 6]])
c = torch.cat((a, b), dim=0)
d = torch.tensor([1, 2, 3])
e = torch.tensor([[100],
                  [200],
                  [300]])
print(d + e)
