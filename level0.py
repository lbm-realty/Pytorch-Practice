import torch
import numpy as np

# print(torch.__version__)
# print(torch.cuda.is_available())

x = torch.tensor([1, 2, 3, 4])
arr = np.array([5, 6, 7])
y = torch.from_numpy(arr)
rand_tensor = torch.rand(1, 3)
z = torch.zeros(2, 2)
o = torch.ones(1, 4)
a = torch.rand(2, 3)
sh = a.shape
dt = a.dtype
d = a.device
x1 = torch.tensor([1, 2, 3])
x2 = torch.tensor([4, 5, 6])
sum1 = torch.sum(x1)
dot_p = torch.dot(x1, x2)
print(dot_p)
