import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

raw_data = datasets.MNIST(
    root="data",
    train=True,
    download=True
)

img, label = raw_data[0]

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize((0.5,), (0.5,))

img_tensor = to_tensor(img)
img_normalized = normalize(img_tensor)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Original PIL image (0–255)
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original (PIL 0–255)")
axes[0].axis("off")

# After ToTensor() (0–1)
axes[1].imshow(img_tensor.squeeze(), cmap="gray", vmin=0, vmax=1)
axes[1].set_title("After ToTensor() (0–1)")
axes[1].axis("off")

# After Normalize() (-1–1)
axes[2].imshow(img_normalized.squeeze(), cmap="gray", vmin=-1, vmax=1)
axes[2].set_title("After Normalize() (-1–1)")
axes[2].axis("off")

plt.show()



# train_dataset = datasets.MNIST(
#     root='./data',
#     train=True,
#     transform=transform,
#     download=True
# )

# test_dataset = datasets.MNIST(
#     root='./data',
#     train=False,
#     transform=transform,
#     download=True
# )

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# images, labels = next(iter(train_loader))
# print(images.shape)
# print(labels.shape)

# plt.figure(figsize=(10, 2))
# for i in range(5):
#     img = images[i].squeeze()  # remove channel dim (1,28,28) → (28,28)
#     plt.subplot(1, 5, i+1)
#     plt.imshow(img, cmap="gray")
#     plt.title(f"Label: {labels[i].item()}")
#     plt.axis("off")
# plt.show()