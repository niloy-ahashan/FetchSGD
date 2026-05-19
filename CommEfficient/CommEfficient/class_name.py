import numpy as np
from torchvision.datasets import CIFAR10

# load CIFAR-10 class names
classes = CIFAR10(root="~/datasets/cifar10", train=True).classes

data = np.load("/homes/ani283/datasets/cifar10/client8.npy")
print("client8.npy shape:", data.shape)
print("This client corresponds to class:", classes[8])
