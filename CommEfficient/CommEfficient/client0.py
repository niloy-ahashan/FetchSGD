import numpy as np

# Load the file
data = np.load("/homes/ani283/datasets/cifar10/client0.npy")

print("Shape:", data.shape)
print("Data type:", data.dtype)
print("Min pixel value:", data.min())
print("Max pixel value:", data.max())

# Show the first image as an array
print("First image array:\n", data[0])

# Optional: visualize with matplotlib
import matplotlib.pyplot as plt
plt.imshow(data[0])
plt.title("Client 0 - First Image (Airplane)")
plt.savefig("client0_first_image.png")
print("Saved image as client0_first_image.png")

