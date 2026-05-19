import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models import FixupResNet9
from utils import get_grad
from utils import parse_args


# ----------------------------
# 1️⃣  Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# CIFAR10 normalization constants (same as your code)
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

# ----------------------------
# 2️⃣  Load small batch from CIFAR10
# ----------------------------
dataset = torchvision.datasets.CIFAR10(
    root="~/datasets/cifar10/",
    train=True,
    download=True,
    transform=transform
)
loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
images, labels = next(iter(loader))
images, labels = images.to(device), labels.to(device)
print("Batch shapes:", images.shape, labels.shape)

# ----------------------------
# 3️⃣  Model setup
# ----------------------------
model = FixupResNet9().to(device)
criterion = nn.CrossEntropyLoss()

# ----------------------------
# 4️⃣  Forward + Backward
# ----------------------------
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()

# ----------------------------
# 5️⃣  Flatten gradients
# ----------------------------
args = parse_args()


grad = get_grad(model, args)
print("\nGradient vector shape:", grad.shape)
print("First 20 gradient values:\n", grad[:20].cpu())
print("Mean:", grad.mean().item())
print("Std:", grad.std().item())
print("Min:", grad.min().item())
print("Max:", grad.max().item())

# ----------------------------
# 6️⃣  (Optional) Per-layer gradients
# ----------------------------
print("\n--- Per-layer gradient preview ---")
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"{name:30s}  grad mean={p.grad.mean():.5f}, std={p.grad.std():.5f}")
        print("  First few values:", p.grad.view(-1)[:5].detach().cpu().numpy())
        print()

# ----------------------------
# 7️⃣  Save gradient vector for later
# ----------------------------
torch.save(grad.cpu(), "grad_example.pt")
print("\nSaved full flattened gradient vector to grad_example.pt")
