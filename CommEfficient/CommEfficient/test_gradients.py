import torch
import torchvision
import torchvision.transforms as transforms
from utils import get_grad, parse_args
import models

# Parse arguments (for device etc.)
args = parse_args()
args.device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Step 1: Create model ----
model_cls = getattr(models, args.model)  # e.g., FixupResNet9
model = model_cls().to(args.device)

# ---- Step 2: Prepare a dummy batch ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2471, 0.2435, 0.2616))
])

trainset = torchvision.datasets.CIFAR10(root=args.dataset_dir,
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=8,
                                          shuffle=True)

images, labels = next(iter(trainloader))
images, labels = images.to(args.device), labels.to(args.device)

# ---- Step 3: Forward + Backward ----
criterion = torch.nn.CrossEntropyLoss()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()

# ---- Step 4: Get gradients ----
grad_vec = get_grad(model, args)

print("Gradient vector shape:", grad_vec.shape)
print("Gradient min:", grad_vec.min().item())
print("Gradient max:", grad_vec.max().item())
print("Gradient mean:", grad_vec.mean().item())
print("Gradient std:", grad_vec.std().item())
