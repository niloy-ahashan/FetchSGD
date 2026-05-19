import torch
import torch.nn as nn
import sys
sys.path.append('.')

from models import FixupResNet9
from utils import get_grad

# Create model
model = FixupResNet9(
    channels={'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512},
    num_classes=10
)

print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")

# Create dummy batch
batch_size = 32
images = torch.randn(batch_size, 3, 32, 32)  # Random CIFAR-10-like images
labels = torch.randint(0, 10, (batch_size,))  # Random labels

# Forward pass
model.train()
outputs = model(images)
loss = nn.CrossEntropyLoss()(outputs, labels)

# Backward pass
model.zero_grad()
loss.backward()

# Get gradient vector
class Args:
    device = 'cpu'
    weight_decay = 0.0005
    num_workers = 1

args = Args()
grad = get_grad(model, args)

# Analyze
print("\n" + "="*70)
print("GRADIENT SIGN ANALYSIS")
print("="*70)

total = grad.numel()
num_pos = (grad > 0).sum().item()
num_neg = (grad < 0).sum().item()
num_zero = (grad == 0).sum().item()

print(f"\nTotal parameters: {total:,}")
print(f"Positive gradients: {num_pos:,} ({100*num_pos/total:.2f}%)")
print(f"Negative gradients: {num_neg:,} ({100*num_neg/total:.2f}%)")
print(f"Zero gradients: {num_zero:,} ({100*num_zero/total:.2f}%)")

print(f"\nStatistics:")
print(f"  Min: {grad.min().item():.6f}")
print(f"  Max: {grad.max().item():.6f}")
print(f"  Mean: {grad.mean().item():.6f}")
print(f"  Std: {grad.std().item():.6f}")
print(f"  Median: {grad.median().item():.6f}")

# Check top-k gradients
k = 10000
abs_grad = grad.abs()
topk_vals, topk_idx = torch.topk(abs_grad, k)
topk_original = grad[topk_idx]
topk_pos = (topk_original > 0).sum().item()
topk_neg = (topk_original < 0).sum().item()

print(f"\nTop-{k} gradients by magnitude:")
print(f"  Positive: {topk_pos:,} ({100*topk_pos/k:.2f}%)")
print(f"  Negative: {topk_neg:,} ({100*topk_neg/k:.2f}%)")
print(f"  Range: [{topk_original.min().item():.6f}, {topk_original.max().item():.6f}]")

print("\n" + "="*70)
print("CONCLUSION:")
if num_neg > 0:
    print("✓ Gradients contain NEGATIVE values!")
    print("✓ Count Sketch (CSVec) is the correct choice.")
    print("✗ Count-Min (CMVec) will NOT work correctly.")
else:
    print("? All gradients are non-negative (unusual!)")
print("="*70)