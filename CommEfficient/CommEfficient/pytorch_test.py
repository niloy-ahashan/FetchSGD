import torch

x = torch.tensor([1.0, 9.0])
print(x.median())  # returns 1.0 (not 5.0)
print(x.mean())    # returns 5.0
