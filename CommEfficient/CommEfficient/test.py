import torch
from csvec import CSVec
from csvecN import CSVecFed

d, r, c = 10, 3, 5
cs = CSVec(d, c, r, numBlocks=1, device='cpu')
csN = CSVecFed(d, c, r, numBlocks=1, device='cpu')

vec = torch.tensor([5.0, -3.0, 0.0, 2.0, -1.0, 4.0, 0.0, 0.0, -2.0, 1.0])


cs.accumulateVec(vec)
csN.accumulateVec(vec)

k = 5

print("Sketch table:\n", cs.table)
rec = cs.unSketch(k)
print("Recovered estimate:\n", rec)

print("Federated Sketch table:\n", csN.table)
recN = csN.unSketch(k)
print("Recovered estimate (Federated):\n", recN)
