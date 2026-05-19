import torch
import numpy as np

LARGEPRIME = 2**61 - 1
cache = {}

class CMVec1(object):
    def __init__(self, d, c, r, numBlocks, device=None):
        self.d = d
        self.c = c
        self.r = r
        self.numBlocks = numBlocks
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.table = torch.zeros((r, c), device=self.device)

        self.total_shift = 0.0
        self.total_scale = 0.0
        self.num_updates = 0

        # hash setup
        tokens = torch.arange(d, dtype=torch.int64, device="cpu").reshape((1, d))
        torch.manual_seed(42)
        hashes = torch.randint(0, LARGEPRIME, (r, 2), dtype=torch.int64)
        h1, h2 = hashes[:, 0:1], hashes[:, 1:2]
        self.buckets = ((h1 * tokens) + h2) % LARGEPRIME % self.c
        self.buckets = self.buckets.to(self.device)

        # print(f"CMVec1 initialized: d={self.d}, c={self.c}, r={self.r}")

    def accumulateTable(self, table):
        self.table += table

    def zero(self):
        """ Set all the entries of the sketch to zero """
        self.table.zero_()

    def accumulateVec(self, vec):
        """Shift so min=0, max=min+max, linearly scale others."""
        vec = vec.to(self.device)
        vmin = vec.min().item()
        vmax = vec.max().item()

        shift = -vmin
        denom = vmax - vmin if vmax != vmin else 1e-12
        scale = (vmin + vmax) / denom

        vec_shifted = (vec + shift) * scale

        for r in range(self.r):
            buckets = self.buckets[r, :]
            self.table[r, :] += torch.bincount(buckets, weights=vec_shifted, minlength=self.c)

        # Track averages for unshifting later
        self.total_shift += shift
        self.total_scale += scale
        self.num_updates += 1

    def _findAllValues(self):
        vals = torch.zeros(self.r, self.d, device=self.device)
        for r in range(self.r):
            vals[r] = self.table[r, self.buckets[r, :]]
        est = vals.min(dim=0)[0]

        # Apply inverse shift
        if self.num_updates > 0:
            avg_shift = self.total_shift / self.num_updates
            avg_scale = self.total_scale / self.num_updates
            est = (est / avg_scale) - avg_shift

        return est

    def _findHHK(self, k):
        vals = self._findAllValues()
        outVals = torch.zeros(k, device=vals.device)
        HHs = torch.zeros(k, device=vals.device, dtype=torch.long)
        torch.topk(vals**2, k, sorted=False, out=(outVals, HHs))
        return HHs, vals[HHs]

    def unSketch(self, k=None):
        hhs = self._findHHK(k)
        unSketched = torch.zeros(self.d, device=self.device)
        unSketched[hhs[0]] = hhs[1]

        # Apply inverse transform again for safety
        if self.num_updates > 0:
            avg_shift = self.total_shift / self.num_updates
            avg_scale = self.total_scale / self.num_updates
            unSketched = (unSketched / avg_scale) - avg_shift

        return unSketched
