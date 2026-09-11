"""
FetchSGD Count Sketch (CSVecFed) + virtual-momentum server step.

Copied into this folder so the hybrid does not modify CommEfficient.
The algorithm matches ``csvecN.CSVecFed`` and
``fed_aggregator._server_helper_sketched``.
"""

from __future__ import annotations

import torch

LARGEPRIME = 2**61 - 1
_cache = {}


class CSVecFed:
    def __init__(self, d, c, r, device=None, numBlocks=1):
        if device is None:
            device = (
                torch.device(torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.r = int(r)
        self.c = int(c)
        self.d = int(d)
        self.numBlocks = numBlocks
        self.device = device
        self.table = torch.zeros((self.r, self.c), device=self.device)

        cache_key = (self.d, self.c, self.r, numBlocks, str(device))
        if cache_key in _cache:
            self.signs = _cache[cache_key]["signs"]
            self.buckets = _cache[cache_key]["buckets"]
            return

        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        hashes = torch.randint(0, LARGEPRIME, (self.r, 4), dtype=torch.int64, device="cpu")
        torch.random.set_rng_state(rand_state)

        tokens = torch.arange(self.d, dtype=torch.int64, device="cpu").reshape(1, self.d)
        h1 = hashes[:, 2:3]
        h2 = hashes[:, 3:4]
        signs = ((h1 * tokens + h2) % LARGEPRIME % 2) * 2 - 1
        self.signs = signs.float().to(self.device)

        h1 = hashes[:, 0:1]
        h2 = hashes[:, 1:2]
        self.buckets = ((h1 * tokens + h2) % LARGEPRIME % self.c).to(self.device)

        _cache[cache_key] = {"signs": self.signs, "buckets": self.buckets}

    def zero(self):
        self.table.zero_()

    def accumulateTable(self, table):
        self.table += table

    def accumulateVec(self, vec):
        vec = vec.view(-1).to(self.device)
        for r in range(self.r):
            buckets = self.buckets[r, :].to(self.device)
            signs = self.signs[r, :].to(self.device)
            self.table[r, :] += torch.bincount(
                input=buckets, weights=signs * vec, minlength=self.c
            )

    def _find_all_values(self):
        vals = torch.zeros(self.r, self.d, device=self.device)
        for r in range(self.r):
            vals[r] = self.table[r, self.buckets[r, :]] * self.signs[r, :]
        return vals.median(dim=0)[0]

    def unSketch(self, k=None):
        vals = self._find_all_values()
        k = int(self.d if k is None else min(k, self.d))
        out_vals = torch.zeros(k, device=vals.device)
        hhs = torch.zeros(k, device=vals.device).long()
        torch.topk(vals**2, k, sorted=False, out=(out_vals, hhs))
        unsketched = torch.zeros(self.d, device=self.device)
        unsketched[hhs] = vals[hhs]
        return unsketched


def sketch_vector(vec, d, num_rows, num_cols, device):
    sketch = CSVecFed(d=d, c=num_cols, r=num_rows, device=device)
    sketch.accumulateVec(vec)
    return sketch.table.clone()


def fetchsgd_server_step(sketched_grad, Vvelocity, Verror, grad_size, num_rows, num_cols, k, rho, lr, device, error_type="virtual"):
    """One FetchSGD parameter-server step on an aggregated sketch table."""
    torch.add(sketched_grad, Vvelocity, alpha=rho, out=Vvelocity)
    if error_type == "virtual":
        Verror = Verror + Vvelocity
        table = Verror
    else:
        table = Vvelocity

    sketch = CSVecFed(d=grad_size, c=num_cols, r=num_rows, device=device)
    sketch.accumulateTable(table)
    update = sketch.unSketch(k=k)
    sketch.zero()
    sketch.accumulateVec(update)
    sketched_update = sketch.table
    nz = sketched_update.nonzero()
    if error_type == "virtual" and nz.numel() > 0:
        Verror[nz[:, 0], nz[:, 1]] = 0
    if nz.numel() > 0:
        Vvelocity[nz[:, 0], nz[:, 1]] = 0
    return update * lr, Vvelocity, Verror
