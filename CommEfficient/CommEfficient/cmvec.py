import math
import numpy as np
import copy
import torch

LARGEPRIME = 2**61-1

cache = {}

class CMVec(object):

    def __init__(self, d, c, r, doInitialize=True, device=None,
                 numBlocks=1):
       
        global cache

        if device is None:
            device = (
                torch.device(torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        self.r = r 
        self.c = c 
        self.d = int(d) 
        self.numBlocks = numBlocks
        self.device = device

        self.table = torch.zeros((r, c), device=self.device)

        cacheKey = (d, c, r, numBlocks, device)
        if cacheKey in cache:
            self.buckets = cache[cacheKey]["buckets"]
            return

        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        hashes = torch.randint(0, LARGEPRIME, (r, 2),
                               dtype=torch.int64, device="cpu")

        torch.random.set_rng_state(rand_state)

        tokens = torch.arange(d, dtype=torch.int64, device="cpu")
        tokens = tokens.reshape((1, d))

        h1 = hashes[:,0:1]
        h2 = hashes[:,1:2]
        self.buckets = ((h1 * tokens) + h2) % LARGEPRIME % self.c

  
        self.buckets = self.buckets.to(self.device)

        cache[cacheKey] = {"buckets": self.buckets}
                        
        
        print(f"CMVecUpdates initialized: d={self.d}, c={self.c}, r={self.r}")


    def zero(self):
        """ Set all the entries of the sketch to zero """
        self.table.zero_()

    def accumulateTable(self, table):
        self.table += table

    def accumulateVec(self, vec):
        for r in range(self.r):
            buckets = self.buckets[r,:].to(self.device)
            self.table[r,:] += torch.bincount(
                                    input=buckets,
                                    weights=vec,
                                    minlength=self.c
                                   )

    def _findHHK(self, k):
        vals = self._findAllValues()
        # print("vals shape:", vals.shape)

        outVals = torch.zeros(k, device=vals.device)
        HHs = torch.zeros(k, device=vals.device).long()
        torch.topk(vals**2, k, sorted=False, out=(outVals, HHs))
        return HHs, vals[HHs]

    
    def _findAllValues(self):
        vals = torch.zeros(self.r, self.d, device=self.device)
        for r in range(self.r):
            vals[r] = (self.table[r, self.buckets[r,:]])

        # mean = torch.sum(self.table[0]) / self.c

        # return vals.min(dim=0)[0]
        return vals.mean(dim=0)
        # return vals.median(dim=0)[0]-mean
        # return vals.median(dim=0)[0]
        #  # CM uses minimum over rows
        # est = vals.min(dim=0)[0]

        # # Undo the shift that was applied at encoding
        # return est - self.last_shift
    

    def unSketch(self, k=None, epsilon=None):
        hhs = self._findHHK(k)
        unSketched = torch.zeros(self.d, device=self.device)
        unSketched[hhs[0]] = hhs[1]
        return unSketched