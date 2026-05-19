import math
import numpy as np
import copy
import torch

LARGEPRIME = 2**61-1
cache = {}

class CMVec2(object):
    """
    Count-Min Sketch for signed values (gradients in federated learning).
    
    Since standard Count-Min only works for non-negative values,
    we split each value into positive and negative parts:
    - table_pos: stores positive parts
    - table_neg: stores negative parts (as positive values)
    - Estimate = min(table_pos) - min(table_neg)
    """
    
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
        
        # Two tables: one for positive parts, one for negative parts
        self.table_pos = torch.zeros((r, c), device=self.device)
        self.table_neg = torch.zeros((r, c), device=self.device)
        
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
        
        # Bucket hash (2-wise independent)
        h1 = hashes[:,0:1]
        h2 = hashes[:,1:2]
        self.buckets = ((h1 * tokens) + h2) % LARGEPRIME % self.c
        self.buckets = self.buckets.to(self.device)
        
        cache[cacheKey] = {"buckets": self.buckets}
        
        print(f"CMVec2 initialized: d={self.d}, c={self.c}, r={self.r}")
    
    @property
    def table(self):
        """
        Return table as a tuple for compatibility with existing code.
        This allows code like: sketch.table to work.
        """
        return (self.table_pos, self.table_neg)
    
    @table.setter
    def table(self, value):
        """
        Set table from a tuple or single tensor.
        """
        if isinstance(value, tuple):
            self.table_pos, self.table_neg = value
        else:
            # If single tensor provided, assume it's old format - split it
            raise ValueError("CMVec requires tuple (table_pos, table_neg)")
    
    def zero(self):
        """Set all entries of the sketch to zero"""
        self.table_pos.zero_()
        self.table_neg.zero_()
    
    def accumulateTable(self, table_input):
        """
        Accumulate another sketch table (for federated aggregation).
        
        Args:
            table_input: Can be:
                - CMVec object
                - Tuple (table_pos, table_neg)
                - Single tensor (will be treated as another CMVec's table property)
        """
        if isinstance(table_input, CMVec2):
            # If passed another CMVec object
            self.table_pos += table_input.table_pos
            self.table_neg += table_input.table_neg
        elif isinstance(table_input, tuple) and len(table_input) == 2:
            # If passed as tuple (table_pos, table_neg)
            self.table_pos += table_input[0]
            self.table_neg += table_input[1]
        else:
            raise ValueError(
                f"accumulateTable expects CMVec object or tuple (pos, neg), got {type(table_input)}"
            )
    
    def accumulateVec(self, vec):
        """
        Sketch a vector into the Count-Min sketch.
        Splits vector into positive and negative parts.
        
        Args:
            vec: 1D tensor of shape (d,)
        """
        # Split vector into positive and negative parts
        vec_pos = torch.clamp(vec, min=0)  # max(vec, 0)
        vec_neg = torch.clamp(vec, max=0).abs()  # max(-vec, 0)
        
        for r in range(self.r):
            buckets = self.buckets[r,:].to(self.device)
            
            # Accumulate positive parts
            self.table_pos[r,:] += torch.bincount(
                input=buckets,
                weights=vec_pos,
                minlength=self.c
            )
            
            # Accumulate negative parts
            self.table_neg[r,:] += torch.bincount(
                input=buckets,
                weights=vec_neg,
                minlength=self.c
            )
    
    def _findAllValues(self):
        """
        Estimate all coordinates using min across rows (Count-Min principle).
        
        Estimate = min(positive) - min(negative)
        """
        vals_pos = torch.zeros(self.r, self.d, device=self.device)
        vals_neg = torch.zeros(self.r, self.d, device=self.device)
        
        for r in range(self.r):
            vals_pos[r] = self.table_pos[r, self.buckets[r,:]]
            vals_neg[r] = self.table_neg[r, self.buckets[r,:]]
        
        # Take minimum across rows for each part
        min_pos = vals_pos.min(dim=0)[0]
        min_neg = vals_neg.min(dim=0)[0]
        
        # Reconstruct signed estimate
        return min_pos - min_neg
    
    def _findHHK(self, k):
        """Find top-k heavy hitters by magnitude"""
        vals = self._findAllValues()
        outVals = torch.zeros(k, device=vals.device)
        HHs = torch.zeros(k, device=vals.device).long()
        torch.topk(vals.abs(), k, sorted=False, out=(outVals, HHs))
        return HHs, vals[HHs]
    
    def unSketch(self, k=None, epsilon=None):
        """
        Recover heavy hitters from the sketch.
        
        Args:
            k: number of top elements to recover (by magnitude)
            epsilon: threshold as fraction of L1 norm (not implemented)
        
        Returns:
            Recovered sparse vector
        """
        hhs = self._findHHK(k)
        unSketched = torch.zeros(self.d, device=self.device)
        unSketched[hhs[0]] = hhs[1]
        return unSketched
    
    def __add__(self, other):
        """Add two sketches (creates new sketch)"""
        assert self.d == other.d and self.c == other.c and self.r == other.r
        result = copy.deepcopy(self)
        result.table_pos += other.table_pos
        result.table_neg += other.table_neg
        return result
    
    def __iadd__(self, other):
        """In-place addition of sketches"""
        assert self.d == other.d and self.c == other.c and self.r == other.r
        self.table_pos += other.table_pos
        self.table_neg += other.table_neg
        return self


# # Test code
# if __name__ == "__main__":
#     # Test signed CM sketch
#     d, c, r = 10000, 1024, 4
#     cm = CMVec(d=d, c=c, r=r, device='cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Create a vector with positive and negative values (like gradients)
#     vec = torch.randn(d, device=cm.device)
#     vec[100] = 10.0
#     vec[500] = -5.0
#     vec[1000] = 8.0
    
#     # Sketch it
#     cm.accumulateVec(vec)
    
#     # Query specific values
#     all_vals = cm._findAllValues()
#     print("Original vs Estimated:")
#     print(f"  vec[100] = {vec[100].item():.2f}, estimate = {all_vals[100].item():.2f}")
#     print(f"  vec[500] = {vec[500].item():.2f}, estimate = {all_vals[500].item():.2f}")
#     print(f"  vec[1000] = {vec[1000].item():.2f}, estimate = {all_vals[1000].item():.2f}")
#     print(f"  vec[999] = {vec[999].item():.2f}, estimate = {all_vals[999].item():.2f}")
    
#     # Recover top-3 elements
#     recovered = cm.unSketch(k=3)
#     print("\nRecovered top-3 by magnitude:")
#     nz = torch.nonzero(recovered).squeeze()
#     for idx in nz:
#         print(f"  recovered[{idx.item()}] = {recovered[idx].item():.2f} (true: {vec[idx].item():.2f})")
    
#     # Test federated aggregation
#     print("\n--- Federated Learning Test ---")
#     cm1 = CMVec(d=d, c=c, r=r, device=cm.device)
#     cm2 = CMVec(d=d, c=c, r=r, device=cm.device)
    
#     vec1 = torch.randn(d, device=cm.device)
#     vec2 = torch.randn(d, device=cm.device)
    
#     cm1.accumulateVec(vec1)
#     cm2.accumulateVec(vec2)
    
#     # Aggregate sketches
#     aggregated = cm1 + cm2
    
#     print(f"Sketch1 estimate L1: {cm1._findAllValues().abs().sum().item():.2f}")
#     print(f"Sketch2 estimate L1: {cm2._findAllValues().abs().sum().item():.2f}")
#     print(f"Aggregated estimate L1: {aggregated._findAllValues().abs().sum().item():.2f}")
#     print(f"True (vec1+vec2) L1: {(vec1+vec2).abs().sum().item():.2f}")