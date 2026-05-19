import math
import numpy as np
import copy
import torch

LARGEPRIME = 2**61-1

cache = {}

class MN(object):

    def __init__(self, d, c, r, doInitialize=True, device=None,
                 numBlocks=1, use_mn=False, m=2000):
       
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
        
        # MN (Mean Noise) measurement parameters
        self.use_mn = use_mn
        self.m = m  # Number of fake items for noise measurement
        self.measured_noise = 0.0  # Store the measured mean noise

        self.table = torch.zeros((r, c), device=self.device)

        cacheKey = (d, c, r, numBlocks, device)
        if cacheKey in cache:
            self.buckets = cache[cacheKey]["buckets"]
            if self.use_mn:
                self.fake_buckets = cache[cacheKey].get("fake_buckets")
        else:
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

            # Generate fake item buckets for MN if enabled
            if self.use_mn:
                # Use different seed for fake items
                rand_state2 = torch.random.get_rng_state()
                torch.random.manual_seed(42)
                fake_hashes = torch.randint(0, LARGEPRIME, (r, 2),
                                           dtype=torch.int64, device="cpu")
                torch.random.set_rng_state(rand_state2)
                
                # Fake items use indices [d, d+m)
                fake_tokens = torch.arange(d, d + m, dtype=torch.int64, device="cpu")
                fake_tokens = fake_tokens.reshape((1, m))

                #need to delete this line. This is only for testing
                self.fake_tokens = fake_tokens   # <--- ADD THIS LINE

                
                h1_fake = fake_hashes[:,0:1]
                h2_fake = fake_hashes[:,1:2]
                self.fake_buckets = ((h1_fake * fake_tokens) + h2_fake) % LARGEPRIME % self.c
                self.fake_buckets = self.fake_buckets.to(self.device)
                
                cache[cacheKey] = {"buckets": self.buckets, "fake_buckets": self.fake_buckets}
            else:
                cache[cacheKey] = {"buckets": self.buckets}
        
        # if self.use_mn:
        #     print(f"MN initialized with {self.m} fake items for noise measurement: d={self.d}, c={self.c}, r={self.r}")
        # else:
        #     print(f"MN initialized: d={self.d}, c={self.c}, r={self.r}")

    def zero(self):
        """ Set all the entries of the sketch to zero """
        self.table.zero_()
        self.measured_noise = 0.0

    def accumulateTable(self, table):
        self.table += table

    def accumulateVec(self, vec):
        """
        Accumulate the gradient vector into the sketch.
        """
        # Check for NaN or Inf in input
        if torch.isnan(vec).any() or torch.isinf(vec).any():
            print("WARNING: NaN or Inf detected in input vector to MN.accumulateVec")
            return
        
        # Accumulate the vector directly (no shifting)
        for r in range(self.r):
            buckets = self.buckets[r,:].to(self.device)
            self.table[r,:] += torch.bincount(
                                    input=buckets,
                                    weights=vec,
                                    minlength=self.c
                                   )

    def measure_noise(self):
        """
        Measure the mean noise using fake items that were never added to the sketch.
        Since these fake items have actual frequency 0, their estimates are pure noise.
        This implements the MN (d-smallest Mean Noise) measurement from the paper.
        """
        if not self.use_mn:
            print("WARNING: MN measurement not enabled. Set use_mn=True during initialization.")
            return 0.0
        
        if self.fake_buckets is None:
            print("WARNING: Fake buckets not initialized.")
            return 0.0
        
        noise_samples = []
        
        # For each fake item, compute its d-smallest noise
        for i in range(self.m):
            fake_item_noise = []
            for r in range(self.r):
                bucket_idx = self.fake_buckets[r, i]
                # Get the counter value (which is pure noise for fake items)
                counter_value = self.table[r, bucket_idx].item()
                fake_item_noise.append(counter_value)
            
            # Take the minimum (d-smallest) noise for this fake item
            # d_smallest_noise = min(fake_item_noise)
            d_smallest_noise = torch.mean(torch.tensor(fake_item_noise))

            noise_samples.append(d_smallest_noise)
        
        # Compute the mean of all d-smallest noise samples
        self.measured_noise = float(np.mean(noise_samples))
        
        # Check for NaN or unreasonable values
        if np.isnan(self.measured_noise) or np.isinf(self.measured_noise):
            print(f"WARNING: Invalid noise measurement detected: {self.measured_noise}")
            self.measured_noise = 0.0

        self.noise_samples = noise_samples
        
        # print(f"MN measurement: Avg noise = {self.measured_noise:.6f}, "
        #       f"Min = {min(noise_samples):.6f}, Max = {max(noise_samples):.6f}, "
        #       f"Std = {np.std(noise_samples):.6f}")
        
        return self.measured_noise

    def _findHHK(self, k):
        vals = self._findAllValues()

        outVals = torch.zeros(k, device=vals.device)
        HHs = torch.zeros(k, device=vals.device).long()
        torch.topk(vals**2, k, sorted=False, out=(outVals, HHs))
        return HHs, vals[HHs]

    def _findAllValues(self):
        """
        Estimate values for all items using the mean across rows.
        If MN is enabled, automatically measure noise and subtract it.
        """
        vals = torch.zeros(self.r, self.d, device=self.device)
        for r in range(self.r):
            vals[r] = (self.table[r, self.buckets[r,:]])
        
        # Get the estimate (mean across rows)
        est = vals.mean(dim=0)
        # est = vals.min(dim=0)[0]

        
        # Check for NaN before noise measurement
        if torch.isnan(est).any() or torch.isinf(est).any():
            print("WARNING: NaN or Inf detected in estimates before noise removal")
            return est
        
        # Automatically measure and remove noise if MN is enabled
        if self.use_mn:
            # Measure noise automatically (only once per sketch)
            if self.measured_noise == 0.0:
                try:
                    self.measure_noise()
                    # print(f"Measured noise: {self.measured_noise:.6f}")
                except Exception as e:
                    print(f"ERROR in noise measurement: {e}")
                    self.measured_noise = 0.0
            
            # Subtract noise if it's valid (can be positive or negative)
            if self.measured_noise != 0.0 and not np.isnan(self.measured_noise):
                # print(f"Subtracting noise: {self.measured_noise:.6f}")
                est = est - self.measured_noise
            else:
                print(f"Noise NOT subtracted. measured_noise = {self.measured_noise}")
        
        return est
    
    def raw_estimate(self):
        """
        Return the raw CM/MN estimate BEFORE noise removal.
        This is simply the mean across sketch rows.
        No MN subtraction happens here.
        """
        vals = torch.zeros(self.r, self.d, device=self.device)
        for r in range(self.r):
            vals[r] = self.table[r, self.buckets[r, :]]

        # raw = vals.mean(dim=0)

        # # Safety check
        # if torch.isnan(raw).any() or torch.isinf(raw).any():
        #     print("WARNING: NaN or Inf detected in raw estimate")
        
        # return vals.mean(dim=0)
        return vals.median(dim=0)[0]



    def unSketch(self, k=None, epsilon=None):
        """
        Recover the top-k items from the sketch.
        If MN is enabled, the noise will be automatically removed in _findAllValues.
        """
        hhs = self._findHHK(k)
        unSketched = torch.zeros(self.d, device=self.device)
        unSketched[hhs[0]] = hhs[1]
        return unSketched