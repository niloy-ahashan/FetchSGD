import torch
from CommEfficient.cmvec import CMVec
from CommEfficient.mn import MN

# --------------------------
# Real vector (ground truth)
# --------------------------
# vec = torch.tensor([1.0, -2.0, 3.0, 0.5, -1.5], dtype=torch.float32)

vec = torch.linspace(-0.1, 0.2, steps=200).float()


# vec = torch.linspace(-5, 5, steps=20).float()


d = 200
c = 10
r = 3

# --------------------------
# Initialize CMVec sketch
# --------------------------
cm = CMVec(d=d, c=c, r=r, device='cpu')
cm.zero()
cm.accumulateVec(vec)

print("\n===== CMVec (NO MN) =====")
print("CMVec sketch table:")
print(cm.table)

print("\nCMBuckets buckets:")
print(cm.buckets)

cm_est = cm._findAllValues()
print("\nCMVec estimate (no noise removal):")
print(cm_est)

# --------------------------
# Initialize MN sketch
# --------------------------
mn = MN(d=d, c=c, r=r, device='cpu', use_mn=True, m=5)
mn.zero()
mn.accumulateVec(vec)

print("\n===== MN (WITH MN) =====")
print("MN sketch table:")
print(mn.table)

print("\nMNBuckets buckets:")
print(mn.buckets)

print("Real vector values:")
print(vec.tolist())   # prints clean Python list

#for testing different range of fake items, override the generated fake tokens
# fake_start = int(vec.min().item()) + 1
# mn.fake_tokens = torch.arange(fake_start, fake_start + mn.m).reshape(1, mn.m)

# print("\nFake items (overridden):", mn.fake_tokens)

print("Fake item IDs:")
print(mn.fake_tokens.tolist())


print("\nFake buckets:")
print(mn.fake_buckets)

# Raw estimate BEFORE MN correction
mn_raw = mn.raw_estimate()
print("\nMN raw estimate BEFORE noise removal:")
print(mn_raw)

# Corrected estimate AFTER MN correction
mn_corrected = mn._findAllValues()
print("\nMN corrected estimate AFTER noise removal:")
print(mn_corrected)

print("MN noise samples:", [float(x) for x in mn.noise_samples])

print(f"\nMeasured MN noise: {mn.measured_noise}\n")



# ----------------------------------------
# Error comparison function
# ----------------------------------------
def compare_errors(real, cm_est, mn_raw, mn_corr):
    # Header with aligned columns
    header = (
        f"{'Idx':>4} | "
        f"{'Real':>8} | "
        f"{'CMVec':>10} | "
        f"{'MN_raw':>10} | "
        f"{'MN_corr':>12} | "
        f"{'Err_CM':>10} | "
        f"{'Err_raw':>10} | "
        f"{'Err_corr':>10} | "
        f"{'Best':>12}"
    )
    # print("=" * len(header))
    # print(header)
    # print("=" * len(header))

    count_cm = 0
    count_raw = 0
    count_corr = 0

    sum_err_cm = 0.0
    sum_err_raw = 0.0
    sum_err_corr = 0.0

    d = len(real)

    for i in range(d):
        rv = real[i].item()
        cmv = cm_est[i].item()
        mnr = mn_raw[i].item()
        mnc = mn_corr[i].item()

        err_cm = abs(cmv - rv)
        err_raw = abs(mnr - rv)
        err_corr = abs(mnc - rv)

        sum_err_cm += err_cm
        sum_err_raw += err_raw
        sum_err_corr += err_corr

        # Determine best method
        if err_cm <= err_raw and err_cm <= err_corr:
            best = "CMVec"
            count_cm += 1
        elif err_raw <= err_cm and err_raw <= err_corr:
            best = "MN_raw"
            count_raw += 1
        else:
            best = "MN_corr"
            count_corr += 1

        # Print aligned row
        # print(
        #     f"{i:>4} | "
        #     f"{rv:>8.4f} | "
        #     f"{cmv:>10.4f} | "
        #     f"{mnr:>10.4f} | "
        #     f"{mnc:>12.4f} | "
        #     f"{err_cm:>10.4f} | "
        #     f"{err_raw:>10.4f} | "
        #     f"{err_corr:>10.4f} | "
        #     f"{best:>12}"
        # )

    # Compute average errors
    avg_err_cm = sum_err_cm / d
    avg_err_raw = sum_err_raw / d
    avg_err_corr = sum_err_corr / d

    # Create list of (method_name, avg_error_value)
    avg_list = [
        ("CMVec", avg_err_cm),
        ("MN_raw", avg_err_raw),
        ("MN_corr", avg_err_corr)
    ]

    # Sort by error (ascending: lowest = best)
    avg_sorted = sorted(avg_list, key=lambda x: x[1])

    print("=" * len(header))
    print(f"Total dimensions (d): {d}")
    print(f"CMVec best count:     {count_cm}")
    print(f"MN_raw best count:    {count_raw}")
    print(f"MN_corr best count:   {count_corr}\n")
    print("----- Average absolute error -----")
    # print(f"Avg CMVec error:      {avg_err_cm:.6f}")
    # print(f"Avg MN_raw error:     {avg_err_raw:.6f}")
    # print(f"Avg MN_corr error:    {avg_err_corr:.6f}")
    print("\n----- Average error ranking (best to worst) -----")
    for name, val in avg_sorted:
        print(f"{name:>10}: {val:.6f}")

    print("=" * len(header))



# ----------------------------------------
# Print comparison
# ----------------------------------------
compare_errors(vec, cm_est, mn_raw, mn_corrected)
