import numpy as np
import torch


def average_weights(weights_list, chosen_weight_indices, global_model, sample_counts=None):
    """FedAvg-style aggregation. With `sample_counts` (one entry per position in
    `weights_list`, i.e. the same indexing as `chosen_weight_indices`), this is the
    paper's Eq. (21): each selected client's weights are scaled by
    |D_k^m| / sum(|D_k^m|) instead of averaged uniformly. Falls back to an equal-weight
    average (identical to the old plain `torch.mean`) if `sample_counts` is omitted,
    mismatched in length, or sums to <= 0.
    """
    valid_idx = [idx for idx in chosen_weight_indices if 0 <= idx < len(weights_list)]
    model_weights = [weights_list[idx] for idx in valid_idx]
    if not model_weights:
        return global_model.state_dict(), 0

    if sample_counts is not None and len(sample_counts) == len(weights_list):
        counts = np.array([sample_counts[idx] for idx in valid_idx], dtype=np.float64)
    else:
        counts = np.array([], dtype=np.float64)
    total = counts.sum() if counts.size == len(model_weights) else 0.0
    if total > 0:
        w = counts / total
    else:
        w = np.full(len(model_weights), 1.0 / len(model_weights))

    device = next(global_model.parameters()).device
    avg_weights = {}
    for key in model_weights[0].keys():
        stacked = torch.stack([weights[key].to(device) for weights in model_weights])
        w_t = torch.tensor(w, dtype=stacked.dtype, device=device).view(
            -1, *([1] * (stacked.dim() - 1))
        )
        avg_weights[key] = (stacked * w_t).sum(dim=0)
    return avg_weights, len(model_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalization(arr):
    min_vals = np.min(arr, axis=1, keepdims=True)
    max_vals = np.max(arr, axis=1, keepdims=True)
    return (arr - min_vals) / (max_vals - min_vals + 1e-10)


def normalize_1d(arr):
    x = np.array(arr, dtype=float)
    min_v = np.nanmin(x)
    max_v = np.nanmax(x)
    return (x - min_v) / (max_v - min_v + 1e-10)


def get_aligned_shap_values(shap_values, valid_modalities, all_modalities):
    aligned = np.zeros(len(all_modalities))
    for idx, modality in enumerate(all_modalities):
        if modality in valid_modalities:
            aligned[idx] = shap_values[valid_modalities.index(modality)]
    return aligned
