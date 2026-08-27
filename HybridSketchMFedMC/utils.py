from collections import namedtuple
import time

import numpy as np
import torch


class PiecewiseLinear(namedtuple("PiecewiseLinear", ("knots", "vals"))):
    def __call__(self, t):
        knots = np.array(self.knots, dtype=float)
        vals = np.array(self.vals, dtype=float)
        return np.interp([t], knots, vals)[0]


class Timer:
    def __init__(self):
        self.times = [time.time()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.times.append(time.time())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t


class TableLogger:
    """Same console table as CommEfficient ``mm_train.py`` / SketchFusionB."""

    def append(self, output):
        if not hasattr(self, "keys"):
            self.keys = output.keys()
            print(*("{:>12s}".format(k) for k in self.keys))
        filtered = [output[k] for k in self.keys]
        print(
            *(
                "{:12.4f}".format(v)
                if isinstance(v, (float, np.float32, np.float64))
                else "{:12}".format(v)
                for v in filtered
            )
        )


def get_param_vec(model):
    return torch.cat(
        [p.data.view(-1).float() for p in model.parameters() if p.requires_grad]
    )


def set_param_vec(model, param_vec):
    start = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        end = start + p.numel()
        p.data.copy_(param_vec[start:end].view_as(p.data))
        start = end


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_branch_parameters(model, indices):
    return int(indices.numel())


def normalization(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
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
