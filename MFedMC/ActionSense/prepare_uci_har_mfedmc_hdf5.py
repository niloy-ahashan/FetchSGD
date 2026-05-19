#!/usr/bin/env python3
"""
Build an ActionSense-compatible HDF5 from the **UCI HAR** raw inertial release so
`dataset.load_and_restructure_hdf5_data` and `main.py` work unchanged.

The six MFedMC modalities are filled from body accelerometer, gyroscope, and total
acceleration (3+3+3 channels), resampled from 128 → 100 timesteps to match the
ActionSense LSTM window length.

Layout (matches `dataset.py` key_map):
  • Eye            (N, 100, 2)       — body_acc x,y
  • EMG-Left       (N, 100, 8)     — body_acc(3)+body_gyro(3)+zeros(2)
  • EMG-Right      (N, 100, 8)     — total_acc(3)+body_gyro(3)+zeros(2)
  • Tactile-Left   (N, 100, 32, 32) — 9 channels laid on a 3×3 patch, rest zeros
  • Tactile-Right  (N, 100, 32, 32) — same 9 channels on a shifted patch (different layout)
  • IMU            (N, 100, 22, 3) — 9 inertial channels + zeros to 66 features

Labels: 6 activities (1-based indexes in HDF5; `dataset.py` subtracts 1).

Requires the inner UCI folder that contains train/ and test/ with Inertial Signals/.

Example:
  python prepare_uci_har_mfedmc_hdf5.py \\
    --uci_root \"../../human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset\" \\
    --output ./data/uci_har_mfedmc.hdf5
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F

UCI_ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def _stack_signals(sig_dir: str, split: str, prefix: str) -> np.ndarray:
    xs = []
    for axis in ("x", "y", "z"):
        path = os.path.join(sig_dir, f"{prefix}_{axis}_{split}.txt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {path}")
        xs.append(np.loadtxt(path))
    stacked = np.stack(xs, axis=-1).astype(np.float32)
    return stacked


def resample_128_to_100(x: np.ndarray) -> np.ndarray:
    """Linear resample along time: (N, 128, C) -> (N, 100, C)."""
    if x.shape[1] == 100:
        return x.astype(np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected (N,128,C), got {x.shape}")
    t = torch.from_numpy(x).float().transpose(1, 2)
    t = F.interpolate(t, size=100, mode="linear", align_corners=True)
    return t.transpose(1, 2).numpy().astype(np.float32)


def build_small_modalities_and_ch9(
    body_acc: np.ndarray,
    body_gyro: np.ndarray,
    total_acc: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Eye, EMG L/R, IMU plus ch9 (N,T,9) for tactile (written in chunks to limit RAM)."""
    N, T, _ = body_acc.shape
    eye = np.stack([body_acc[:, :, 0], body_acc[:, :, 1]], axis=-1).astype(np.float32)

    z2 = np.zeros((N, T, 2), dtype=np.float32)
    emg_left = np.concatenate([body_acc, body_gyro, z2], axis=-1)
    emg_right = np.concatenate([total_acc, body_gyro, z2], axis=-1)

    ch9 = np.concatenate([body_acc, body_gyro, total_acc], axis=-1)

    flat66 = np.zeros((N, T, 66), dtype=np.float32)
    flat66[:, :, :9] = ch9
    imu = flat66.reshape(N, T, 22, 3)

    return (
        {
            "Eye": eye,
            "EMG-Left": emg_left,
            "EMG-Right": emg_right,
            "IMU": imu,
        },
        ch9,
    )


def tactile_pair_from_ch9(ch9_blk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ch9_blk (b,T,9) -> tactile left/right (b,T,32,32)."""
    b, T, _ = ch9_blk.shape
    tl = np.zeros((b, T, 32, 32), dtype=np.float32)
    tr = np.zeros((b, T, 32, 32), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            c = i * 3 + j
            tl[:, :, i, j] = ch9_blk[:, :, c]
            tr[:, :, i + 4, j + 4] = ch9_blk[:, :, c]
    return tl, tr


def inverse_key_to_hdf5_name(mapped: str) -> str:
    """dataset.py key_map inverse: modality name -> example_matrices_* suffix parts."""
    inv = {
        "Eye": ("eye-tracking-gaze", "position"),
        "EMG-Left": ("myo-left", "emg"),
        "EMG-Right": ("myo-right", "emg"),
        "Tactile-Left": ("tactile-glove-left", "tactile", "data"),
        "Tactile-Right": ("tactile-glove-right", "tactile", "data"),
        "IMU": ("xsens-joints", "rotation", "xzy", "deg"),
    }
    parts = inv[mapped]
    return "example_matrices_" + "_".join(parts)


def load_split(
    root: str, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sig = os.path.join(root, split, "Inertial Signals")
    body_acc = _stack_signals(sig, split, "body_acc")
    body_gyro = _stack_signals(sig, split, "body_gyro")
    total_acc = _stack_signals(sig, split, "total_acc")
    y_path = os.path.join(root, split, f"y_{split}.txt")
    s_path = os.path.join(root, split, f"subject_{split}.txt")
    y = np.loadtxt(y_path).astype(np.int64).ravel()
    subj = np.loadtxt(s_path).astype(np.int64).ravel()
    return body_acc, body_gyro, total_acc, y, subj


def main() -> None:
    p = argparse.ArgumentParser(description="UCI HAR → MFedMC ActionSense HDF5")
    p.add_argument(
        "--uci_root",
        type=str,
        required=True,
        help="Inner UCI HAR folder (contains train/ and test/ with Inertial Signals/).",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .hdf5 path (e.g. MFedMC/ActionSense/data/uci_har_mfedmc.hdf5).",
    )
    p.add_argument(
        "--chunk_rows",
        type=int,
        default=512,
        help="HDF5 write chunk size for large tactile arrays (avoids multi-GB RAM).",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on N after merging train+test (for quick tests).",
    )
    args = p.parse_args()

    root = os.path.abspath(args.uci_root)
    for split in ("train", "test"):
        d = os.path.join(root, split, "Inertial Signals")
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing {d}")

    ba_tr, bg_tr, ta_tr, y_tr, s_tr = load_split(root, "train")
    ba_te, bg_te, ta_te, y_te, s_te = load_split(root, "test")

    body_acc = np.concatenate([ba_tr, ba_te], axis=0)
    body_gyro = np.concatenate([bg_tr, bg_te], axis=0)
    total_acc = np.concatenate([ta_tr, ta_te], axis=0)
    y = np.concatenate([y_tr, y_te], axis=0)
    subj = np.concatenate([s_tr, s_te], axis=0)

    if args.max_samples is not None:
        m = min(int(args.max_samples), body_acc.shape[0])
        body_acc = body_acc[:m]
        body_gyro = body_gyro[:m]
        total_acc = total_acc[:m]
        y = y[:m]
        subj = subj[:m]

    body_acc = resample_128_to_100(body_acc)
    body_gyro = resample_128_to_100(body_gyro)
    total_acc = resample_128_to_100(total_acc)

    small, ch9 = build_small_modalities_and_ch9(body_acc, body_gyro, total_acc)
    N, T = body_acc.shape[0], body_acc.shape[1]
    del body_acc, body_gyro, total_acc

    label_idx_1based = y.astype(np.int32)
    label_strings = np.array(
        [UCI_ACTIVITY_NAMES[int(c) - 1] for c in label_idx_1based],
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    subj_str = np.array(
        [f"U{int(s):02d}" for s in subj],
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    chunk = max(1, int(args.chunk_rows))
    key_tl = inverse_key_to_hdf5_name("Tactile-Left")
    key_tr = inverse_key_to_hdf5_name("Tactile-Right")

    with h5py.File(args.output, "w") as h5f:
        h5f.create_dataset("example_label_indexes", data=label_idx_1based)
        h5f.create_dataset("example_labels", data=label_strings)
        h5f.create_dataset("example_subject_ids", data=subj_str)
        for name, arr in small.items():
            h5f.create_dataset(inverse_key_to_hdf5_name(name), data=arr)

        ds_tl = h5f.create_dataset(
            key_tl,
            shape=(N, T, 32, 32),
            dtype=np.float32,
            chunks=(min(chunk, N), T, 32, 32),
        )
        ds_tr = h5f.create_dataset(
            key_tr,
            shape=(N, T, 32, 32),
            dtype=np.float32,
            chunks=(min(chunk, N), T, 32, 32),
        )
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            blk = ch9[start:end]
            tl_b, tr_b = tactile_pair_from_ch9(blk)
            ds_tl[start:end] = tl_b
            ds_tr[start:end] = tr_b

    print(f"Wrote {os.path.abspath(args.output)}")
    print(
        f"  N={N}, T={T}, num_classes=6, modalities: "
        f"{list(small.keys())} + Tactile-Left + Tactile-Right"
    )
    print(f"  Unique subjects: {len(np.unique(subj))}")


if __name__ == "__main__":
    main()
