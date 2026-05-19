#!/usr/bin/env python3
"""
Build FedMultiModal `data.npz` from UCI HAR **raw inertial signals** for MFedMC-style
time×features encoders (2401.16685v2): each modality is (time × channels).

Modalities (two streams, consistent with acc vs gyro separation):
  • img_* — body accelerometer x, y, z  → shape (N, 128, 3)
  • txt_* — body gyroscope x, y, z      → shape (N, 128, 3)

128 = window length in the UCI HAR release; 3 = per-timestep feature dimension.

Requires the original UCI HAR folder layout, e.g.:
  <uci_root>/train/Inertial Signals/body_acc_x_train.txt
  <uci_root>/test/Inertial Signals/body_acc_x_test.txt
  ...

Usage
-----
  python prepare_uci_har_lstm_mm.py \\
    --uci_root "/path/to/UCI HAR Dataset/UCI HAR Dataset" \\
    --out_dir  "datasets/uci_har_lstm_mm"
"""

import argparse
import os

import numpy as np


def _stack_signals(sig_dir: str, split: str, prefix: str) -> np.ndarray:
    """Load body_acc_* or body_gyro_* components → (N, 128, 3)."""
    xs = []
    for axis in ("x", "y", "z"):
        path = os.path.join(sig_dir, f"{prefix}_{axis}_{split}.txt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {path}")
        xs.append(np.loadtxt(path))
    # each (N, 128)
    n = xs[0].shape[0]
    for a in xs:
        if a.shape != xs[0].shape:
            raise ValueError(f"Shape mismatch in {prefix}: {a.shape} vs {xs[0].shape}")
    stacked = np.stack(xs, axis=-1).astype(np.float32)
    return stacked


def main() -> None:
    p = argparse.ArgumentParser(
        description="UCI HAR inertial signals → (N,T,F) multimodal data.npz",
    )
    p.add_argument(
        "--uci_root",
        type=str,
        required=True,
        help="Folder containing train/, test/ with Inertial Signals/",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for data.npz",
    )
    args = p.parse_args()

    root = os.path.abspath(args.uci_root)
    for split in ("train", "test"):
        sig_dir = os.path.join(root, split, "Inertial Signals")
        if not os.path.isdir(sig_dir):
            raise FileNotFoundError(
                f"Missing directory:\n  {sig_dir}\n"
                f"--uci_root must be the inner folder that contains train/ and test/ "
                f"(with Inertial Signals/), not a placeholder path. "
                f"Example:\n"
                f"  --uci_root \".../UCI HAR Dataset/UCI HAR Dataset\""
            )

    acc_tr = _stack_signals(
        os.path.join(root, "train", "Inertial Signals"), "train", "body_acc",
    )
    gyr_tr = _stack_signals(
        os.path.join(root, "train", "Inertial Signals"), "train", "body_gyro",
    )
    acc_te = _stack_signals(
        os.path.join(root, "test", "Inertial Signals"), "test", "body_acc",
    )
    gyr_te = _stack_signals(
        os.path.join(root, "test", "Inertial Signals"), "test", "body_gyro",
    )

    y_tr = np.loadtxt(os.path.join(root, "train", "y_train.txt"))
    y_te = np.loadtxt(os.path.join(root, "test", "y_test.txt"))
    y_tr = (y_tr - 1).astype(np.int64)
    y_te = (y_te - 1).astype(np.int64)

    os.makedirs(args.out_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, "data.npz")
    np.savez(
        out_npz,
        img_train=acc_tr,
        txt_train=gyr_tr,
        labels_train=y_tr,
        img_test=acc_te,
        txt_test=gyr_te,
        labels_test=y_te,
    )
    print(f"Wrote {out_npz}")
    print(f"  img_train {acc_tr.shape}  txt_train {gyr_tr.shape}")
    print(f"  img_test  {acc_te.shape}   txt_test  {gyr_te.shape}")
    print("  Per sample: (time=128, features=3) per modality; 6 activity classes.")


if __name__ == "__main__":
    main()
