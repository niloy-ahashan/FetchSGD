#!/usr/bin/env python3
"""
Build FedMultiModal `data.npz` from the UCI HAR Smartphone dataset.

Following multimodal FL practice (cf. separate global modality streams), we split
the 561-dimensional feature vector into two physical modalities:

  • img_*  — accelerometer-related features (names containing no "Gyro")
  • txt_*  — gyroscope-related features (names containing "Gyro")

This mirrors treating IMU as distinct sensing channels rather than image/text;
MultiModalNet still expects two vectors (img_dim / txt_dim).

Usage
-----
  python prepare_uci_har_mm.py \\
    --uci_root "/path/to/UCI HAR Dataset/UCI HAR Dataset" \\
    --out_dir  "/path/to/output_uci_mm"
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _parse_feature_columns(features_txt: str) -> tuple[list[int], list[int]]:
    """Return 0-based column indices for acc vs gyro modalities."""
    acc_idx: list[int] = []
    gyro_idx: list[int] = []
    with open(features_txt, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            col = int(parts[0]) - 1  # 1-based in file
            name = parts[1]
            if "Gyro" in name:
                gyro_idx.append(col)
            else:
                acc_idx.append(col)
    return acc_idx, gyro_idx


def main() -> None:
    p = argparse.ArgumentParser(description="UCI HAR → multimodal data.npz")
    p.add_argument(
        "--uci_root",
        type=str,
        required=True,
        help="Folder containing features.txt, train/, test/ (inner UCI HAR dir)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory for data.npz (FedMultiModal will cache clients here)",
    )
    args = p.parse_args()

    root = os.path.abspath(args.uci_root)
    feat_path = os.path.join(root, "features.txt")
    if not os.path.isfile(feat_path):
        raise FileNotFoundError(f"Missing {feat_path}")

    acc_idx, gyro_idx = _parse_feature_columns(feat_path)
    if not acc_idx or not gyro_idx:
        raise RuntimeError("Empty modality split; check features.txt")

    x_tr = np.loadtxt(os.path.join(root, "train", "X_train.txt"))
    y_tr = np.loadtxt(os.path.join(root, "train", "y_train.txt"), dtype=np.int64)
    x_te = np.loadtxt(os.path.join(root, "test", "X_test.txt"))
    y_te = np.loadtxt(os.path.join(root, "test", "y_test.txt"), dtype=np.int64)

    if x_tr.ndim == 1:
        x_tr = x_tr.reshape(1, -1)
    if x_te.ndim == 1:
        x_te = x_te.reshape(1, -1)

    # UCI labels are 1..6
    y_tr = (y_tr - 1).astype(np.int64)
    y_te = (y_te - 1).astype(np.int64)

    img_train = x_tr[:, acc_idx].astype(np.float32)
    txt_train = x_tr[:, gyro_idx].astype(np.float32)
    img_test = x_te[:, acc_idx].astype(np.float32)
    txt_test = x_te[:, gyro_idx].astype(np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, "data.npz")
    np.savez(
        out_npz,
        img_train=img_train,
        txt_train=txt_train,
        labels_train=y_tr,
        img_test=img_test,
        txt_test=txt_test,
        labels_test=y_te,
    )
    print(f"Wrote {out_npz}")
    print(f"  img_train {img_train.shape}  txt_train {txt_train.shape}")
    print(f"  img_test  {img_test.shape}   txt_test  {txt_test.shape}")
    print(f"  classes: 6 (single-label), img_dim={img_train.shape[1]}, "
          f"txt_dim={txt_train.shape[1]}")


if __name__ == "__main__":
    main()
