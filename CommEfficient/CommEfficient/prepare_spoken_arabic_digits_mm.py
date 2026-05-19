#!/usr/bin/env python3
"""
Spoken Arabic Digits (UEA / TS format) → FedMultiModal ``data.npz``.

Uses the **equal-length** variant (``*_eq_*.ts``): 13 MFCC streams × 65 time steps.

Two synthetic modalities for multimodal FL (split along coefficient index):
  • img_* — MFCC dimensions 0..5   (flattened length 65×6 = 390)
  • txt_* — MFCC dimensions 6..12  (flattened length 65×7 = 455)

Class labels in the file are 1..10 (digits 0..9); stored as 0..9 for CrossEntropy.

Usage
-----
  python prepare_spoken_arabic_digits_mm.py \\
    --ts_dir datasets/SpokenArabicDigits \\
    --out_dir datasets/spoken_arabic_digits_mm
"""

import argparse
import os

import numpy as np


def _parse_ts_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return X (N, 13, T), y (N,) with labels 0..9."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    data_start = False
    rows: list[tuple[np.ndarray, int]] = []
    for line in lines:
        line = line.strip()
        if line == "@data":
            data_start = True
            continue
        if not data_start or not line or line.startswith("#"):
            continue

        parts = line.split(":")
        if len(parts) < 2:
            continue
        label = int(float(parts[-1]))
        dim_strs = parts[:-1]
        if len(dim_strs) != 13:
            raise ValueError(
                f"Expected 13 dimensions before label, got {len(dim_strs)} in {path}"
            )
        series = []
        for ds in dim_strs:
            vals = np.array([float(x) for x in ds.split(",")], dtype=np.float32)
            series.append(vals)
        T = len(series[0])
        for s in series:
            if len(s) != T:
                raise ValueError(f"Inconsistent length in row (expected {T})")
        # (13, T)
        rows.append((np.stack(series, axis=0), label))

    if not rows:
        raise RuntimeError(f"No samples parsed from {path}")

    X = np.stack([r[0] for r in rows], axis=0)
    y = np.array([r[1] for r in rows], dtype=np.int64)
    # 1..10 -> 0..9
    y = y - 1
    return X, y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ts_dir",
        type=str,
        required=True,
        help="Directory containing SpokenArabicDigits_eq_TRAIN.ts / _TEST.ts",
    )
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    ts_dir = os.path.abspath(args.ts_dir)
    train_path = os.path.join(ts_dir, "SpokenArabicDigits_eq_TRAIN.ts")
    test_path = os.path.join(ts_dir, "SpokenArabicDigits_eq_TEST.ts")
    for fp in (train_path, test_path):
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Missing {fp} (use equal-length *_eq_*.ts files)")

    X_tr, y_tr = _parse_ts_file(train_path)
    X_te, y_te = _parse_ts_file(test_path)

    # (N, 13, T) -> modality splits -> flat vectors for MultiModalNet / SketchFusionB
    img_tr = X_tr[:, :6, :].reshape(X_tr.shape[0], -1).astype(np.float32)
    txt_tr = X_tr[:, 6:, :].reshape(X_tr.shape[0], -1).astype(np.float32)
    img_te = X_te[:, :6, :].reshape(X_te.shape[0], -1).astype(np.float32)
    txt_te = X_te[:, 6:, :].reshape(X_te.shape[0], -1).astype(np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, "data.npz")
    np.savez(
        out_npz,
        img_train=img_tr,
        txt_train=txt_tr,
        labels_train=y_tr,
        img_test=img_te,
        txt_test=txt_te,
        labels_test=y_te,
    )
    print(f"Wrote {out_npz}")
    print(f"  train: img {img_tr.shape}  txt {txt_tr.shape}  labels {y_tr.shape}")
    print(f"  test:  img {img_te.shape}  txt {txt_te.shape}  labels {y_te.shape}")
    print("  num_classes=10 (digits 0–9); img_dim=390, txt_dim=455")


if __name__ == "__main__":
    main()
