#!/usr/bin/env python3
"""
Face Detection (UEA / TS format) → FedMultiModal ``data.npz``.

**144** multivariate series × **62** time steps; binary labels **0** / **1**.

Two modalities (split along series index), same pattern as Spoken Arabic Digits:
  • img_* — dimensions 0..71   (flattened length 62×72 = 4464)
  • txt_* — dimensions 72..143  (flattened length 62×72 = 4464)

Usage
-----
  python prepare_face_detection_mm.py \\
    --ts_dir datasets/FaceDetection \\
    --out_dir datasets/face_detection_mm
"""

from __future__ import annotations

import argparse
import os

import numpy as np

_NUM_DIMS = 144
_SERIES_LEN = 62
_IMG_DIMS = 72  # 0..71
_TXT_DIMS = 72  # 72..143


def _parse_ts_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return X (N, 144, T), y (N,) with labels 0/1."""
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
        if label not in (0, 1):
            raise ValueError(f"Expected label 0 or 1, got {label} in {path}")
        dim_strs = parts[:-1]
        if len(dim_strs) != _NUM_DIMS:
            raise ValueError(
                f"Expected {_NUM_DIMS} dimensions before label, got {len(dim_strs)} in {path}"
            )
        series = []
        for ds in dim_strs:
            vals = np.array([float(x) for x in ds.split(",")], dtype=np.float32)
            series.append(vals)
        t = len(series[0])
        if t != _SERIES_LEN:
            raise ValueError(
                f"Expected series length {_SERIES_LEN}, got {t} in {path}"
            )
        for s in series:
            if len(s) != t:
                raise ValueError("Inconsistent length in row")
        rows.append((np.stack(series, axis=0), label))

    if not rows:
        raise RuntimeError(f"No samples parsed from {path}")

    x = np.stack([r[0] for r in rows], axis=0)
    y = np.array([r[1] for r in rows], dtype=np.int64)
    return x, y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ts_dir",
        type=str,
        required=True,
        help="Directory containing FaceDetection_TRAIN.ts / FaceDetection_TEST.ts",
    )
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    ts_dir = os.path.abspath(args.ts_dir)
    train_path = os.path.join(ts_dir, "FaceDetection_TRAIN.ts")
    test_path = os.path.join(ts_dir, "FaceDetection_TEST.ts")
    for fp in (train_path, test_path):
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Missing {fp}")

    x_tr, y_tr = _parse_ts_file(train_path)
    x_te, y_te = _parse_ts_file(test_path)

    assert _IMG_DIMS + _TXT_DIMS == _NUM_DIMS
    img_tr = x_tr[:, :_IMG_DIMS, :].reshape(x_tr.shape[0], -1).astype(np.float32)
    txt_tr = x_tr[:, _IMG_DIMS : _NUM_DIMS, :].reshape(x_tr.shape[0], -1).astype(
        np.float32
    )
    img_te = x_te[:, :_IMG_DIMS, :].reshape(x_te.shape[0], -1).astype(np.float32)
    txt_te = x_te[:, _IMG_DIMS : _NUM_DIMS, :].reshape(x_te.shape[0], -1).astype(
        np.float32
    )

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
    print("  num_classes=2; img_dim=4464, txt_dim=4464")


if __name__ == "__main__":
    main()
