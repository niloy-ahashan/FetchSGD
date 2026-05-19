#!/usr/bin/env python3
"""
Heartbeat (UEA / TS format) → FedMultiModal ``data.npz``.

PhysioNet/CinC 2016 heart-sound spectrograms: **61 frequency bands** × **405** time steps,
binary labels **normal** / **abnormal**.

Two synthetic modalities (split along band index), analogous to other time-series MM setups:
  • img_* — spectrogram bands 0..29   (flattened length 30×405 = 12150)
  • txt_* — spectrogram bands 30..60  (flattened length 31×405 = 12555)

Labels stored as 0 = normal, 1 = abnormal for CrossEntropy (``num_classes=2``).

Usage
-----
  python prepare_heartbeat_mm.py \\
    --ts_dir datasets/Heartbeat \\
    --out_dir datasets/heartbeat_mm
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# Fixed for the published Heartbeat UEA archive files
_NUM_BANDS = 61
_SERIES_LEN = 405
_IMG_BANDS = 30  # 0..29
_TXT_BANDS = 31  # 30..60

_LABEL_TO_ID = {"normal": 0, "abnormal": 1}


def _parse_ts_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return X (N, 61, T), y (N,) with labels 0/1."""
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
        label_raw = parts[-1].strip()
        if label_raw not in _LABEL_TO_ID:
            raise ValueError(f"Unknown class label {label_raw!r} in {path}")
        label = _LABEL_TO_ID[label_raw]
        dim_strs = parts[:-1]
        if len(dim_strs) != _NUM_BANDS:
            raise ValueError(
                f"Expected {_NUM_BANDS} dimensions before label, got {len(dim_strs)} in {path}"
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
                raise ValueError("Inconsistent length within row")
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
        help="Directory containing Heartbeat_TRAIN.ts / Heartbeat_TEST.ts",
    )
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    ts_dir = os.path.abspath(args.ts_dir)
    train_path = os.path.join(ts_dir, "Heartbeat_TRAIN.ts")
    test_path = os.path.join(ts_dir, "Heartbeat_TEST.ts")
    for fp in (train_path, test_path):
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Missing {fp}")

    x_tr, y_tr = _parse_ts_file(train_path)
    x_te, y_te = _parse_ts_file(test_path)

    img_tr = x_tr[:, :_IMG_BANDS, :].reshape(x_tr.shape[0], -1).astype(np.float32)
    txt_tr = x_tr[:, _IMG_BANDS : _NUM_BANDS, :].reshape(x_tr.shape[0], -1).astype(
        np.float32
    )
    img_te = x_te[:, :_IMG_BANDS, :].reshape(x_te.shape[0], -1).astype(np.float32)
    txt_te = x_te[:, _IMG_BANDS : _NUM_BANDS, :].reshape(x_te.shape[0], -1).astype(
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
    img_dim = img_tr.shape[1]
    txt_dim = txt_tr.shape[1]
    print(f"Wrote {out_npz}")
    print(f"  train: img {img_tr.shape}  txt {txt_tr.shape}  labels {y_tr.shape}")
    print(f"  test:  img {img_te.shape}  txt {txt_te.shape}  labels {y_te.shape}")
    print(
        f"  num_classes=2 (normal/abnormal); img_dim={img_dim}, txt_dim={txt_dim}"
    )


if __name__ == "__main__":
    main()
