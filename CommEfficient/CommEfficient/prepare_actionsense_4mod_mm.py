#!/usr/bin/env python3
"""
ActionSense HDF5 → FedMultiModal4 ``data.npz`` (four modality vectors).

Splits the six sensor streams into **four** federated multimodal inputs::

  m0 — Eye (gaze)
  m1 — EMG-Left + EMG-Right (concatenated, flattened per window)
  m2 — Tactile-Left + Tactile-Right (concat; optional zero for S06–S09)
  m3 — IMU (Xsens joints)

Same HDF5 layout as ``prepare_actionsense_allstreams_mm.py``.
Writes ``prepare_stats.json`` with ``mod_dims`` (list of 4 ints) and ``num_classes``.

Usage
-----
  python prepare_actionsense_4mod_mm.py \\
    --hdf5_path datasets/S00_preprocessed.hdf5 \\
    --out_dir datasets/actionsense_s00_4mod_mm
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

_KEY_MAP = {
    ("eye-tracking-gaze", "position"): "Eye",
    ("myo-left", "emg"): "EMG-Left",
    ("myo-right", "emg"): "EMG-Right",
    ("tactile-glove-left", "tactile", "data"): "Tactile-Left",
    ("tactile-glove-right", "tactile", "data"): "Tactile-Right",
    ("xsens-joints", "rotation", "xzy", "deg"): "IMU",
}

_M0 = ("Eye",)
_M1 = ("EMG-Left", "EMG-Right")
_M2 = ("Tactile-Left", "Tactile-Right")
_M3 = ("IMU",)
_ALL_STREAMS = _M0 + _M1 + _M2 + _M3
_ZERO_TACTILE_SUBJECTS = frozenset({"S06", "S07", "S08", "S09"})


def _decode_sid(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _load_example_matrices(hdf_file: h5py.File) -> dict[str, np.ndarray]:
    example_matrices: dict[str, np.ndarray] = {}
    for key in hdf_file.keys():
        if not key.startswith("example_matrices_"):
            continue
        device_stream = tuple(key.split("_")[2:])
        mapped = _KEY_MAP.get(device_stream)
        if mapped is None:
            continue
        example_matrices[mapped] = np.asarray(hdf_file[key][:])
    return example_matrices


def _concat_streams(
    example_matrices: dict[str, np.ndarray],
    keys: tuple[str, ...],
    i: int,
    sid: str,
    zero_tactile: bool,
) -> np.ndarray:
    chunks = []
    for key in keys:
        x = np.asarray(example_matrices[key][i], dtype=np.float32).copy()
        if zero_tactile and sid in _ZERO_TACTILE_SUBJECTS:
            if key in ("Tactile-Left", "Tactile-Right"):
                x.fill(0.0)
        chunks.append(x.reshape(-1))
    return np.concatenate(chunks)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--test_size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--zero_tactile_s06_s09",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero Tactile-Left/Right for subjects S06–S09 (MFedMC convention).",
    )
    args = p.parse_args(argv)

    if not os.path.isfile(args.hdf5_path):
        raise FileNotFoundError(args.hdf5_path)

    with h5py.File(args.hdf5_path, "r") as hdf_file:
        example_label_indexes = np.asarray(hdf_file["example_label_indexes"][:]).squeeze()
        example_subject_ids = hdf_file["example_subject_ids"][:]
        example_matrices = _load_example_matrices(hdf_file)

    for k in _ALL_STREAMS:
        if k not in example_matrices:
            raise KeyError(
                f"Missing stream {k!r} in HDF5. Found: {sorted(example_matrices.keys())}"
            )

    n = int(example_label_indexes.shape[0])
    for k, arr in example_matrices.items():
        if arr.shape[0] != n:
            raise ValueError(
                f"Length mismatch: {k} has {arr.shape[0]} rows, expected {n}"
            )

    labels = example_label_indexes.astype(np.int64) - 1
    if labels.min() < 0:
        raise ValueError("example_label_indexes must be >= 1")

    m0_rows, m1_rows, m2_rows, m3_rows = [], [], [], []
    for i in range(n):
        sid = _decode_sid(example_subject_ids[i])
        m0_rows.append(
            _concat_streams(example_matrices, _M0, i, sid, args.zero_tactile_s06_s09)
        )
        m1_rows.append(
            _concat_streams(example_matrices, _M1, i, sid, args.zero_tactile_s06_s09)
        )
        m2_rows.append(
            _concat_streams(example_matrices, _M2, i, sid, args.zero_tactile_s06_s09)
        )
        m3_rows.append(
            _concat_streams(example_matrices, _M3, i, sid, args.zero_tactile_s06_s09)
        )

    m0_all = np.stack(m0_rows, axis=0).astype(np.float32)
    m1_all = np.stack(m1_rows, axis=0).astype(np.float32)
    m2_all = np.stack(m2_rows, axis=0).astype(np.float32)
    m3_all = np.stack(m3_rows, axis=0).astype(np.float32)

    mod_dims = [int(m0_all.shape[1]), int(m1_all.shape[1]),
                int(m2_all.shape[1]), int(m3_all.shape[1])]
    num_classes = int(labels.max()) + 1

    try:
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise ImportError(
            "prepare_actionsense_4mod_mm requires scikit-learn "
            "(pip install scikit-learn)"
        ) from e

    idx = np.arange(n)
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(
        os.path.join(args.out_dir, "data.npz"),
        m0_train=m0_all[tr_idx],
        m1_train=m1_all[tr_idx],
        m2_train=m2_all[tr_idx],
        m3_train=m3_all[tr_idx],
        labels_train=labels[tr_idx],
        m0_test=m0_all[te_idx],
        m1_test=m1_all[te_idx],
        m2_test=m2_all[te_idx],
        m3_test=m3_all[te_idx],
        labels_test=labels[te_idx],
    )
    stats = {
        "mod_dims": mod_dims,
        "num_modalities": 4,
        "num_classes": num_classes,
        "n_train": int(tr_idx.size),
        "n_test": int(te_idx.size),
        "hdf5_path": os.path.abspath(args.hdf5_path),
    }
    stats_path = os.path.join(args.out_dir, "prepare_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote {os.path.join(args.out_dir, 'data.npz')}")
    print(f"  train {stats['n_train']}  test {stats['n_test']}  "
          f"num_classes={num_classes}")
    print(f"  mod_dims={mod_dims}")
    print(f"  stats → {stats_path}")


if __name__ == "__main__":
    main()
