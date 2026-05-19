#!/usr/bin/env python3
"""
ActionSense HDF5 (all-streams MFedMC layout) → FedMultiModal ``data.npz``.

Loads the same structure as ``MFedMC/ActionSense/dataset.load_and_restructure_hdf5_data``:
``example_matrices_*``, ``example_label_indexes``, ``example_subject_ids``.

Two modalities for FetchSGD ``MultiModalNet`` / ``SketchFusionB`` (flattened time-series):
  • img_* — Eye + EMG-Left + EMG-Right (concatenated per sample)
  • txt_* — Tactile-Left + Tactile-Right + IMU

Optionally zeros tactile streams for subjects S06–S09 (same convention as MFedMC).

Train/test split: stratified holdout (default 70/30, seed 42).

Writes ``prepare_stats.json`` with img_dim, txt_dim, num_classes for shell wrappers.

Usage
-----
  python prepare_actionsense_allstreams_mm.py \\
    --hdf5_path MFedMC/ActionSense/data/data_processed_allStreams_10s_10hz_5subj_ex20-20_allActs.hdf5 \\
    --out_dir datasets/actionsense_allstreams_mm
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

# Same mapping as MFedMC/ActionSense/dataset.py
_KEY_MAP = {
    ("eye-tracking-gaze", "position"): "Eye",
    ("myo-left", "emg"): "EMG-Left",
    ("myo-right", "emg"): "EMG-Right",
    ("tactile-glove-left", "tactile", "data"): "Tactile-Left",
    ("tactile-glove-right", "tactile", "data"): "Tactile-Right",
    ("xsens-joints", "rotation", "xzy", "deg"): "IMU",
}

_IMG_KEYS = ("Eye", "EMG-Left", "EMG-Right")
_TXT_KEYS = ("Tactile-Left", "Tactile-Right", "IMU")
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

    for k in _IMG_KEYS + _TXT_KEYS:
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

    img_rows = []
    txt_rows = []

    for i in range(n):
        sid = _decode_sid(example_subject_ids[i])

        chunks_img = []
        for key in _IMG_KEYS:
            x = np.asarray(example_matrices[key][i], dtype=np.float32).copy()
            chunks_img.append(x.reshape(-1))

        chunks_txt = []
        for key in _TXT_KEYS:
            x = np.asarray(example_matrices[key][i], dtype=np.float32).copy()
            if args.zero_tactile_s06_s09 and sid in _ZERO_TACTILE_SUBJECTS:
                if key in ("Tactile-Left", "Tactile-Right"):
                    x.fill(0.0)
            chunks_txt.append(x.reshape(-1))

        img_rows.append(np.concatenate(chunks_img))
        txt_rows.append(np.concatenate(chunks_txt))

    img_all = np.stack(img_rows, axis=0).astype(np.float32)
    txt_all = np.stack(txt_rows, axis=0).astype(np.float32)

    img_dim = img_all.shape[1]
    txt_dim = txt_all.shape[1]

    # Verify constant dimensions across samples (already stacked)
    num_classes = int(labels.max()) + 1

    try:
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise ImportError(
            "prepare_actionsense_allstreams_mm requires scikit-learn "
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
        img_train=img_all[tr_idx],
        txt_train=txt_all[tr_idx],
        labels_train=labels[tr_idx],
        img_test=img_all[te_idx],
        txt_test=txt_all[te_idx],
        labels_test=labels[te_idx],
    )
    stats = {
        "img_dim": img_dim,
        "txt_dim": txt_dim,
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
    print(f"  img_dim={img_dim}  txt_dim={txt_dim}")
    print(f"  stats → {stats_path}")


if __name__ == "__main__":
    main()
