#!/usr/bin/env python3
"""
Build an ActionSense-compatible HDF5 from the **Heartbeat** UEA time-series archive
so `dataset.load_and_restructure_hdf5_data` and `main.py` work unchanged.

The Heartbeat dataset is PhysioNet/CinC 2016 heart-sound spectrograms with
61 frequency bands × 405 time steps, binary labels: normal / abnormal.

Six MFedMC modalities are synthesised by splitting the 61 bands:
  • Eye            (N, 405, 2)        — bands 0–1
  • EMG-Left       (N, 405, 8)        — bands 2–9
  • EMG-Right      (N, 405, 8)        — bands 10–17
  • Tactile-Left   (N, 405, 32, 32)   — bands 18–26 (9 bands) placed on 3×3 patch
  • Tactile-Right  (N, 405, 32, 32)   — bands 27–35 (9 bands) placed on shifted patch
  • IMU            (N, 405, 22, 3)    — bands 36–60 (25 bands) + zeros to 66 features

Labels: 2 classes (1-based in HDF5; dataset.py subtracts 1).
  1 = normal, 2 = abnormal

Synthetic client IDs are created by round-robin assigning samples to N clients.

Example:
  python prepare_heartbeat_mfedmc_hdf5.py \
    --ts_dir ../../datasets/Heartbeat \
    --output ./data/heartbeat_mfedmc.hdf5 \
    --num_clients 10
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np


HEARTBEAT_NUM_BANDS = 61
HEARTBEAT_SERIES_LEN = 405
HEARTBEAT_CLASSES = ["normal", "abnormal"]

BAND_SPLIT = {
    "Eye": (0, 2),
    "EMG-Left": (2, 10),
    "EMG-Right": (10, 18),
    "Tactile-Left": (18, 27),
    "Tactile-Right": (27, 36),
    "IMU": (36, 61),
}


def _parse_ts_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return X (N, 61, 405), y (N,) with string labels."""
    label_map = {"normal": "normal", "abnormal": "abnormal"}
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    data_start = False
    rows: list[tuple[np.ndarray, str]] = []
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
        if label_raw not in label_map:
            raise ValueError(f"Unknown class label {label_raw!r} in {path}")
        dim_strs = parts[:-1]
        if len(dim_strs) != HEARTBEAT_NUM_BANDS:
            raise ValueError(
                f"Expected {HEARTBEAT_NUM_BANDS} dims, got {len(dim_strs)} in {path}"
            )
        series = []
        for ds in dim_strs:
            vals = np.array([float(x) for x in ds.split(",")], dtype=np.float32)
            series.append(vals)
        rows.append((np.stack(series, axis=0), label_raw))

    if not rows:
        raise RuntimeError(f"No samples parsed from {path}")

    x = np.stack([r[0] for r in rows], axis=0)  # (N, 61, 405)
    y = np.array([r[1] for r in rows])
    return x, y


def build_modalities(x: np.ndarray) -> dict[str, np.ndarray]:
    """Map (N, 61, 405) spectrograms to 6 ActionSense-shaped modalities."""
    N = x.shape[0]
    T = HEARTBEAT_SERIES_LEN

    # Transpose to (N, T, bands) for LSTM consumption
    x_t = x.transpose(0, 2, 1)  # (N, 405, 61)

    start, end = BAND_SPLIT["Eye"]
    eye = x_t[:, :, start:end].astype(np.float32)  # (N, 405, 2)

    start, end = BAND_SPLIT["EMG-Left"]
    emg_left = x_t[:, :, start:end].astype(np.float32)  # (N, 405, 8)

    start, end = BAND_SPLIT["EMG-Right"]
    emg_right = x_t[:, :, start:end].astype(np.float32)  # (N, 405, 8)

    # Tactile: 9 bands placed in a 3x3 patch of a 32x32 grid
    start, end = BAND_SPLIT["Tactile-Left"]
    tl_bands = x_t[:, :, start:end]  # (N, 405, 9)
    tactile_left = np.zeros((N, T, 32, 32), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            c = i * 3 + j
            tactile_left[:, :, i, j] = tl_bands[:, :, c]

    start, end = BAND_SPLIT["Tactile-Right"]
    tr_bands = x_t[:, :, start:end]  # (N, 405, 9)
    tactile_right = np.zeros((N, T, 32, 32), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            c = i * 3 + j
            tactile_right[:, :, i + 4, j + 4] = tr_bands[:, :, c]

    # IMU: 25 bands padded to 66, reshaped to (N, T, 22, 3)
    start, end = BAND_SPLIT["IMU"]
    imu_bands = x_t[:, :, start:end]  # (N, 405, 25)
    flat66 = np.zeros((N, T, 66), dtype=np.float32)
    flat66[:, :, :25] = imu_bands
    imu = flat66.reshape(N, T, 22, 3)

    return {
        "Eye": eye,
        "EMG-Left": emg_left,
        "EMG-Right": emg_right,
        "Tactile-Left": tactile_left,
        "Tactile-Right": tactile_right,
        "IMU": imu,
    }


def inverse_key_to_hdf5_name(mapped: str) -> str:
    """dataset.py key_map inverse: modality name -> example_matrices_* suffix."""
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


def main() -> None:
    p = argparse.ArgumentParser(description="Heartbeat → MFedMC ActionSense HDF5")
    p.add_argument(
        "--ts_dir",
        type=str,
        required=True,
        help="Directory with Heartbeat_TRAIN.ts and Heartbeat_TEST.ts",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .hdf5 path (e.g. ./data/heartbeat_mfedmc.hdf5)",
    )
    p.add_argument(
        "--num_clients",
        type=int,
        default=10,
        help="Number of synthetic clients to create (default: 10)",
    )
    p.add_argument(
        "--chunk_rows",
        type=int,
        default=128,
        help="HDF5 write chunk size for large arrays",
    )
    args = p.parse_args()

    ts_dir = os.path.abspath(args.ts_dir)
    train_path = os.path.join(ts_dir, "Heartbeat_TRAIN.ts")
    test_path = os.path.join(ts_dir, "Heartbeat_TEST.ts")
    for fp in (train_path, test_path):
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Missing {fp}")

    print("Parsing .ts files...")
    x_tr, y_tr = _parse_ts_file(train_path)
    x_te, y_te = _parse_ts_file(test_path)

    x = np.concatenate([x_tr, x_te], axis=0)  # (N, 61, 405)
    y = np.concatenate([y_tr, y_te], axis=0)
    N = x.shape[0]
    print(f"  Total samples: {N} ({x_tr.shape[0]} train + {x_te.shape[0]} test)")

    # 1-based label indexes (dataset.py subtracts 1)
    label_idx_1based = np.array(
        [HEARTBEAT_CLASSES.index(lbl) + 1 for lbl in y], dtype=np.int32
    )
    label_strings = np.array(y, dtype=h5py.string_dtype(encoding="utf-8"))

    # Synthetic subject IDs via round-robin
    num_clients = args.num_clients
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    client_ids_arr = np.empty(N, dtype=object)
    for i, idx in enumerate(indices):
        client_ids_arr[idx] = f"C{(i % num_clients):02d}"
    subj_str = np.array(
        client_ids_arr.tolist(), dtype=h5py.string_dtype(encoding="utf-8")
    )

    print("Building modalities...")
    modalities = build_modalities(x)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    chunk = max(1, int(args.chunk_rows))

    print("Writing HDF5...")
    with h5py.File(args.output, "w") as h5f:
        h5f.create_dataset("example_label_indexes", data=label_idx_1based)
        h5f.create_dataset("example_labels", data=label_strings)
        h5f.create_dataset("example_subject_ids", data=subj_str)

        for name, arr in modalities.items():
            ds_name = inverse_key_to_hdf5_name(name)
            if arr.ndim == 4:
                h5f.create_dataset(
                    ds_name,
                    data=arr,
                    chunks=(min(chunk, N), arr.shape[1], arr.shape[2], arr.shape[3]),
                )
            else:
                h5f.create_dataset(ds_name, data=arr)

    print(f"\nWrote {os.path.abspath(args.output)}")
    print(f"  N={N}, T={HEARTBEAT_SERIES_LEN}, num_classes=2")
    print(f"  Modalities: {list(modalities.keys())}")
    print(f"  Unique clients: {num_clients}")
    print(f"\nRun with:")
    print(f"  python main.py --data_path {args.output} --num_classes 2")


if __name__ == "__main__":
    main()
