#!/usr/bin/env python3
"""
Build an ActionSense-compatible HDF5 from the **WISDM** raw sensor dataset so
`dataset.load_and_restructure_hdf5_data` and `main.py` work unchanged.

WISDM has four raw sensor streams (each x/y/z at 20 Hz):
  phone_accel (3)  phone_gyro (3)  watch_accel (3)  watch_gyro (3) = 12 channels

Mapping into the six ActionSense modality slots:
  • Eye            (N,T,2)         — phone_accel x,y (dominant body-motion axes)
  • EMG-Left       (N,T,8)         — phone_accel(3) + phone_gyro(3) + zeros(2)
  • EMG-Right      (N,T,8)         — watch_accel(3) + watch_gyro(3) + zeros(2)
  • Tactile-Left   (N,T,32,32)     — 12 channels in 3×4 top-left patch
  • Tactile-Right  (N,T,32,32)     — 12 channels in 3×4 shifted patch
  • IMU            (N,T,22,3)      — 12 channels padded to 66 features

Processing pipeline:
  1. Parse raw CSV files per subject × activity for all 4 sensor streams.
  2. Segment into 200-sample windows (10 s at 20 Hz), using min across streams.
  3. Resample 200 → 100 timesteps (ActionSense LSTM window length).
  4. Map into the 6 modality tensors.
  5. Write HDF5 with 1-based labels, activity strings, and subject IDs.

Example:
  python prepare_wisdm_mfedmc_hdf5.py \\
    --wisdm_root ../../datasets/wisdm/wisdm-dataset \\
    --output ./data/wisdm_mfedmc.hdf5
"""

from __future__ import annotations

import argparse
import os
import warnings
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn.functional as F


SAMPLE_RATE = 20
WINDOW_SAMPLES = 200       # 10 s × 20 Hz
TARGET_T = 100              # ActionSense LSTM window length

ACTIVITY_MAP = {
    "A": 1,  "B": 2,  "C": 3,  "D": 4,  "E": 5,  "F": 6,
    "G": 7,  "H": 8,  "I": 9,  "J": 10, "K": 11, "L": 12,
    "M": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18,
}
ACTIVITY_NAMES = {
    1: "walking", 2: "jogging", 3: "stairs", 4: "sitting",
    5: "standing", 6: "typing", 7: "teeth", 8: "soup",
    9: "chips", 10: "pasta", 11: "drinking", 12: "sandwich",
    13: "kicking", 14: "catch", 15: "dribbling", 16: "writing",
    17: "clapping", 18: "folding",
}
NUM_CLASSES = 18


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------
def parse_sensor_file(path: str) -> dict[str, np.ndarray]:
    activity_data: dict[str, list] = defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().rstrip(";").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                act = parts[1].strip()
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            except (ValueError, IndexError):
                continue
            activity_data[act].append([x, y, z])
    return {act: np.array(rows, dtype=np.float32)
            for act, rows in activity_data.items() if rows}


# ------------------------------------------------------------------
# Windowing & resampling
# ------------------------------------------------------------------
def segment_windows(data: np.ndarray, win: int) -> list[np.ndarray]:
    return [data[i * win:(i + 1) * win] for i in range(len(data) // win)]


def resample_to_target(x: np.ndarray, target: int = TARGET_T) -> np.ndarray:
    """(N, T_in, C) → (N, target, C) via linear interpolation."""
    if x.shape[1] == target:
        return x
    t = torch.from_numpy(x).float().transpose(1, 2)   # (N, C, T_in)
    t = F.interpolate(t, size=target, mode="linear", align_corners=True)
    return t.transpose(1, 2).numpy().astype(np.float32)


# ------------------------------------------------------------------
# Modality builders
# ------------------------------------------------------------------
def build_modalities(
    pa: np.ndarray, pg: np.ndarray,
    wa: np.ndarray, wg: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    pa/pg/wa/wg: each (N, T, 3) — phone accel/gyro, watch accel/gyro.
    Returns dict mapping modality name → array.
    """
    N, T, _ = pa.shape
    z2 = np.zeros((N, T, 2), dtype=np.float32)

    eye = np.stack([pa[:, :, 0], pa[:, :, 1]], axis=-1).astype(np.float32)

    emg_left = np.concatenate([pa, pg, z2], axis=-1)      # (N,T,8)
    emg_right = np.concatenate([wa, wg, z2], axis=-1)      # (N,T,8)

    ch12 = np.concatenate([pa, pg, wa, wg], axis=-1)       # (N,T,12)
    flat66 = np.zeros((N, T, 66), dtype=np.float32)
    flat66[:, :, :12] = ch12
    imu = flat66.reshape(N, T, 22, 3)                      # (N,T,22,3)

    return {
        "Eye": eye,
        "EMG-Left": emg_left,
        "EMG-Right": emg_right,
        "IMU": imu,
    }, ch12


def tactile_from_ch12(ch12_blk: np.ndarray):
    """(b,T,12) → tactile_left (b,T,32,32), tactile_right (b,T,32,32)."""
    b, T, _ = ch12_blk.shape
    tl = np.zeros((b, T, 32, 32), dtype=np.float32)
    tr = np.zeros((b, T, 32, 32), dtype=np.float32)
    for i in range(3):
        for j in range(4):
            c = i * 4 + j
            tl[:, :, i, j] = ch12_blk[:, :, c]
            tr[:, :, i + 4, j + 4] = ch12_blk[:, :, c]
    return tl, tr


def inverse_key(name: str) -> str:
    inv = {
        "Eye": ("eye-tracking-gaze", "position"),
        "EMG-Left": ("myo-left", "emg"),
        "EMG-Right": ("myo-right", "emg"),
        "Tactile-Left": ("tactile-glove-left", "tactile", "data"),
        "Tactile-Right": ("tactile-glove-right", "tactile", "data"),
        "IMU": ("xsens-joints", "rotation", "xzy", "deg"),
    }
    return "example_matrices_" + "_".join(inv[name])


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="WISDM → MFedMC ActionSense HDF5")
    p.add_argument("--wisdm_root", type=str, required=True,
                   help="Path to wisdm-dataset/ (contains raw/, activity_key.txt)")
    p.add_argument("--output", type=str, required=True,
                   help="Output .hdf5 path")
    p.add_argument("--chunk_rows", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    root = os.path.abspath(args.wisdm_root)
    raw_dir = os.path.join(root, "raw")
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Missing {raw_dir}")

    pa_dir = os.path.join(raw_dir, "phone", "accel")
    pg_dir = os.path.join(raw_dir, "phone", "gyro")
    wa_dir = os.path.join(raw_dir, "watch", "accel")
    wg_dir = os.path.join(raw_dir, "watch", "gyro")

    subject_ids = sorted({
        int(fn.split("_")[1])
        for fn in os.listdir(pa_dir)
        if fn.startswith("data_") and fn.endswith(".txt")
    })
    print(f"Found {len(subject_ids)} subjects: {subject_ids[0]}–{subject_ids[-1]}")

    all_pa, all_pg, all_wa, all_wg = [], [], [], []
    all_labels, all_subjects = [], []

    for sid in subject_ids:
        paths = {
            "pa": os.path.join(pa_dir, f"data_{sid}_accel_phone.txt"),
            "pg": os.path.join(pg_dir, f"data_{sid}_gyro_phone.txt"),
            "wa": os.path.join(wa_dir, f"data_{sid}_accel_watch.txt"),
            "wg": os.path.join(wg_dir, f"data_{sid}_gyro_watch.txt"),
        }
        missing = [k for k, v in paths.items() if not os.path.isfile(v)]
        if missing:
            warnings.warn(f"Subject {sid}: missing {missing}, skipping")
            continue

        streams = {k: parse_sensor_file(v) for k, v in paths.items()}
        common_acts = (set(streams["pa"]) & set(streams["pg"]) &
                       set(streams["wa"]) & set(streams["wg"]))

        for act in sorted(common_acts):
            if act not in ACTIVITY_MAP:
                continue
            wins = {k: segment_windows(streams[k][act], WINDOW_SAMPLES)
                    for k in ("pa", "pg", "wa", "wg")}
            n = min(len(w) for w in wins.values())
            if n == 0:
                continue
            for i in range(n):
                all_pa.append(wins["pa"][i])
                all_pg.append(wins["pg"][i])
                all_wa.append(wins["wa"][i])
                all_wg.append(wins["wg"][i])
                all_labels.append(ACTIVITY_MAP[act])
                all_subjects.append(sid)

    pa_arr = np.array(all_pa, dtype=np.float32)  # (N, 200, 3)
    pg_arr = np.array(all_pg, dtype=np.float32)
    wa_arr = np.array(all_wa, dtype=np.float32)
    wg_arr = np.array(all_wg, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int32)
    subjects = np.array(all_subjects, dtype=np.int64)

    if args.max_samples is not None:
        m = min(int(args.max_samples), len(labels))
        pa_arr, pg_arr = pa_arr[:m], pg_arr[:m]
        wa_arr, wg_arr = wa_arr[:m], wg_arr[:m]
        labels, subjects = labels[:m], subjects[:m]

    N = len(labels)
    print(f"Total windows: {N}")

    pa_arr = resample_to_target(pa_arr)
    pg_arr = resample_to_target(pg_arr)
    wa_arr = resample_to_target(wa_arr)
    wg_arr = resample_to_target(wg_arr)
    T = TARGET_T

    small, ch12 = build_modalities(pa_arr, pg_arr, wa_arr, wg_arr)
    del pa_arr, pg_arr, wa_arr, wg_arr

    label_strings = np.array(
        [ACTIVITY_NAMES[int(c)] for c in labels],
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    subj_str = np.array(
        [f"U{int(s) - 1600:02d}" for s in subjects],
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    chunk = max(1, int(args.chunk_rows))
    key_tl = inverse_key("Tactile-Left")
    key_tr = inverse_key("Tactile-Right")

    with h5py.File(args.output, "w") as h5f:
        h5f.create_dataset("example_label_indexes", data=labels)
        h5f.create_dataset("example_labels", data=label_strings)
        h5f.create_dataset("example_subject_ids", data=subj_str)
        for name, arr in small.items():
            h5f.create_dataset(inverse_key(name), data=arr)

        ds_tl = h5f.create_dataset(
            key_tl, shape=(N, T, 32, 32), dtype=np.float32,
            chunks=(min(chunk, N), T, 32, 32),
        )
        ds_tr = h5f.create_dataset(
            key_tr, shape=(N, T, 32, 32), dtype=np.float32,
            chunks=(min(chunk, N), T, 32, 32),
        )
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            tl_b, tr_b = tactile_from_ch12(ch12[start:end])
            ds_tl[start:end] = tl_b
            ds_tr[start:end] = tr_b

    print(f"\nWrote {os.path.abspath(args.output)}")
    print(f"  N={N}, T={T}, num_classes={NUM_CLASSES}")
    print(f"  Modalities: {list(small.keys())} + Tactile-Left + Tactile-Right")
    print(f"  Unique subjects: {len(np.unique(subjects))}")


if __name__ == "__main__":
    main()
