#!/usr/bin/env python3
"""
Build FedMultiModal4 ``data.npz`` from the mHealth dataset (4 modalities).

The mHealth dataset records body motion and vital signs for 10 subjects
performing 12 activities, using Shimmer2 sensors on the chest, left ankle,
and right-lower-arm, sampled at 50 Hz. Each ``mHealth_subject<N>.log`` is a
tab-separated file with 24 columns and no header:

  col  0- 2: chest accelerometer (x,y,z)
  col  3- 4: chest ECG (lead 1, lead 2)
  col  5- 7: left-ankle accelerometer (x,y,z)
  col  8-10: left-ankle gyroscope (x,y,z)
  col 11-13: left-ankle magnetometer (x,y,z)
  col 14-16: right-lower-arm accelerometer (x,y,z)
  col 17-19: right-lower-arm gyroscope (x,y,z)
  col 20-22: right-lower-arm magnetometer (x,y,z)
  col    23: activity label (0 = null class, discarded; 1-12 = activities)

Multimodal split
-----------------
  • Acc  (m0) — chest + ankle + arm accelerometer (3 locations)
  • Gyro (m1) — ankle + arm gyroscope (2 locations; mHealth has no chest gyro)
  • Mag  (m2) — ankle + arm magnetometer (2 locations)
  • ECG  (m3) — chest 2-lead ECG (1 location, 2 channels)

Processing pipeline
--------------------
1. Parse each subject's log, dropping null-class (label 0) rows.
2. Within each subject, split into maximal contiguous runs of a single
   nonzero label, then window each run (window_size=128, stride=64 → 50%
   overlap) — windows never cross an activity/subject boundary.
3. Extract 14 time-domain features per axis (mean/std/min/max/range/median/
   mad/iqr/skew/kurtosis/rms/energy/mean_abs_diff/zcr) plus the same 14 for
   the window's magnitude channel plus pairwise inter-axis correlations —
   the same feature function used by prepare_wisdm_mm.py, generalized here
   to an arbitrary channel count per group instead of a fixed 3.
4. Concatenate per-location feature vectors to form each modality vector.
5. Fixed subject-wise train/test split (subjects 1-8 train, 9-10 test).
6. Per-feature z-score normalization, fit on train only.

Usage
-----
  python prepare_mhealth_mm.py \\
    --mhealth_root datasets/mhealth+dataset/MHEALTHDATASET \\
    --out_dir      datasets/mhealth_mm
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import stats as sp_stats

WINDOW_SIZE = 128
STRIDE = 64
SAMPLE_RATE = 50
NUM_CLASSES = 12
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 10]

ACTIVITY_NAMES = [
    "Standing still",
    "Sitting and relaxing",
    "Lying down",
    "Walking",
    "Climbing stairs",
    "Waist bends forward",
    "Frontal elevation of arms",
    "Knees bending (crouching)",
    "Cycling",
    "Jogging",
    "Running",
    "Jump front & back",
]

# Column groups, 0-indexed, matching the README.txt layout.
_COLS = {
    "chest_acc": slice(0, 3),
    "chest_ecg": slice(3, 5),
    "ankle_acc": slice(5, 8),
    "ankle_gyro": slice(8, 11),
    "ankle_mag": slice(11, 14),
    "arm_acc": slice(14, 17),
    "arm_gyro": slice(17, 20),
    "arm_mag": slice(20, 23),
    "label": 23,
}

# Which column groups make up each of the 4 modalities, in order.
MODALITIES = ["Acc", "Gyro", "Mag", "ECG"]
_MODALITY_GROUPS = {
    "Acc": ["chest_acc", "ankle_acc", "arm_acc"],
    "Gyro": ["ankle_gyro", "arm_gyro"],
    "Mag": ["ankle_mag", "arm_mag"],
    "ECG": ["chest_ecg"],
}


# ------------------------------------------------------------------
# Feature extraction (generalized from prepare_wisdm_mm.py)
# ------------------------------------------------------------------
def _axis_features(x: np.ndarray) -> np.ndarray:
    """14 time-domain features for a single axis/channel window."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1) if n > 1 else 0.0
    mn, mx = np.min(x), np.max(x)
    rng = mx - mn
    med = np.median(x)
    mad = np.mean(np.abs(x - mean))
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    sk = float(sp_stats.skew(x, bias=False)) if n > 2 else 0.0
    ku = float(sp_stats.kurtosis(x, bias=False)) if n > 3 else 0.0
    rms = np.sqrt(np.mean(x ** 2))
    energy = np.sum(x ** 2) / n
    mean_abs_diff = np.mean(np.abs(np.diff(x))) if n > 1 else 0.0
    zcr = np.sum(np.diff(np.sign(x - mean)) != 0) / n if n > 1 else 0.0
    return np.array([mean, std, mn, mx, rng, med, mad, iqr,
                      sk, ku, rms, energy, mean_abs_diff, zcr],
                     dtype=np.float64)


def extract_group_features(window: np.ndarray) -> np.ndarray:
    """
    Extract features from a (WINDOW_SIZE, k) window of k related channels
    (e.g. the x/y/z axes of one sensor, or the 2 leads of an ECG).

    Returns: k axes × 14 features + magnitude 14 features + C(k,2)
    pairwise correlations.
    """
    k = window.shape[1]
    feats = [_axis_features(window[:, ax]) for ax in range(k)]

    mag = np.sqrt(np.sum(window ** 2, axis=1))
    feats.append(_axis_features(mag))

    corrs = []
    for i in range(k):
        for j in range(i + 1, k):
            corrs.append(np.corrcoef(window[:, i], window[:, j])[0, 1])
    corrs = np.nan_to_num(np.array(corrs, dtype=np.float64), nan=0.0)
    feats.append(corrs)

    return np.concatenate(feats)


def extract_modality_features(row_window: np.ndarray, modality: str) -> np.ndarray:
    """Concatenate per-location group features for one of the 4 modalities."""
    chunks = []
    for group in _MODALITY_GROUPS[modality]:
        cols = _COLS[group]
        chunks.append(extract_group_features(row_window[:, cols]))
    return np.concatenate(chunks)


# ------------------------------------------------------------------
# Parsing + windowing
# ------------------------------------------------------------------
def _load_subject_log(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter="\t", dtype=np.float64)


def _contiguous_label_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (start, end, label) for each maximal run of a single nonzero label."""
    runs = []
    n = len(labels)
    i = 0
    while i < n:
        lbl = labels[i]
        if lbl == 0:
            i += 1
            continue
        j = i + 1
        while j < n and labels[j] == lbl:
            j += 1
        runs.append((i, j, int(lbl)))
        i = j
    return runs


def _window_subject(raw: np.ndarray, subject_id: int):
    labels = raw[:, _COLS["label"]]
    mod_feats = {m: [] for m in MODALITIES}
    out_labels = []
    out_subjects = []

    for start, end, lbl in _contiguous_label_runs(labels):
        run_len = end - start
        n_windows = (run_len - WINDOW_SIZE) // STRIDE + 1 if run_len >= WINDOW_SIZE else 0
        for w in range(n_windows):
            ws = start + w * STRIDE
            we = ws + WINDOW_SIZE
            window = raw[ws:we]
            for m in MODALITIES:
                mod_feats[m].append(extract_modality_features(window, m))
            out_labels.append(lbl - 1)  # 0-indexed
            out_subjects.append(subject_id)

    return mod_feats, out_labels, out_subjects


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="mHealth → FedMultiModal4 data.npz (4 modalities)")
    p.add_argument("--mhealth_root", type=str,
                    default="datasets/mhealth+dataset/MHEALTHDATASET",
                    help="Path to the dir containing mHealth_subject<N>.log files")
    p.add_argument("--out_dir", type=str, default="datasets/mhealth_mm",
                    help="Directory for data.npz / prepare_stats.json")
    args = p.parse_args(argv)

    root = os.path.abspath(args.mhealth_root)

    all_mod_feats = {m: [] for m in MODALITIES}
    all_labels = []
    all_subjects = []

    all_subject_ids = TRAIN_SUBJECTS + TEST_SUBJECTS
    for sid in all_subject_ids:
        path = os.path.join(root, f"mHealth_subject{sid}.log")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        raw = _load_subject_log(path)
        mod_feats, labels, subjects = _window_subject(raw, sid)
        for m in MODALITIES:
            all_mod_feats[m].extend(mod_feats[m])
        all_labels.extend(labels)
        all_subjects.extend(subjects)
        print(f"  subject {sid}: {len(labels)} windows")

    labels = np.array(all_labels, dtype=np.int64)
    subjects = np.array(all_subjects, dtype=np.int64)
    mod_arrays = {
        m: np.array(all_mod_feats[m], dtype=np.float32) for m in MODALITIES
    }

    print(f"Total windows: {len(labels)}")
    for m in MODALITIES:
        print(f"  {m}: dim={mod_arrays[m].shape[1]}")

    train_mask = np.isin(subjects, TRAIN_SUBJECTS)
    test_mask = np.isin(subjects, TEST_SUBJECTS)

    savez_kwargs = {
        "labels_train": labels[train_mask],
        "labels_test": labels[test_mask],
        "subjects_train": subjects[train_mask],
        "subjects_test": subjects[test_mask],
        "mod_names": np.array(MODALITIES),
    }

    mod_dims = []
    for k, m in enumerate(MODALITIES):
        train = mod_arrays[m][train_mask]
        test = mod_arrays[m][test_mask]

        mean = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        train = (train - mean) / std
        test = (test - mean) / std
        np.nan_to_num(train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        savez_kwargs[f"m{k}_train"] = train.astype(np.float32)
        savez_kwargs[f"m{k}_test"] = test.astype(np.float32)
        mod_dims.append(int(train.shape[1]))

    os.makedirs(args.out_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, "data.npz")
    np.savez(out_npz, **savez_kwargs)

    stats = {
        "dataset": "MHEALTH",
        "source": "https://archive.ics.uci.edu/dataset/319/mhealth+dataset",
        "modalities": MODALITIES,
        "mod_dims": mod_dims,
        "num_classes": NUM_CLASSES,
        "num_modalities": len(MODALITIES),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "sample_rate": SAMPLE_RATE,
        "train_subjects": TRAIN_SUBJECTS,
        "test_subjects": TEST_SUBJECTS,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "activity_names": ACTIVITY_NAMES,
    }
    stats_path = os.path.join(args.out_dir, "prepare_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nWrote {out_npz}")
    print(f"  train {stats['n_train']}  test {stats['n_test']}  "
          f"num_classes={NUM_CLASSES}")
    print(f"  mod_dims={mod_dims}")
    print(f"  stats -> {stats_path}")


if __name__ == "__main__":
    main()
