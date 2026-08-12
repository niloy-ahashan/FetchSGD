#!/usr/bin/env python3
"""
Build FedMultiModal `data.npz` from the WISDM Smartphone and Smartwatch dataset.

The WISDM dataset contains accelerometer and gyroscope data from a smartphone
(pocket) and a smartwatch (wrist) collected from 51 subjects performing 18
activities.  Raw sensor readings are at 20 Hz.

Multimodal split
----------------
  • img_*  — phone sensors  (pocket accel + gyro → captures body motion)
  • txt_*  — watch sensors  (wrist accel + gyro → captures hand motion)

These are genuinely different physical sensor locations, making them a natural
multimodal pair.

Processing pipeline
-------------------
1. Parse raw CSV files (subject,activity,timestamp,x,y,z;) per sensor stream.
2. For each subject × activity, align phone and watch streams by taking the
   common timestamp range, then segment into fixed-length windows
   (default 10 s = 200 samples at 20 Hz).
3. Extract time-domain features per window for each axis (×3) and magnitude,
   plus inter-axis correlations — separately for accelerometer and gyroscope.
4. Concatenate accel + gyro features per device to form each modality vector.
5. Subject-wise 80/20 train/test split.

Usage
-----
  python prepare_wisdm_mm.py \
    --wisdm_root datasets/wisdm/wisdm-dataset \
    --out_dir    datasets/wisdm_mm
"""

from __future__ import annotations

import argparse
import os
import warnings
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats


SAMPLE_RATE = 20          # Hz
WINDOW_SEC = 10           # seconds per window
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SEC   # 200 samples

ACTIVITY_MAP = {
    "A": 0,  "B": 1,  "C": 2,  "D": 3,  "E": 4,  "F": 5,
    "G": 6,  "H": 7,  "I": 8,  "J": 9,  "K": 10, "L": 11,
    "M": 12, "O": 13, "P": 14, "Q": 15, "R": 16, "S": 17,
}
NUM_CLASSES = 18

TEST_SUBJECTS = {1600, 1610, 1620, 1630, 1640,
                 1605, 1615, 1625, 1635, 1645}


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------
def parse_sensor_file(path: str) -> dict[str, list[np.ndarray]]:
    """
    Parse a WISDM raw sensor file.

    Returns {activity_code: array of shape (N, 3)} where columns are x, y, z.
    """
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
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
            except (ValueError, IndexError):
                continue
            activity_data[act].append([x, y, z])

    return {act: np.array(rows, dtype=np.float64)
            for act, rows in activity_data.items() if rows}


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------
def _axis_features(x: np.ndarray) -> np.ndarray:
    """14 time-domain features for a single axis window."""
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


def extract_sensor_features(window: np.ndarray) -> np.ndarray:
    """
    Extract features from a (WINDOW_SIZE, 3) sensor window.

    Returns a 1-D feature vector:
      3 axes × 14 features + magnitude 14 features + 3 correlations = 59
    """
    feats = []
    for ax in range(3):
        feats.append(_axis_features(window[:, ax]))

    mag = np.sqrt(np.sum(window ** 2, axis=1))
    feats.append(_axis_features(mag))

    corr_xy = np.corrcoef(window[:, 0], window[:, 1])[0, 1]
    corr_xz = np.corrcoef(window[:, 0], window[:, 2])[0, 1]
    corr_yz = np.corrcoef(window[:, 1], window[:, 2])[0, 1]
    corrs = np.array([corr_xy, corr_xz, corr_yz], dtype=np.float64)
    corrs = np.nan_to_num(corrs, nan=0.0)

    feats.append(corrs)
    return np.concatenate(feats)


def extract_device_features(accel_win: np.ndarray,
                            gyro_win: np.ndarray) -> np.ndarray:
    """
    Features for one device (phone or watch).

    accel_win, gyro_win: each (WINDOW_SIZE, 3).
    Returns 59 (accel) + 59 (gyro) = 118 features.
    """
    return np.concatenate([
        extract_sensor_features(accel_win),
        extract_sensor_features(gyro_win),
    ])


# ------------------------------------------------------------------
# Windowing
# ------------------------------------------------------------------
def segment_windows(data: np.ndarray, window_size: int) -> list[np.ndarray]:
    """Split data into non-overlapping windows, dropping the remainder."""
    n = len(data)
    n_windows = n // window_size
    windows = []
    for i in range(n_windows):
        windows.append(data[i * window_size:(i + 1) * window_size])
    return windows


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="WISDM → multimodal data.npz for FedMultiModal")
    p.add_argument("--wisdm_root", type=str, required=True,
                   help="Path to wisdm-dataset/ containing raw/, activity_key.txt")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory for data.npz")
    p.add_argument("--window_sec", type=int, default=WINDOW_SEC,
                   help="Window duration in seconds (default: 10)")
    p.add_argument("--test_frac", type=float, default=0.2,
                   help="Fraction of subjects for test (default: 0.2)")
    args = p.parse_args()

    root = os.path.abspath(args.wisdm_root)
    window_size = SAMPLE_RATE * args.window_sec

    raw_dir = os.path.join(root, "raw")
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Missing {raw_dir}")

    phone_accel_dir = os.path.join(raw_dir, "phone", "accel")
    phone_gyro_dir = os.path.join(raw_dir, "phone", "gyro")
    watch_accel_dir = os.path.join(raw_dir, "watch", "accel")
    watch_gyro_dir = os.path.join(raw_dir, "watch", "gyro")

    subject_ids = set()
    for fn in os.listdir(phone_accel_dir):
        if fn.startswith("data_") and fn.endswith(".txt"):
            sid = int(fn.split("_")[1])
            subject_ids.add(sid)
    subject_ids = sorted(subject_ids)
    print(f"Found {len(subject_ids)} subjects: {subject_ids[0]}–{subject_ids[-1]}")

    all_phone_feats = []
    all_watch_feats = []
    all_labels = []
    all_subjects = []

    for sid in subject_ids:
        pa_path = os.path.join(phone_accel_dir, f"data_{sid}_accel_phone.txt")
        pg_path = os.path.join(phone_gyro_dir, f"data_{sid}_gyro_phone.txt")
        wa_path = os.path.join(watch_accel_dir, f"data_{sid}_accel_watch.txt")
        wg_path = os.path.join(watch_gyro_dir, f"data_{sid}_gyro_watch.txt")

        missing = [p for p in [pa_path, pg_path, wa_path, wg_path]
                   if not os.path.isfile(p)]
        if missing:
            warnings.warn(f"Subject {sid}: missing files {missing}, skipping")
            continue

        pa = parse_sensor_file(pa_path)
        pg = parse_sensor_file(pg_path)
        wa = parse_sensor_file(wa_path)
        wg = parse_sensor_file(wg_path)

        common_activities = (set(pa.keys()) & set(pg.keys()) &
                             set(wa.keys()) & set(wg.keys()))

        for act in sorted(common_activities):
            if act not in ACTIVITY_MAP:
                continue

            pa_wins = segment_windows(pa[act], window_size)
            pg_wins = segment_windows(pg[act], window_size)
            wa_wins = segment_windows(wa[act], window_size)
            wg_wins = segment_windows(wg[act], window_size)

            n_wins = min(len(pa_wins), len(pg_wins),
                         len(wa_wins), len(wg_wins))
            if n_wins == 0:
                continue

            for i in range(n_wins):
                phone_feat = extract_device_features(pa_wins[i], pg_wins[i])
                watch_feat = extract_device_features(wa_wins[i], wg_wins[i])
                all_phone_feats.append(phone_feat)
                all_watch_feats.append(watch_feat)
                all_labels.append(ACTIVITY_MAP[act])
                all_subjects.append(sid)

    phone_feats = np.array(all_phone_feats, dtype=np.float32)
    watch_feats = np.array(all_watch_feats, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    subjects = np.array(all_subjects, dtype=np.int64)

    print(f"Total windows: {len(labels)}")
    print(f"Phone features (img_dim): {phone_feats.shape[1]}")
    print(f"Watch features (txt_dim): {watch_feats.shape[1]}")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        act_name = [k for k, v in ACTIVITY_MAP.items() if v == u][0]
        print(f"  Class {u} ({act_name}): {c} windows")

    # --- subject-wise train/test split ---
    test_sids = TEST_SUBJECTS & set(subject_ids)
    if not test_sids:
        n_test = max(1, int(len(subject_ids) * args.test_frac))
        rng = np.random.RandomState(42)
        test_sids = set(rng.choice(subject_ids, size=n_test, replace=False))
    train_sids = set(subject_ids) - test_sids
    print(f"Train subjects ({len(train_sids)}): {sorted(train_sids)}")
    print(f"Test subjects  ({len(test_sids)}):  {sorted(test_sids)}")

    train_mask = np.isin(subjects, list(train_sids))
    test_mask = np.isin(subjects, list(test_sids))

    img_train = phone_feats[train_mask]
    txt_train = watch_feats[train_mask]
    labels_train = labels[train_mask]
    img_test = phone_feats[test_mask]
    txt_test = watch_feats[test_mask]
    labels_test = labels[test_mask]

    # --- normalize: fit on train, apply to both ---
    mean_img = img_train.mean(axis=0, keepdims=True)
    std_img = img_train.std(axis=0, keepdims=True)
    std_img[std_img < 1e-8] = 1.0
    img_train = (img_train - mean_img) / std_img
    img_test = (img_test - mean_img) / std_img

    mean_txt = txt_train.mean(axis=0, keepdims=True)
    std_txt = txt_train.std(axis=0, keepdims=True)
    std_txt[std_txt < 1e-8] = 1.0
    txt_train = (txt_train - mean_txt) / std_txt
    txt_test = (txt_test - mean_txt) / std_txt

    np.nan_to_num(img_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(img_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(txt_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(txt_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    os.makedirs(args.out_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, "data.npz")
    np.savez(
        out_npz,
        img_train=img_train,
        txt_train=txt_train,
        labels_train=labels_train,
        img_test=img_test,
        txt_test=txt_test,
        labels_test=labels_test,
    )
    print(f"\nWrote {out_npz}")
    print(f"  img_train {img_train.shape}  txt_train {txt_train.shape}")
    print(f"  img_test  {img_test.shape}   txt_test  {txt_test.shape}")
    print(f"  classes: {NUM_CLASSES} (single-label), "
          f"img_dim={img_train.shape[1]}, txt_dim={txt_train.shape[1]}")


if __name__ == "__main__":
    main()
