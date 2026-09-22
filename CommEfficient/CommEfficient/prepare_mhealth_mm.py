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
5. Train/test split, one of three toggleable modes (``--test_split``):
     - ``random`` (current default): ignore subject identity entirely;
       pool every subject's windows together and take a class-stratified
       random split (``--test_size``, ``--seed``).
       CAUTION: windows are cut with WINDOW_SIZE=128, STRIDE=64 (50%
       overlap), so consecutive windows in the same run share half their
       raw samples. Under this mode a training window and a near-duplicate
       overlapping test window (same subject, same activity run) can end
       up on opposite sides of the split — a real leakage risk, and a
       *stronger* one than ordinary same-subject leakage since the raw
       samples literally overlap. This is NOT the same thing
       ``prepare_uci_har_mm.py`` does: that script never pools or
       re-splits at all, it just inherits UCI's own pre-made,
       subject-disjoint train/test files as-is.
     - ``random_group``: the same pooled/random philosophy as ``random``
       — subject identity is not used to fix a holdout, and runs (and
       their subjects) are still scattered across train/test — but the
       split is done over groups instead of individual windows, where a
       group is one contiguous same-activity run (identified by
       ``(subject_id, run_index)``), via sklearn's ``GroupShuffleSplit``
       (``--test_size``, ``--seed``). Every window belonging to a run —
       including all of its 50%-overlapping neighbors — stays on the
       same side of the split, so the overlap-leakage risk described
       above for ``random`` cannot occur here. CAVEAT:
       ``GroupShuffleSplit`` has no ``stratify=`` option, so class
       balance across the split is only approximate, not the exact
       per-window stratification ``random`` achieves.
     - ``subject``: the original, protocol-matched behavior — fixed
       subject-wise holdout, subjects 1-8 train, 9-10 test. Immune to the
       overlap-leakage issue above, since two overlapping windows only
       ever occur within one subject's run and are therefore always on
       the same side of this split. Use this if comparing against UCI
       HAR's own (also subject-disjoint) test set.
6. Per-feature z-score normalization, fit on train only.

Usage
-----
  python prepare_mhealth_mm.py \\
    --mhealth_root datasets/mhealth+dataset/MHEALTHDATASET \\
    --out_dir      datasets/mhealth_mm

  # UCI-HAR-style pooled/random split instead, kept in its own dir so
  # both caches exist side by side:
  python prepare_mhealth_mm.py \\
    --mhealth_root datasets/mhealth+dataset/MHEALTHDATASET \\
    --out_dir      datasets/mhealth_mm_random \\
    --test_split   random

  # Group-random split: pooled/random like above, but grouped by
  # (subject, activity run) so 50%-overlapping windows never split
  # across train/test -- fixes the leakage CAUTION above while still
  # scattering runs across all subjects:
  python prepare_mhealth_mm.py \\
    --mhealth_root datasets/mhealth+dataset/MHEALTHDATASET \\
    --out_dir      datasets/mhealth_mm_random_group \\
    --test_split   random_group
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
    """Returns (mod_feats, out_labels, out_subjects, out_run_ids).

    ``out_run_ids[i]`` is the 0-indexed position, within this subject's
    own log, of the contiguous same-activity run window ``i`` came from
    (see ``_contiguous_label_runs``). It is only unique *within one
    subject* -- callers that need a globally-unique group id across all
    subjects (e.g. for ``GroupShuffleSplit``) must combine it with
    ``subject_id``.
    """
    labels = raw[:, _COLS["label"]]
    mod_feats = {m: [] for m in MODALITIES}
    out_labels = []
    out_subjects = []
    out_run_ids = []

    for run_idx, (start, end, lbl) in enumerate(_contiguous_label_runs(labels)):
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
            out_run_ids.append(run_idx)

    return mod_feats, out_labels, out_subjects, out_run_ids


# ------------------------------------------------------------------
# Train/test split — two toggleable modes
# ------------------------------------------------------------------
def _build_split(
    subjects: np.ndarray,
    labels: np.ndarray,
    test_split: str,
    test_size: float,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, test_mask) over all windows, one row each.

    ``subject``: the original fixed subject-wise holdout — every window
    from a TRAIN_SUBJECTS subject is train, every window from a
    TEST_SUBJECTS subject is test. Windows from the same subject can
    never appear on both sides.

    ``random``: subject identity is ignored entirely. Every window from
    every subject is pooled together, then a class-stratified random
    split assigns each window to train or test independently — the same
    pooled-then-split philosophy ``prepare_uci_har_mm.py`` uses (it just
    consumes an already-pooled, already-split file). Windows from the
    same subject can end up on both sides here. CAUTION: because windows
    overlap 50% within a run, a training window and an overlapping test
    window can share half their raw samples — see module docstring.

    ``random_group``: same pooled-random philosophy as ``random`` — runs
    (and their subjects) are still scattered across train/test, not held
    out wholesale like ``subject`` — but the split unit is ``groups``
    (one id per contiguous same-activity run) instead of the individual
    window, via sklearn's GroupShuffleSplit. Every window sharing a group
    lands on the same side, and since overlap only ever occurs *within*
    one run, this eliminates the overlap-leakage risk that ``random``
    has. CAVEAT: GroupShuffleSplit has no stratify= option, so class
    balance across the split is only approximate here, not the exact
    per-window stratification ``random`` gets.
    """
    if test_split == "subject":
        train_mask = np.isin(subjects, TRAIN_SUBJECTS)
        test_mask = np.isin(subjects, TEST_SUBJECTS)
        return train_mask, test_mask

    if test_split == "random_group":
        from sklearn.model_selection import GroupShuffleSplit

        if groups is None:
            raise ValueError(
                "groups is required for test_split='random_group'")
        idx = np.arange(len(labels))
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(idx, labels, groups))
        train_mask = np.zeros(len(labels), dtype=bool)
        test_mask = np.zeros(len(labels), dtype=bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        return train_mask, test_mask

    # test_split == "random"
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=seed,
    )
    train_mask = np.zeros(len(labels), dtype=bool)
    test_mask = np.zeros(len(labels), dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    return train_mask, test_mask


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
    p.add_argument("--test_split", type=str,
                    choices=["subject", "random", "random_group"],
                    default="random",
                    help="random (current default): ignore subject "
                    "identity, class-stratified random split over the "
                    "pooled windows from all 10 subjects. CAUTION: with "
                    "50%%-overlapping windows (STRIDE=64 < WINDOW_SIZE="
                    "128), this can leak near-duplicate windows across "
                    "train/test — NOT equivalent to how "
                    "prepare_uci_har_mm.py's split works (that script "
                    "never pools/re-splits at all). random_group: same "
                    "pooled/random philosophy as random -- runs are "
                    "still scattered across all subjects -- but splits "
                    "by (subject, contiguous same-activity run) group "
                    "via sklearn's GroupShuffleSplit instead of by "
                    "individual window, so 50%%-overlapping windows "
                    "from the same run can never land on opposite "
                    "sides. This fixes the leakage CAUTION above. "
                    "CAVEAT: GroupShuffleSplit has no stratify= option, "
                    "so class balance is only approximate here, not "
                    "the exact per-window stratification random gets. "
                    "subject: the original, protocol-matched, "
                    "leakage-safe fixed subject 1-8 train / 9-10 test "
                    "holdout. Toggle this and use a different --out_dir "
                    "to keep all caches around to compare.")
    p.add_argument("--test_size", type=float, default=0.3,
                    help="Test fraction for --test_split random or "
                    "random_group (ignored for subject; subject mode's "
                    "own ratio works out to ~0.198, close to this "
                    "default). For random_group this is a target, not "
                    "exact: GroupShuffleSplit assigns whole groups to "
                    "each side, so the realized test fraction can "
                    "drift slightly depending on group sizes.")
    p.add_argument("--seed", type=int, default=42,
                    help="RNG seed for --test_split random or "
                    "random_group (ignored for subject).")
    args = p.parse_args(argv)

    root = os.path.abspath(args.mhealth_root)

    all_mod_feats = {m: [] for m in MODALITIES}
    all_labels = []
    all_subjects = []
    all_run_ids = []

    all_subject_ids = TRAIN_SUBJECTS + TEST_SUBJECTS
    for sid in all_subject_ids:
        path = os.path.join(root, f"mHealth_subject{sid}.log")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        raw = _load_subject_log(path)
        mod_feats, labels, subjects, run_ids = _window_subject(raw, sid)
        for m in MODALITIES:
            all_mod_feats[m].extend(mod_feats[m])
        all_labels.extend(labels)
        all_subjects.extend(subjects)
        all_run_ids.extend(run_ids)
        print(f"  subject {sid}: {len(labels)} windows")

    labels = np.array(all_labels, dtype=np.int64)
    subjects = np.array(all_subjects, dtype=np.int64)
    run_ids = np.array(all_run_ids, dtype=np.int64)

    # Globally-unique (subject, run) group id for --test_split
    # random_group. GROUP_ID_MULTIPLIER must exceed the max number of
    # contiguous same-activity runs any one subject has -- observed max
    # is 17 (subject 7) with WINDOW_SIZE=128/STRIDE=64; 1000 leaves
    # large headroom if those constants ever change.
    GROUP_ID_MULTIPLIER = 1000
    assert run_ids.max() < GROUP_ID_MULTIPLIER, (
        "a subject has more contiguous runs than GROUP_ID_MULTIPLIER "
        "assumes -- raise GROUP_ID_MULTIPLIER in prepare_mhealth_mm.py"
    )
    groups = subjects * GROUP_ID_MULTIPLIER + run_ids

    mod_arrays = {
        m: np.array(all_mod_feats[m], dtype=np.float32) for m in MODALITIES
    }

    print(f"Total windows: {len(labels)}")
    for m in MODALITIES:
        print(f"  {m}: dim={mod_arrays[m].shape[1]}")

    train_mask, test_mask = _build_split(
        subjects, labels, args.test_split, args.test_size, args.seed,
        groups=groups,
    )

    savez_kwargs = {
        "labels_train": labels[train_mask],
        "labels_test": labels[test_mask],
        "subjects_train": subjects[train_mask],
        "subjects_test": subjects[test_mask],
        "groups_train": groups[train_mask],
        "groups_test": groups[test_mask],
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
        "test_split": args.test_split,
        "test_size": (
            args.test_size
            if args.test_split in ("random", "random_group") else None
        ),
        "seed": (
            args.seed
            if args.test_split in ("random", "random_group") else None
        ),
        "train_subjects": TRAIN_SUBJECTS if args.test_split == "subject" else None,
        "test_subjects": TEST_SUBJECTS if args.test_split == "subject" else None,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_groups_train": (
            int(len(np.unique(groups[train_mask])))
            if args.test_split == "random_group" else None
        ),
        "n_groups_test": (
            int(len(np.unique(groups[test_mask])))
            if args.test_split == "random_group" else None
        ),
        "activity_names": ACTIVITY_NAMES,
    }
    stats_path = os.path.join(args.out_dir, "prepare_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nWrote {out_npz}")
    print(f"  test_split={args.test_split}"
          + (f" test_size={args.test_size} seed={args.seed}"
             if args.test_split in ("random", "random_group") else ""))
    print(f"  train {stats['n_train']}  test {stats['n_test']}  "
          f"num_classes={NUM_CLASSES}")
    print(f"  mod_dims={mod_dims}")
    print(f"  stats -> {stats_path}")


if __name__ == "__main__":
    main()
