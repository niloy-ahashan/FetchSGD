#!/usr/bin/env python3
"""
Build FedMultiModal ``data.npz`` from the PTB-XL ECG dataset.

Following MFedMC [Yuan et al., 2024] and the protocol in [53], the 12-lead
ECG is split into two modalities:

* **img_feats (Limb Lead ECG)** — leads I, II, III, aVR, aVL, aVF
  Raw 100Hz waveform flattened: 1000 timesteps × 6 leads = 6000-dim

* **txt_feats (Precordial Lead ECG)** — leads V1, V2, V3, V4, V5, V6
  Raw 100Hz waveform flattened: 1000 timesteps × 6 leads = 6000-dim

Labels: 5 diagnostic superclasses (multi-label, one-hot):
  0=NORM, 1=MI, 2=STTC, 3=CD, 4=HYP
  An SCP code contributes to a superclass when its likelihood ≥ 50%.

Train/Test: PTB-XL recommended stratified folds 1-8 for training,
folds 9+10 for testing.

Usage
-----
  python prepare_ptbxl.py \\
    --ptbxl_root datasets/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \\
    --out_dir    datasets/ptbxl_mm \\
    --sampling_rate 100

Requirements: wfdb, pandas
"""

from __future__ import annotations

import argparse
import ast
import os

import numpy as np
import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

LIMB_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF"]
PRECORDIAL_LEADS = ["V1", "V2", "V3", "V4", "V5", "V6"]


def _load_signals(ptbxl_root: str, filenames: list[str],
                  sampling_rate: int) -> np.ndarray:
    """Load ECG waveforms. Returns (N, timesteps, 12) array."""
    import wfdb

    folder = "records100" if sampling_rate == 100 else "records500"
    signals = []
    for i, fn in enumerate(filenames):
        if (i + 1) % 2000 == 0 or i == 0:
            print(f"  Loading signal {i+1}/{len(filenames)}", flush=True)
        record = wfdb.rdrecord(os.path.join(ptbxl_root, fn))
        signals.append(record.p_signal.astype(np.float32))

    return np.stack(signals)  # (N, timesteps, 12)


def _split_leads(signals: np.ndarray,
                 lead_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Split 12-lead signals into Limb and Precordial lead arrays."""
    limb_idx = [lead_names.index(l) for l in LIMB_LEADS]
    prec_idx = [lead_names.index(l) for l in PRECORDIAL_LEADS]

    limb = signals[:, :, limb_idx]    # (N, T, 6)
    prec = signals[:, :, prec_idx]    # (N, T, 6)

    N, T, C = limb.shape
    return limb.reshape(N, T * C), prec.reshape(N, T * C)


def _build_labels(df: pd.DataFrame, scp_df: pd.DataFrame,
                  min_likelihood: float = 50.0) -> np.ndarray:
    """Create multi-label (N, 5) one-hot array from SCP codes."""
    diag = scp_df[scp_df["diagnostic"] == 1]
    code_to_super = diag["diagnostic_class"].to_dict()

    labels = np.zeros((len(df), len(SUPERCLASSES)), dtype=np.float32)
    for i, scp_codes_str in enumerate(df["scp_codes"]):
        codes = ast.literal_eval(scp_codes_str)
        for code, likelihood in codes.items():
            if code in code_to_super and likelihood >= min_likelihood:
                sc = code_to_super[code]
                labels[i, SUPERCLASSES.index(sc)] = 1.0
    return labels


def main() -> None:
    p = argparse.ArgumentParser(description="PTB-XL → multimodal data.npz")
    p.add_argument("--ptbxl_root", type=str, required=True,
                   help="Path to ptb-xl-...-1.0.3 directory")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--sampling_rate", type=int, default=100,
                   choices=[100, 500],
                   help="Sampling rate (100 or 500 Hz)")
    p.add_argument("--min_likelihood", type=float, default=50.0,
                   help="Minimum likelihood for SCP code to count")
    args = p.parse_args()

    root = os.path.abspath(args.ptbxl_root)

    # ---- Load metadata ----
    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"))
    scp_df = pd.read_csv(os.path.join(root, "scp_statements.csv"),
                         index_col=0)

    fn_col = "filename_lr" if args.sampling_rate == 100 else "filename_hr"

    # ---- Build labels ----
    labels = _build_labels(df, scp_df, args.min_likelihood)
    has_label = labels.sum(axis=1) > 0

    print(f"Total records: {len(df)}")
    print(f"Records with ≥1 superclass label: {has_label.sum()}")
    print(f"Multi-label records: {(labels.sum(axis=1) > 1).sum()}")
    for j, sc in enumerate(SUPERCLASSES):
        print(f"  {sc}: {int(labels[:, j].sum())}")

    # ---- Train/Test split (PTB-XL recommended folds) ----
    train_mask = (df["strat_fold"] <= 8).values & has_label
    test_mask = (df["strat_fold"] >= 9).values & has_label

    train_files = df.loc[train_mask, fn_col].tolist()
    test_files = df.loc[test_mask, fn_col].tolist()
    train_labels = labels[train_mask]
    test_labels = labels[test_mask]

    print(f"\nTrain: {len(train_files)} records (folds 1-8)")
    print(f"Test:  {len(test_files)} records (folds 9-10)")

    # ---- Load signals ----
    print("\nLoading train signals ...", flush=True)
    train_sig = _load_signals(root, train_files, args.sampling_rate)
    print(f"  Train signals shape: {train_sig.shape}")

    print("Loading test signals ...", flush=True)
    test_sig = _load_signals(root, test_files, args.sampling_rate)
    print(f"  Test signals shape: {test_sig.shape}")

    # Lead names from first record
    import wfdb
    rec0 = wfdb.rdrecord(os.path.join(root, train_files[0]))
    lead_names = [l.upper() for l in rec0.sig_name]
    print(f"Lead order: {lead_names}")

    # ---- Split into Limb + Precordial ----
    train_limb, train_prec = _split_leads(train_sig, lead_names)
    test_limb, test_prec = _split_leads(test_sig, lead_names)

    del train_sig, test_sig

    # ---- Normalize per-lead (z-score on train, apply to test) ----
    mean_limb = train_limb.mean(axis=0, keepdims=True)
    std_limb = train_limb.std(axis=0, keepdims=True) + 1e-8
    train_limb = ((train_limb - mean_limb) / std_limb).astype(np.float32)
    test_limb = ((test_limb - mean_limb) / std_limb).astype(np.float32)

    mean_prec = train_prec.mean(axis=0, keepdims=True)
    std_prec = train_prec.std(axis=0, keepdims=True) + 1e-8
    train_prec = ((train_prec - mean_prec) / std_prec).astype(np.float32)
    test_prec = ((test_prec - mean_prec) / std_prec).astype(np.float32)

    # ---- Save ----
    os.makedirs(args.out_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, "data.npz")
    np.savez(
        out_npz,
        img_train=train_limb,       # Limb Lead ECG
        txt_train=train_prec,       # Precordial Lead ECG
        labels_train=train_labels,  # multi-label (N, 5)
        img_test=test_limb,
        txt_test=test_prec,
        labels_test=test_labels,
    )

    print(f"\nWrote {out_npz}")
    print(f"  img_train (Limb Lead)      {train_limb.shape}")
    print(f"  txt_train (Precordial Lead) {train_prec.shape}")
    print(f"  labels_train (multi-label)  {train_labels.shape}")
    print(f"  img_test  {test_limb.shape}")
    print(f"  txt_test  {test_prec.shape}")
    print(f"  labels_test {test_labels.shape}")
    print(f"  classes: 5 (multi-label), "
          f"img_dim={train_limb.shape[1]}, txt_dim={train_prec.shape[1]}")


if __name__ == "__main__":
    main()
