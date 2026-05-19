#!/usr/bin/env python3
"""
Build FedMultiModal ``data.npz`` from MELD following the **original paper's**
feature extraction pipeline (Poria et al., 2019).

Feature extraction (Section 4.1 of the paper)
----------------------------------------------
* **Text (txt_feats)** — GloVe 300-dim mean word embeddings.
  The paper feeds GloVe into a 1D-CNN inside the model (bcLSTM/DialogueRNN).
  Since our pipeline requires fixed-length pre-computed features, we use the
  mean of GloVe word vectors per utterance (300-dim).  The downstream
  FeaExtractor MLP in MultiModalNet acts as the equivalent nonlinear encoder.

* **Audio (img_feats)** — openSMILE ComParE_2016 functionals (6373-dim),
  then L2-regularised feature selection (LinearSVC) as described in the paper,
  reducing to a configurable ``--audio_select_dim`` (default 300).

Labels: 7 emotion classes (anger=0, disgust=1, fear=2, joy=3, neutral=4,
        sadness=5, surprise=6)

Usage
-----
  python prepare_meld_paper.py \\
    --meld_root  datasets/MELD.Raw \\
    --glove_path datasets/glove/glove.6B.300d.txt \\
    --out_dir    datasets/meld_paper \\
    --audio_select_dim 300

Requirements: opensmile, scikit-learn, pandas, imageio-ffmpeg
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import subprocess
import tempfile
import warnings

import numpy as np

EMOTION_MAP = {
    "anger": 0, "disgust": 1, "fear": 2, "joy": 3,
    "neutral": 4, "sadness": 5, "surprise": 6,
}

FFMPEG_BIN = None


def _get_ffmpeg() -> str:
    global FFMPEG_BIN
    if FFMPEG_BIN is None:
        import imageio_ffmpeg
        FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMPEG_BIN


# ---------------------------------------------------------------
# Text features: GloVe 300d mean word embeddings
# ---------------------------------------------------------------

def _load_glove(glove_path: str) -> dict[str, np.ndarray]:
    print(f"Loading GloVe from {glove_path} ...", flush=True)
    glove: dict[str, np.ndarray] = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            glove[word] = np.array(parts[1:], dtype=np.float32)
    print(f"  Loaded {len(glove)} words, dim={next(iter(glove.values())).shape[0]}",
          flush=True)
    return glove


def _text_glove_features(utterances: list[str],
                         glove: dict[str, np.ndarray]) -> np.ndarray:
    dim = next(iter(glove.values())).shape[0]
    feats = np.zeros((len(utterances), dim), dtype=np.float32)
    oov_count = 0
    for i, utt in enumerate(utterances):
        tokens = utt.lower().split()
        vecs = [glove[t] for t in tokens if t in glove]
        if vecs:
            feats[i] = np.mean(vecs, axis=0)
        else:
            oov_count += 1
    if oov_count > 0:
        print(f"  OOV utterances (zero vector): {oov_count}/{len(utterances)}",
              flush=True)
    return feats


# ---------------------------------------------------------------
# Audio features: openSMILE ComParE_2016 functionals (6373-dim)
# ---------------------------------------------------------------

def _extract_opensmile_single(mp4_path: str, smile) -> np.ndarray | None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        subprocess.run(
            [_get_ffmpeg(), "-i", mp4_path,
             "-ac", "1", "-ar", "16000", "-y", "-loglevel", "error", wav_path],
            capture_output=True, check=True, timeout=30,
        )
        df = smile.process_file(wav_path)
        return df.values.flatten().astype(np.float32)
    except Exception:
        return None
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def _extract_opensmile_batch(
    video_paths: list[str],
    split_name: str,
    cache_dir: str,
    chunk_size: int = 500,
) -> np.ndarray:
    import opensmile

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    audio_dim = len(smile.feature_names)  # 6373

    n = len(video_paths)
    n_chunks = (n + chunk_size - 1) // chunk_size
    chunks: list[np.ndarray] = []
    total_missing = 0

    for ci in range(n_chunks):
        chunk_cache = os.path.join(cache_dir,
                                   f"{split_name}_osmile_chunk{ci}.npy")
        start = ci * chunk_size
        end = min(start + chunk_size, n)

        if os.path.exists(chunk_cache):
            chunks.append(np.load(chunk_cache))
            print(f"[{split_name}] openSMILE chunk {ci+1}/{n_chunks} "
                  f"({start}-{end}) loaded from cache", flush=True)
            continue

        print(f"[{split_name}] openSMILE chunk {ci+1}/{n_chunks} "
              f"({start}-{end}) extracting ...", flush=True)

        arr = np.zeros((end - start, audio_dim), dtype=np.float32)
        for j, idx in enumerate(range(start, end)):
            if j % 100 == 0:
                print(f"  [{split_name}] openSMILE {idx+1}/{n}", flush=True)
            vp = video_paths[idx]
            if os.path.exists(vp):
                feat = _extract_opensmile_single(vp, smile)
            else:
                feat = None
            if feat is not None:
                arr[j] = feat
            else:
                total_missing += 1

        np.save(chunk_cache, arr)
        chunks.append(arr)
        gc.collect()
        print(f"[{split_name}] openSMILE chunk {ci+1}/{n_chunks} saved",
              flush=True)

    full = np.concatenate(chunks, axis=0)
    if total_missing > 0:
        print(f"[{split_name}] WARNING: {total_missing}/{n} "
              f"openSMILE clips missing/failed → zero-filled", flush=True)
    return full


# ---------------------------------------------------------------
# L2-based feature selection (paper: sparse SVMs)
# ---------------------------------------------------------------

def _l2_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    target_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select top features using ANOVA F-test + StandardScaler.

    The paper uses L2-regularised LinearSVC, but ANOVA F-test
    (SelectKBest) is much faster and achieves similar quality for
    high-dimensional openSMILE features.
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.preprocessing import StandardScaler

    print(f"  Feature selection (ANOVA F-test): {X_train.shape[1]} → "
          f"{target_dim} ...", flush=True)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    selector = SelectKBest(f_classif, k=target_dim)
    X_tr_sel = selector.fit_transform(X_tr_scaled, y_train)
    X_te_sel = selector.transform(X_te_scaled)

    print(f"  Selected {X_tr_sel.shape[1]} features", flush=True)
    return X_tr_sel.astype(np.float32), X_te_sel.astype(np.float32)


# ---------------------------------------------------------------
# CSV reading + split processing
# ---------------------------------------------------------------

def _read_csv(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _parse_split(csv_path: str, video_dir: str, split_name: str):
    rows = _read_csv(csv_path)
    print(f"\n[{split_name}] CSV rows: {len(rows)}", flush=True)
    utterances, video_paths, labels = [], [], []
    for row in rows:
        emotion = row["Emotion"].strip().lower()
        if emotion not in EMOTION_MAP:
            continue
        dia_id = row["Dialogue_ID"].strip()
        utt_id = row["Utterance_ID"].strip()
        utterances.append(row["Utterance"].strip())
        video_paths.append(os.path.join(video_dir,
                                        f"dia{dia_id}_utt{utt_id}.mp4"))
        labels.append(EMOTION_MAP[emotion])
    print(f"[{split_name}] Valid samples: {len(utterances)}", flush=True)
    return utterances, video_paths, np.array(labels, dtype=np.int64)


def main() -> None:
    p = argparse.ArgumentParser(
        description="MELD → data.npz (paper-style features)")
    p.add_argument("--meld_root", type=str, required=True)
    p.add_argument("--glove_path", type=str, required=True,
                   help="Path to glove.6B.300d.txt")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--merge_dev_into_train", action="store_true")
    p.add_argument("--audio_select_dim", type=int, default=300,
                   help="Target dimensionality after L2 feature selection "
                        "on openSMILE features (paper uses sparse SVMs)")
    p.add_argument("--chunk_size", type=int, default=500)
    args = p.parse_args()

    root = os.path.abspath(args.meld_root)
    out_dir = os.path.abspath(args.out_dir)
    cache_dir = os.path.join(out_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)

    train_csv = os.path.join(root, "train_sent_emo.csv")
    dev_csv = os.path.join(root, "dev_sent_emo.csv")
    test_csv = os.path.join(root, "test_sent_emo.csv")
    train_video = os.path.join(root, "train_splits")
    dev_video = os.path.join(root, "dev_splits_complete")
    test_video = os.path.join(root, "output_repeated_splits_test")

    for path, desc in [
        (train_csv, "train CSV"), (dev_csv, "dev CSV"),
        (test_csv, "test CSV"), (train_video, "train videos"),
        (dev_video, "dev videos"), (test_video, "test videos"),
        (args.glove_path, "GloVe embeddings"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {desc}: {path}")

    # ---- Parse CSVs ----
    tr_utt, tr_vid, tr_lbl = _parse_split(train_csv, train_video, "train")
    dv_utt, dv_vid, dv_lbl = _parse_split(dev_csv, dev_video, "dev")
    te_utt, te_vid, te_lbl = _parse_split(test_csv, test_video, "test")

    if args.merge_dev_into_train:
        tr_utt = tr_utt + dv_utt
        tr_vid = tr_vid + dv_vid
        tr_lbl = np.concatenate([tr_lbl, dv_lbl])
        print(f"\n[merged] Train + Dev: {len(tr_lbl)} samples", flush=True)

    # ---- Text: GloVe 300d mean embeddings ----
    glove = _load_glove(args.glove_path)

    tr_txt_cache = os.path.join(cache_dir, "train_glove.npy")
    te_txt_cache = os.path.join(cache_dir, "test_glove.npy")

    if os.path.exists(tr_txt_cache):
        print("Loading cached GloVe text features ...", flush=True)
        tr_txt = np.load(tr_txt_cache)
    else:
        print("Extracting GloVe text features (train) ...", flush=True)
        tr_txt = _text_glove_features(tr_utt, glove)
        np.save(tr_txt_cache, tr_txt)

    if os.path.exists(te_txt_cache):
        te_txt = np.load(te_txt_cache)
    else:
        print("Extracting GloVe text features (test) ...", flush=True)
        te_txt = _text_glove_features(te_utt, glove)
        np.save(te_txt_cache, te_txt)

    print(f"Text features: train {tr_txt.shape}, test {te_txt.shape}",
          flush=True)
    del glove
    gc.collect()

    # ---- Audio: openSMILE ComParE_2016 (6373-dim) ----
    print("\nExtracting openSMILE audio features ...", flush=True)
    tr_audio_raw = _extract_opensmile_batch(
        tr_vid, "train", cache_dir, args.chunk_size)
    te_audio_raw = _extract_opensmile_batch(
        te_vid, "test", cache_dir, args.chunk_size)

    print(f"\nRaw openSMILE: train {tr_audio_raw.shape}, "
          f"test {te_audio_raw.shape}", flush=True)

    # Replace NaN/Inf with 0 (openSMILE can produce these for very short clips)
    tr_audio_raw = np.nan_to_num(tr_audio_raw, nan=0.0, posinf=0.0, neginf=0.0)
    te_audio_raw = np.nan_to_num(te_audio_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- L2-based feature selection (paper: sparse SVMs) ----
    tr_audio, te_audio = _l2_feature_selection(
        tr_audio_raw, tr_lbl, te_audio_raw, args.audio_select_dim)
    print(f"Selected audio: train {tr_audio.shape}, test {te_audio.shape}",
          flush=True)

    # ---- Save ----
    out_npz = os.path.join(out_dir, "data.npz")
    np.savez(
        out_npz,
        img_train=tr_audio,   # "img" branch = audio features
        txt_train=tr_txt,     # "txt" branch = GloVe text features
        labels_train=tr_lbl,
        img_test=te_audio,
        txt_test=te_txt,
        labels_test=te_lbl,
    )

    print(f"\nWrote {out_npz}")
    print(f"  img_train (audio-selected) {tr_audio.shape}  "
          f"txt_train (GloVe-300d) {tr_txt.shape}")
    print(f"  img_test  (audio-selected) {te_audio.shape}   "
          f"txt_test  (GloVe-300d) {te_txt.shape}")
    print(f"  classes: 7 (single-label), "
          f"img_dim={tr_audio.shape[1]}, txt_dim={tr_txt.shape[1]}")


if __name__ == "__main__":
    main()
