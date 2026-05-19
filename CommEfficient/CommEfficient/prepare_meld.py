#!/usr/bin/env python3
"""
Build FedMultiModal ``data.npz`` from the MELD (Multimodal EmotionLines Dataset).

Modalities
----------
* **txt_feats** — Sentence embeddings from ``all-MiniLM-L6-v2`` (384-dim)
* **img_feats** — Audio features extracted from video clips:
    40 MFCCs  (mean + std)  = 80
    12 chroma (mean + std)  = 24
    1  RMS    (mean + std)  =  2
    1  ZCR    (mean + std)  =  2
    7  spectral contrast (mean + std) = 14
    128 mel-spectrogram  (mean + std) = 256
    Total = 378-dim

Labels: 7 emotion classes (anger=0, disgust=1, fear=2, joy=3, neutral=4,
        sadness=5, surprise=6)

Usage
-----
  python prepare_meld.py \\
    --meld_root datasets/MELD.Raw \\
    --out_dir   datasets/meld_mm

Supports checkpointing: audio features are saved per-chunk so the script
can resume if interrupted.
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
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}

FFMPEG_BIN = None


def _get_ffmpeg() -> str:
    global FFMPEG_BIN
    if FFMPEG_BIN is None:
        import imageio_ffmpeg
        FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMPEG_BIN


def _extract_audio_features(mp4_path: str, sr: int = 16000) -> np.ndarray | None:
    import librosa

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        subprocess.run(
            [_get_ffmpeg(), "-i", mp4_path,
             "-ac", "1", "-ar", str(sr), "-y", "-loglevel", "error", wav_path],
            capture_output=True, check=True, timeout=30,
        )
        y, _ = librosa.load(wav_path, sr=sr)
    except Exception:
        return None
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    if len(y) < sr * 0.1:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

    def _stats(feat: np.ndarray) -> np.ndarray:
        return np.concatenate([feat.mean(axis=1), feat.std(axis=1)])

    vec = np.concatenate([
        _stats(mfcc), _stats(chroma), _stats(rms),
        _stats(zcr), _stats(contrast), _stats(mel_db),
    ]).astype(np.float32)

    del y, mfcc, chroma, rms, zcr, contrast, mel, mel_db
    return vec


def _extract_text_features(texts: list[str], batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, batch_size=batch_size,
                              show_progress_bar=True, convert_to_numpy=True)
    del model
    gc.collect()
    return embeddings.astype(np.float32)


def _read_csv(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _process_split(
    csv_path: str,
    video_dir: str,
    split_name: str,
    cache_dir: str,
    chunk_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process one MELD split with checkpointing."""

    rows = _read_csv(csv_path)
    print(f"\n[{split_name}] CSV rows: {len(rows)}", flush=True)

    utterances: list[str] = []
    video_paths: list[str] = []
    labels: list[int] = []

    for row in rows:
        dia_id = row["Dialogue_ID"].strip()
        utt_id = row["Utterance_ID"].strip()
        emotion = row["Emotion"].strip().lower()
        utterance = row["Utterance"].strip()
        if emotion not in EMOTION_MAP:
            continue
        mp4_name = f"dia{dia_id}_utt{utt_id}.mp4"
        mp4_path = os.path.join(video_dir, mp4_name)
        utterances.append(utterance)
        video_paths.append(mp4_path)
        labels.append(EMOTION_MAP[emotion])

    n = len(utterances)
    print(f"[{split_name}] Valid samples: {n}", flush=True)

    # --- Text features (with cache) ---
    txt_cache = os.path.join(cache_dir, f"{split_name}_txt.npy")
    if os.path.exists(txt_cache):
        print(f"[{split_name}] Loading cached text features from {txt_cache}",
              flush=True)
        txt_feats = np.load(txt_cache)
    else:
        print(f"[{split_name}] Extracting text features ...", flush=True)
        txt_feats = _extract_text_features(utterances)
        np.save(txt_cache, txt_feats)
        print(f"[{split_name}] Text features shape: {txt_feats.shape}", flush=True)

    # --- Audio features (chunked with cache) ---
    n_chunks = (n + chunk_size - 1) // chunk_size
    audio_chunks: list[np.ndarray] = []
    audio_dim = None
    total_missing = 0

    for ci in range(n_chunks):
        chunk_cache = os.path.join(cache_dir,
                                   f"{split_name}_audio_chunk{ci}.npy")
        start = ci * chunk_size
        end = min(start + chunk_size, n)

        if os.path.exists(chunk_cache):
            chunk_arr = np.load(chunk_cache)
            if audio_dim is None:
                audio_dim = chunk_arr.shape[1]
            audio_chunks.append(chunk_arr)
            print(f"[{split_name}] Audio chunk {ci+1}/{n_chunks} "
                  f"({start}-{end}) loaded from cache", flush=True)
            continue

        print(f"[{split_name}] Audio chunk {ci+1}/{n_chunks} "
              f"({start}-{end}) extracting ...", flush=True)

        chunk_feats: list[np.ndarray | None] = []
        for i in range(start, end):
            if (i - start) % 200 == 0:
                print(f"  [{split_name}] audio {i+1}/{n}", flush=True)

            vp = video_paths[i]
            if not os.path.exists(vp):
                feat = None
            else:
                feat = _extract_audio_features(vp)

            if feat is not None and audio_dim is None:
                audio_dim = feat.shape[0]
            chunk_feats.append(feat)

        if audio_dim is None:
            raise RuntimeError(f"No audio features in chunk {ci}")

        chunk_arr = np.zeros((end - start, audio_dim), dtype=np.float32)
        for j, feat in enumerate(chunk_feats):
            if feat is not None:
                chunk_arr[j] = feat
            else:
                total_missing += 1

        np.save(chunk_cache, chunk_arr)
        audio_chunks.append(chunk_arr)
        del chunk_feats
        gc.collect()
        print(f"[{split_name}] Audio chunk {ci+1}/{n_chunks} saved", flush=True)

    audio_feats = np.concatenate(audio_chunks, axis=0)
    if total_missing > 0:
        print(f"[{split_name}] WARNING: {total_missing}/{n} "
              f"clips missing/failed → zero-filled", flush=True)

    labels_arr = np.array(labels, dtype=np.int64)
    print(f"[{split_name}] Audio features shape: {audio_feats.shape}", flush=True)
    print(f"[{split_name}] Label distribution: "
          f"{dict(zip(*np.unique(labels_arr, return_counts=True)))}", flush=True)

    return audio_feats, txt_feats, labels_arr


def main() -> None:
    p = argparse.ArgumentParser(description="MELD → multimodal data.npz")
    p.add_argument("--meld_root", type=str, required=True,
                   help="Path to extracted MELD.Raw directory")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory for data.npz")
    p.add_argument("--merge_dev_into_train", action="store_true",
                   help="Merge dev split into training data")
    p.add_argument("--chunk_size", type=int, default=1000,
                   help="Audio processing chunk size for checkpointing")
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
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {desc}: {path}")

    tr_audio, tr_txt, tr_labels = _process_split(
        train_csv, train_video, "train", cache_dir, args.chunk_size)

    if args.merge_dev_into_train:
        dv_audio, dv_txt, dv_labels = _process_split(
            dev_csv, dev_video, "dev", cache_dir, args.chunk_size)
        tr_audio = np.concatenate([tr_audio, dv_audio])
        tr_txt = np.concatenate([tr_txt, dv_txt])
        tr_labels = np.concatenate([tr_labels, dv_labels])
        print(f"\n[merged] Train + Dev: {len(tr_labels)} samples", flush=True)

    te_audio, te_txt, te_labels = _process_split(
        test_csv, test_video, "test", cache_dir, args.chunk_size)

    out_npz = os.path.join(out_dir, "data.npz")
    np.savez(
        out_npz,
        img_train=tr_audio,
        txt_train=tr_txt,
        labels_train=tr_labels,
        img_test=te_audio,
        txt_test=te_txt,
        labels_test=te_labels,
    )

    print(f"\nWrote {out_npz}")
    print(f"  img_train (audio) {tr_audio.shape}  txt_train (text) {tr_txt.shape}")
    print(f"  img_test  (audio) {te_audio.shape}   txt_test  (text) {te_txt.shape}")
    print(f"  classes: 7 (single-label), img_dim={tr_audio.shape[1]}, "
          f"txt_dim={tr_txt.shape[1]}")


if __name__ == "__main__":
    main()
