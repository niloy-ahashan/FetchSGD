#!/usr/bin/env python3
"""
Build FedMultiModal ``data.npz`` from VQA v2.0 (Balanced Real Images).

Two modalities:
* **img_feats** — ResNet-50 (pretrained, ImageNet) global avg-pool → 2048-dim
* **txt_feats** — sentence-transformers ``all-MiniLM-L6-v2`` → 384-dim

Labels: top-K most frequent ``multiple_choice_answer`` from training set.
Samples whose answer is not in the top-K vocabulary are dropped.

Train/Test: VQA train2014 / val2014.

Usage
-----
  python prepare_vqa.py \\
    --vqa_root  datasets/vqa_v2 \\
    --out_dir   datasets/vqa_mm \\
    --top_k     1000 \\
    --batch_size 256

Requirements: torchvision, sentence-transformers, Pillow
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image


# ──────────────────────────────────────────────────────────────
# Image feature extraction (ResNet-50)
# ──────────────────────────────────────────────────────────────

def _build_image_model(device: str):
    import torchvision.models as models
    import torchvision.transforms as T

    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device).eval()

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    return resnet, transform


CHUNK_SIZE = 5000  # images per cache chunk


def _extract_image_features(
    image_ids: List[int],
    img_dir: str,
    split: str,
    device: str,
    batch_size: int,
    cache_dir: str,
) -> Dict[int, np.ndarray]:
    """Return {image_id: feature_vector} for all unique image_ids.

    Features are extracted in chunks of CHUNK_SIZE and cached separately
    so the process can resume after an OOM kill.
    """
    unique_ids = sorted(set(image_ids))
    n_chunks = (len(unique_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE

    feats: Dict[int, np.ndarray] = {}
    model = None

    for ci in range(n_chunks):
        chunk_cache = os.path.join(cache_dir,
                                   f"img_feats_{split}_chunk{ci}.npy")
        chunk_ids = unique_ids[ci * CHUNK_SIZE : (ci + 1) * CHUNK_SIZE]

        if os.path.exists(chunk_cache):
            arr = np.load(chunk_cache, allow_pickle=True).item()
            feats.update(arr)
            print(f"  Chunk {ci}/{n_chunks}: loaded {len(arr)} cached feats",
                  flush=True)
            continue

        if model is None:
            print(f"  Extracting ResNet-50 features for {len(unique_ids)} "
                  f"unique {split} images ({n_chunks} chunks) ...",
                  flush=True)
            model, transform = _build_image_model(device)

        chunk_feats: Dict[int, np.ndarray] = {}
        for start in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[start : start + batch_size]
            imgs, valid_ids = [], []
            for iid in batch_ids:
                fname = f"COCO_{split}_{iid:012d}.jpg"
                path = os.path.join(img_dir, fname)
                if not os.path.exists(path):
                    continue
                img = Image.open(path).convert("RGB")
                imgs.append(transform(img))
                valid_ids.append(iid)

            if not imgs:
                continue

            batch_tensor = torch.stack(imgs).to(device)
            with torch.no_grad():
                out = model(batch_tensor).cpu().numpy()
            del batch_tensor
            torch.cuda.empty_cache()

            for j, iid in enumerate(valid_ids):
                chunk_feats[iid] = out[j].astype(np.float32)

        np.save(chunk_cache, chunk_feats)
        feats.update(chunk_feats)
        total_done = min((ci + 1) * CHUNK_SIZE, len(unique_ids))
        print(f"  Chunk {ci}/{n_chunks}: extracted {len(chunk_feats)} feats  "
              f"({total_done}/{len(unique_ids)} total)", flush=True)
        del chunk_feats

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Total image features: {len(feats)}")
    return feats


# ──────────────────────────────────────────────────────────────
# Text feature extraction (sentence-transformers)
# ──────────────────────────────────────────────────────────────

def _extract_text_features(
    questions: List[str],
    cache_path: str,
    batch_size: int = 512,
) -> np.ndarray:
    """Return (N, 384) array of question embeddings."""
    if os.path.exists(cache_path):
        print(f"  Loading cached text features from {cache_path}")
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer
    print(f"  Encoding {len(questions)} questions with all-MiniLM-L6-v2 ...",
          flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(cache_path, embeddings)
    print(f"  Cached to {cache_path}")
    return embeddings


# ──────────────────────────────────────────────────────────────
# Build answer vocabulary (top-K from training annotations)
# ──────────────────────────────────────────────────────────────

def _build_answer_vocab(annotations: List[dict], top_k: int) -> Dict[str, int]:
    from collections import Counter
    counter = Counter(ann["multiple_choice_answer"] for ann in annotations)
    vocab = {}
    for i, (ans, _) in enumerate(counter.most_common(top_k)):
        vocab[ans] = i
    return vocab


# ──────────────────────────────────────────────────────────────
# Process one split
# ──────────────────────────────────────────────────────────────

def _process_split(
    questions_path: str,
    annotations_path: str,
    img_dir: str,
    split: str,
    ans_vocab: Dict[str, int],
    device: str,
    batch_size: int,
    cache_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (img_feats, txt_feats, labels) for a split."""
    with open(questions_path) as f:
        q_data = json.load(f)["questions"]
    with open(annotations_path) as f:
        a_data = json.load(f)["annotations"]

    qid_to_question = {q["question_id"]: q["question"] for q in q_data}
    qid_to_imgid = {q["question_id"]: q["image_id"] for q in q_data}

    # Filter to samples with top-K answers
    filtered = []
    for ann in a_data:
        ans = ann["multiple_choice_answer"]
        if ans in ans_vocab:
            qid = ann["question_id"]
            filtered.append({
                "question_id": qid,
                "image_id": qid_to_imgid[qid],
                "question": qid_to_question[qid],
                "label": ans_vocab[ans],
            })

    print(f"  {split}: {len(filtered)}/{len(a_data)} samples "
          f"after top-K answer filter")

    image_ids = [s["image_id"] for s in filtered]
    questions = [s["question"] for s in filtered]
    labels = np.array([s["label"] for s in filtered], dtype=np.int64)

    # Image features
    img_feat_map = _extract_image_features(
        image_ids, img_dir, split, device, batch_size, cache_dir,
    )

    # Text features
    txt_cache = os.path.join(cache_dir, f"txt_feats_{split}.npy")
    txt_feats = _extract_text_features(questions, txt_cache, batch_size=512)

    # Build aligned arrays (skip samples whose image is missing)
    img_list, txt_list, lbl_list = [], [], []
    for i, sample in enumerate(filtered):
        iid = sample["image_id"]
        if iid in img_feat_map:
            img_list.append(img_feat_map[iid])
            txt_list.append(txt_feats[i])
            lbl_list.append(labels[i])

    print(f"  {split}: {len(img_list)} samples with valid images")
    return (
        np.stack(img_list).astype(np.float32),
        np.stack(txt_list).astype(np.float32),
        np.array(lbl_list, dtype=np.int64),
    )


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="VQA v2.0 → multimodal data.npz")
    p.add_argument("--vqa_root", type=str, required=True,
                   help="Path to vqa_v2/ directory")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--top_k", type=int, default=1000,
                   help="Number of most-frequent answers to keep as classes")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Batch size for image feature extraction")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    root = os.path.abspath(args.vqa_root)
    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)

    q_train = os.path.join(root, "Questions",
                           "v2_OpenEnded_mscoco_train2014_questions.json")
    q_val = os.path.join(root, "Questions",
                         "v2_OpenEnded_mscoco_val2014_questions.json")
    a_train = os.path.join(root, "Annotations",
                           "v2_mscoco_train2014_annotations.json")
    a_val = os.path.join(root, "Annotations",
                         "v2_mscoco_val2014_annotations.json")
    img_train = os.path.join(root, "Images", "mscoco", "train2014")
    img_val = os.path.join(root, "Images", "mscoco", "val2014")

    # Build answer vocabulary from training set
    print("Building answer vocabulary ...")
    with open(a_train) as f:
        train_anns = json.load(f)["annotations"]
    ans_vocab = _build_answer_vocab(train_anns, args.top_k)
    print(f"  Vocabulary: top-{args.top_k} answers")
    del train_anns

    # Save vocabulary for reference
    vocab_path = os.path.join(args.out_dir, "answer_vocab.json")
    inv_vocab = {v: k for k, v in ans_vocab.items()}
    with open(vocab_path, "w") as f:
        json.dump(inv_vocab, f, indent=2)
    print(f"  Saved answer vocab to {vocab_path}")

    # Process splits
    print("\n=== Training split ===")
    img_tr, txt_tr, lbl_tr = _process_split(
        q_train, a_train, img_train, "train2014",
        ans_vocab, args.device, args.batch_size, cache_dir,
    )

    print("\n=== Validation split (used as test) ===")
    img_te, txt_te, lbl_te = _process_split(
        q_val, a_val, img_val, "val2014",
        ans_vocab, args.device, args.batch_size, cache_dir,
    )

    # Save
    out_npz = os.path.join(args.out_dir, "data.npz")
    np.savez(
        out_npz,
        img_train=img_tr,
        txt_train=txt_tr,
        labels_train=lbl_tr,
        img_test=img_te,
        txt_test=txt_te,
        labels_test=lbl_te,
    )

    print(f"\nWrote {out_npz}")
    print(f"  img_train (ResNet-50)         {img_tr.shape}")
    print(f"  txt_train (MiniLM)            {txt_tr.shape}")
    print(f"  labels_train                  {lbl_tr.shape}  "
          f"({args.top_k} classes)")
    print(f"  img_test                      {img_te.shape}")
    print(f"  txt_test                      {txt_te.shape}")
    print(f"  labels_test                   {lbl_te.shape}")
    print(f"  img_dim={img_tr.shape[1]}, txt_dim={txt_tr.shape[1]}, "
          f"num_classes={args.top_k}")


if __name__ == "__main__":
    main()
