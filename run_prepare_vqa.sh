#!/usr/bin/env bash
set -euo pipefail

# Prepare VQA v2.0 → datasets/vqa_mm/data.npz
# Uses ResNet-50 for image features, MiniLM for question embeddings
# Top-1000 most frequent answers as classes

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "$ROOT/venv/bin/python" -u CommEfficient/CommEfficient/prepare_vqa.py \
  --vqa_root  datasets/vqa_v2 \
  --out_dir   datasets/vqa_mm \
  --top_k     1000 \
  --batch_size 128 \
  --device    cuda
