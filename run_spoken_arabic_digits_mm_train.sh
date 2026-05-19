#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# Multimodal FetchSGD — Spoken Arabic Digits (10-class speech)
#
# Prepare data once:
#   python3 CommEfficient/CommEfficient/prepare_spoken_arabic_digits_mm.py \
#     --ts_dir datasets/SpokenArabicDigits \
#     --out_dir datasets/spoken_arabic_digits_mm
#
# Modality split: MFCC dims 0–5 → img (65×6=390); dims 6–12 → txt (65×7=455).
# MAP retrieval is disabled (--skip_map); CrossEntropy on fused logits.
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

python3 CommEfficient/CommEfficient/mm_train.py \
  --dataset_dir datasets/spoken_arabic_digits_mm/ \
  --dataset_name MultiModal \
  --model MultiModalNet \
  --img_dim 390 \
  --txt_dim 455 \
  --feat_dim 512 \
  --mm_dropout 0.3 \
  --num_classes 10 \
  --dirichlet_alpha 0.1 \
  --sim_loss_weight 0.0 \
  --missing_loss_weight 0.0 \
  --missing_prob 0.0 \
  --skip_map \
  --mm_local_epochs 10 \
  --local_batch_size -1 \
  --local_momentum 0.0 \
  --virtual_momentum 0.9 \
  --error_type virtual \
  --mode sketch \
  --num_clients 10 \
  --num_devices 1 \
  --num_workers 5 \
  --share_ps_gpu \
  --k 20000 \
  --num_rows 3 \
  --num_cols 5000 \
  --lr_scale 0.1 \
  --pivot_epoch 15 \
  --num_blocks 1 \
  --num_epochs 40 \
  --device cuda
