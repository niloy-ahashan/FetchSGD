#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# Multimodal FetchSGD — MELD (7-class emotion recognition)
#
# Prepare data once:
#   python3 CommEfficient/CommEfficient/prepare_meld.py \
#     --meld_root datasets/MELD.Raw \
#     --out_dir   datasets/meld_mm \
#     --merge_dev_into_train
#
# Modalities:
#   img branch (audio features, 378-dim) — MFCCs, chroma, mel-spec, etc.
#   txt branch (text features, 384-dim)  — sentence-transformers embeddings
#
# Gradient compression: standard FetchSGD sketch (single sketch for all params)
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

python3 CommEfficient/CommEfficient/mm_train.py \
  --dataset_dir datasets/meld_mm/ \
  --dataset_name MultiModal \
  --model MultiModalNet \
  --img_dim 378 \
  --txt_dim 384 \
  --feat_dim 512 \
  --mm_dropout 0.3 \
  --num_classes 7 \
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
  --k 50000 \
  --num_rows 3 \
  --num_cols 10000 \
  --lr_scale 0.1 \
  --pivot_epoch 15 \
  --num_blocks 1 \
  --num_epochs 50 \
  --device cuda
