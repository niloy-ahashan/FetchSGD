#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# Multimodal FetchSGD — VQA v2.0
#
# Two modalities (Visual Question Answering):
#   img branch: ResNet-50 image features  — 2048-dim
#   txt branch: MiniLM question embeddings — 384-dim
#   Labels: top-1000 most frequent answers (single-label)
#
# Prepare data once:
#   python3 CommEfficient/CommEfficient/prepare_vqa.py \
#     --vqa_root datasets/vqa_v2 --out_dir datasets/vqa_mm \
#     --top_k 1000 --batch_size 256
#
# Model: MultiModalNet (concatenation fusion)
# Gradient compression: standard FetchSGD sketch
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

python3 CommEfficient/CommEfficient/mm_train.py \
  --dataset_dir datasets/vqa_mm/ \
  --dataset_name MultiModal \
  --model MultiModalNet \
  --img_dim 2048 \
  --txt_dim 384 \
  --feat_dim 512 \
  --mm_dropout 0.3 \
  --num_classes 1000 \
  --dirichlet_alpha 0.5 \
  --sim_loss_weight 0.0 \
  --missing_loss_weight 0.0 \
  --missing_prob 0.0 \
  --skip_map \
  --mm_local_epochs 5 \
  --local_batch_size -1 \
  --local_momentum 0.0 \
  --virtual_momentum 0.9 \
  --error_type virtual \
  --mode sketch \
  --num_clients 20 \
  --num_devices 1 \
  --num_workers 5 \
  --share_ps_gpu \
  --k 50000 \
  --num_rows 3 \
  --num_cols 10000 \
  --lr_scale 0.1 \
  --pivot_epoch 15 \
  --num_blocks 1 \
  --num_epochs 40 \
  --device cuda
