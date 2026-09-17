#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR with **Independent Compression**
#
# Multimodal fusion is ordinary summation of refined features:
#     fused = f'_acc + f'_gyro
# There is no sketch-based fusion (unlike SketchFusion A/B/C)
# and no cross-modal / missing-modality heads.
#
# The fused model is then trained with original FetchSGD
# (Rothchild et al.): one Count Sketch of the full gradient.
#
# Requires: datasets/uci_har_mm/data.npz (see prepare_uci_har_mm.py).
# Cache still uses img_*/txt_* keys; they are acc/gyro features.
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

python3 CommEfficient/CommEfficient/mm_train_independent.py \
  --dataset_dir datasets/uci_har_mm/ \
  --dataset_name MultiModal \
  --model IndependentCompression \
  --acc_dim 348 \
  --gyro_dim 213 \
  --feat_dim 512 \
  --mm_dropout 0.3 \
  --num_classes 6 \
  --dirichlet_alpha 0.1 \
  --skip_map \
  --mm_local_epochs 1 \
  --local_batch_size -1 \
  --local_momentum 0.0 \
  --virtual_momentum 0.9 \
  --error_type virtual \
  --mode sketch \
  --num_clients 10 \
  --num_devices 1 \
  --num_workers 10 \
  --share_ps_gpu \
  --k 20000 \
  --num_rows 3 \
  --num_cols 5000 \
  --lr_scale 0.05 \
  --pivot_epoch 20 \
  --num_blocks 1 \
  --num_epochs 85 \
  --device cuda
