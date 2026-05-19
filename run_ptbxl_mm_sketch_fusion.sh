#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# PTB-XL with **two-way sketch fusion** (FetchSGD sketch mode)
#
# Modalities (following MFedMC):
#   img branch: Limb Lead ECG   (I, II, III, aVR, aVL, aVF)  — 6000-dim
#   txt branch: Precordial Lead  (V1–V6)                      — 6000-dim
#   Labels: NORM, MI, STTC, CD, HYP (multi-label)
#
# Requires: datasets/ptbxl_mm/data.npz (see prepare_ptbxl.py).
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

python3 CommEfficient/CommEfficient/mm_train.py \
  --dataset_dir datasets/ptbxl_mm/ \
  --dataset_name MultiModal \
  --model MultiModalNet \
  --img_dim 6000 \
  --txt_dim 6000 \
  --feat_dim 512 \
  --mm_dropout 0.3 \
  --num_classes 5 \
  --dirichlet_alpha 0.1 \
  --sim_loss_weight 0.0 \
  --missing_loss_weight 0.0 \
  --missing_prob 0.0 \
  --skip_map \
  --mm_sketch_fusion \
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
  --num_rows 5 \
  --num_cols 600000 \
  --lr_scale 0.1 \
  --pivot_epoch 15 \
  --num_blocks 1 \
  --num_epochs 50 \
  --device cuda
