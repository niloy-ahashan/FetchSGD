#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR with **SketchFusionB** (Fixed Count Sketch + MLP Head)
#
# Uses fixed random Count Sketch hash functions to map each
# modality's refined features into a shared table.  The sketch
# operation is differentiable (scatter_add), so gradients flow
# through it.  A nonlinear MLP head decodes the fused sketch.
#
# Gradient compression still uses standard FetchSGD sketch mode.
#
# Requires: datasets/uci_har_mm/data.npz (see prepare_uci_har_mm.py).
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

python3 CommEfficient/CommEfficient/mm_train.py \
  --dataset_dir datasets/uci_har_mm/ \
  --dataset_name MultiModal \
  --model SketchFusionB \
  --img_dim 348 \
  --txt_dim 213 \
  --feat_dim 512 \
  --sketch_r 4 \
  --sketch_c 128 \
  --mm_dropout 0.3 \
  --num_classes 6 \
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
  --pivot_epoch 10\
  --num_blocks 1 \
  --num_epochs 19 \
  --device cuda
