#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# Heartbeat with **SketchFusionB** (Fixed Count Sketch + MLP Head)
#
# Requires: datasets/heartbeat_mm/data.npz (see prepare_heartbeat_mm.py).
#
# ~204 train samples: num_workers=1, num_clients=5, dirichlet_alpha=1.0
# — see run_heartbeat_mm_train.sh header for why (avoid batch skips / NaN).
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

python3 CommEfficient/CommEfficient/mm_train.py \
  --dataset_dir datasets/heartbeat_mm/ \
  --dataset_name MultiModal \
  --model SketchFusionB \
  --img_dim 12150 \
  --txt_dim 12555 \
  --feat_dim 512 \
  --sketch_r 4 \
  --sketch_c 128 \
  --mm_dropout 0.3 \
  --num_classes 2 \
  --dirichlet_alpha 1.0 \
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
  --num_clients 5 \
  --num_devices 1 \
  --num_workers 1 \
  --share_ps_gpu \
  --k 20000 \
  --num_rows 3 \
  --num_cols 5000 \
  --lr_scale 0.1 \
  --pivot_epoch 15 \
  --num_blocks 1 \
  --num_epochs 40 \
  --device cuda
