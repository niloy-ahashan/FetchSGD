#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# Multimodal FetchSGD — Heartbeat (binary heart-sound spectrograms)
#
# Prepare data once:
#   python3 CommEfficient/CommEfficient/prepare_heartbeat_mm.py \
#     --ts_dir datasets/Heartbeat \
#     --out_dir datasets/heartbeat_mm
#
# Modality split: spectrogram bands 0–29 → img (30×405=12150);
# bands 30–60 → txt (31×405=12555). Classes: normal vs abnormal.
#
# Tiny train set (~204): use num_workers=1 so mm_train’s check
# (unique clients in batch >= num_workers with local_batch_size=-1)
# matches FedSampler — with num_workers=5 most batches were skipped
# (SKIPPING BATCH: NOT ENOUGH CLIENTS) and training went NaN. Fewer
# clients + milder Dirichlet reduces empty/skewed shards.
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

python3 CommEfficient/CommEfficient/mm_train.py \
  --dataset_dir datasets/heartbeat_mm/ \
  --dataset_name MultiModal \
  --model MultiModalNet \
  --img_dim 12150 \
  --txt_dim 12555 \
  --feat_dim 512 \
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
