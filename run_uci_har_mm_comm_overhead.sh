#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR — MFedMC-style **communication metrics** (see paper Table 2):
#
#  (i) Accuracy under a fixed comm budget → use run_uci_har_comm_budget.sh
#      with COMM_MB=5 (or --max_comm_megabytes in mm_train.py).
#
#  (ii) Communication overhead (MiB) to reach a **target test accuracy** —
#      use --target_test_acc below. When test_acc first crosses the
#      threshold, the run prints cumulative train download+upload (MiB)
#      using FetchSGD’s fed_aggregator accounting (not ActionSense’s code;
#      the public MFedMC repo does not ship UCI HAR).
#
# Optional: --stop_on_target_acc stops training once the target is hit
# (saves time when you only care about overhead to that accuracy).
#
# Per-epoch logs include column **comm_cum_mib** (cumulative MiB).
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

# Target accuracy as a fraction (e.g. 0.95 = 95%). Tune for your task.
TARGET_ACC="${TARGET_ACC:-0.95}"

python3 CommEfficient/CommEfficient/mm_train.py \
  --dataset_dir datasets/uci_har_mm/ \
  --dataset_name MultiModal \
  --model MultiModalNet \
  --img_dim 348 \
  --txt_dim 213 \
  --feat_dim 512 \
  --mm_dropout 0.3 \
  --num_classes 6 \
  --dirichlet_alpha 0.1 \
  --sim_loss_weight 0.0 \
  --missing_loss_weight 0.0 \
  --missing_prob 0.0 \
  --skip_map \
  --target_test_acc "${TARGET_ACC}" \
  --stop_on_target_acc \
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
  --num_epochs 100 \
  --device cuda
