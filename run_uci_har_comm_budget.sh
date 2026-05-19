#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR — test accuracy under a **total training communication**
# budget (MiB), in the spirit of MFedMC Table 2(i) “Accuracy under
# 5 MB Communication Constraint” (MFedMC paper / arXiv:2401.16685).
#
# This script uses FetchSGD’s own accounting from fed_aggregator:
#   per training batch:  sum over clients of (download_bytes + upload_bytes)
#   sketch upload ≈ num_rows * num_cols * 4 bytes per participating client
#   (see CommEfficient/CommEfficient/fed_aggregator.py).
#
# **Important:** With large sketches (e.g. num_cols=600000), a **single**
# federated batch can exceed 5 MiB. You will still get a valid “accuracy
# at the first budget crossing,” but to approximate many small steps under
# 5 MiB total, reduce --num_cols / --num_rows / --k or use --mode fedavg
# with a small model — see paper/supplement for their exact compressor.
#
# Usage:
#   COMM_MB=5 bash run_uci_har_comm_budget.sh
#
# For Table 2(ii)-style “MiB to reach target accuracy”, see
#   run_uci_har_mm_comm_overhead.sh  (--target_test_acc / comm_cum_mib).
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

COMM_MB="${COMM_MB:-5}"

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
  --max_comm_megabytes "${COMM_MB}" \
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
  --k 500 \
  --num_rows 3 \
  --num_cols 600 \
  --lr_scale 0.1 \
  --pivot_epoch 15 \
  --num_blocks 1 \
  --num_epochs 50 \
  --device cuda
