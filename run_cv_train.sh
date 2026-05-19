#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to run cv_train.py with the same flags you provided.
# Run from workspace root or call this script directly.

python3 CommEfficient/CommEfficient/cv_train.py \
  --dataset_dir ~/datasets/cifar10/ \
  --dataset_name CIFAR10 \
  --model FixupResNet9 \
  --local_batch_size 5 \
  --local_momentum 0.0 \
  --virtual_momentum 0.9 \
  --error_type virtual \
  --num_clients 10000 \
  --num_devices 1 \
  --num_workers 100 \
  --share_ps_gpu \
  --mode sketch \
  --k 10000 \
  --num_rows 5 \
  --num_cols 600000 \
  --lr_scale 0.06 \
  --num_blocks 1