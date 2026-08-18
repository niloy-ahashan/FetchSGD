#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# Multimodal FetchSGD — UCI HAR (6-class activity recognition)
#
# Prepare data once (acc vs gyro split from features.txt):
#   python3 CommEfficient/CommEfficient/prepare_uci_har_mm.py \
#     --uci_root "human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset" \
#     --out_dir "datasets/uci_har_mm"
#
# Modality split: columns whose feature name contains "Gyro" → txt branch;
# the rest (accelerometer + gravity + angles, etc.) → img branch.
# MAP retrieval is disabled (--skip_map); use CrossEntropy on fused logits.
#
# Note: [MFedMC](https://github.com/liangqiyuan/MFedMC) structures separate
# modality streams per dataset; we map UCI HAR’s fixed feature vector to two
# streams analogously (sensor-type split rather than image/text).
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

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
  --pivot_epoch 10 \
  --num_blocks 1 \
  --num_epochs 20 \
  --device cuda
