#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR — SketchFusionB fusion + MFedMC-style modality encoders (2401.16685v2)
#
# Paper settings referenced for encoders & local training:
#   • Each modality reshaped as time × features; single LSTM (128 hidden)
#     + fully connected stack (implemented as LSTM + 2-layer MLP to feat_dim).
#   • Optimizer: SGD with η = 0.1 (here: peak --lr_scale 0.1 with the usual
#     FetchSGD PiecewiseLinear schedule on optimizer steps; adjust --pivot_epoch
#     / --num_epochs if you need a flatter schedule).
#   • Batch size 32, local epochs E = 5, cross-entropy (handled in mm_train).
#
# Data: raw inertial signals → (N, 128, 3) per modality (acc vs gyro).
# Prepare once (--uci_root = inner folder with train/Inertial Signals/):
#   python3 CommEfficient/CommEfficient/prepare_uci_har_lstm_mm.py \
#     --uci_root "human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset" \
#     --out_dir "datasets/uci_har_lstm_mm"
#
# Clearing old vector-based client caches if you switch to this dataset:
#   rm -f datasets/uci_har_lstm_mm/client*.npz datasets/uci_har_lstm_mm/test.npz \
#         datasets/uci_har_lstm_mm/stats.json
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

python3 CommEfficient/CommEfficient/mm_train.py \
  --dataset_dir datasets/uci_har_lstm_mm/ \
  --dataset_name MultiModal \
  --model SketchFusionBLSTM \
  --img_dim 3 \
  --txt_dim 3 \
  --lstm_hidden 128 \
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
  --pivot_epoch 10 \
  --num_blocks 1 \
  --num_epochs 20 \
  --device cuda
