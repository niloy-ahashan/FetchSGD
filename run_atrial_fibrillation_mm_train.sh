#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# Multimodal FetchSGD — Atrial Fibrillation (3-class ECG, UEA)
#
# Prepare data once:
#   python3 CommEfficient/CommEfficient/prepare_atrial_fibrillation_mm.py \
#     --ts_dir datasets/AtrialFibrillation \
#     --out_dir datasets/atrial_fibrillation_mm
#
# Modality split: series 0 → img (640); series 1 → txt (640).
# Same training recipe as Spoken Arabic Digits (feat_dim, sketch, epochs).
#
# The UEA split is tiny (~15 train / ~15 test): use num_workers=1 and few
# clients so mm_train does not skip most batches (see Heartbeat scripts).
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

python3 CommEfficient/CommEfficient/mm_train.py \
  --dataset_dir datasets/atrial_fibrillation_mm/ \
  --dataset_name MultiModal \
  --model MultiModalNet \
  --img_dim 640 \
  --txt_dim 640 \
  --feat_dim 512 \
  --mm_dropout 0.3 \
  --num_classes 3 \
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
  --num_clients 3 \
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
