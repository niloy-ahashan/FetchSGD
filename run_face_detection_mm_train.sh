#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# Multimodal FetchSGD — Face Detection (binary, UEA)
#
# Prepare data once:
#   python3 CommEfficient/CommEfficient/prepare_face_detection_mm.py \
#     --ts_dir datasets/FaceDetection \
#     --out_dir datasets/face_detection_mm
#
# Modality split: series dims 0–71 → img (62×72=4464); 72–143 → txt (4464).
# MAP retrieval is disabled (--skip_map); CrossEntropy on fused logits.
#
# With local_batch_size=-1, mm_train requires unique clients per batch ≥ num_workers.
# dirichlet_alpha=0.1 + 10 clients produced empty clients and batches with only 3
# active clients → SKIPPING BATCH / NaN. Milder Dirichlet, fewer clients, and
# num_workers=1 avoids that (same idea as run_heartbeat_mm_train.sh).
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

python3 CommEfficient/CommEfficient/mm_train.py \
  --dataset_dir datasets/face_detection_mm/ \
  --dataset_name MultiModal \
  --model MultiModalNet \
  --img_dim 4464 \
  --txt_dim 4464 \
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
