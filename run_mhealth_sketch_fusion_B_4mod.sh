#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# mHealth (4 modalities: Acc/Gyro/Mag/ECG) + SketchFusionB4 + FetchSGD sketch
#
# Modalities (see prepare_mhealth_mm.py):
#   m0 Acc (chest+ankle+arm accel) | m1 Gyro (ankle+arm) |
#   m2 Mag (ankle+arm) | m3 ECG (chest 2-lead)
#
# This is the mHealth sibling of run_actionsense_s00_sketch_fusion_B_4mod.sh
# (same mm_train_actionsense_4mod.py + SketchFusionB4 pipeline — no code
# changes needed, only a new dataset dir), with hyperparameters mirroring
# run_uci_har_sketch_fusion_B.sh for closest parity with the UCI HAR configs.
#
# Prep (writes datasets/mhealth_mm/data.npz + prepare_stats.json):
#   python3 CommEfficient/CommEfficient/prepare_mhealth_mm.py
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

OUT_DIR="${ROOT}/datasets/mhealth_mm"
if [[ ! -f "${OUT_DIR}/data.npz" ]]; then
  echo "Missing ${OUT_DIR}/data.npz — run prepare_mhealth_mm.py first:" >&2
  echo "  python3 CommEfficient/CommEfficient/prepare_mhealth_mm.py" >&2
  exit 1
fi

python3 CommEfficient/CommEfficient/mm_train_actionsense_4mod.py \
  --dataset_dir "${OUT_DIR}/" \
  --dataset_name MultiModal \
  --model SketchFusionB4 \
  --feat_dim 512 \
  --sketch_r 4 \
  --sketch_c 128 \
  --mm_dropout 0.3 \
  --num_classes 12 \
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
  --num_cols 1000 \
  --lr_scale 0.01 \
  --pivot_epoch 10 \
  --num_blocks 1 \
  --num_epochs 19 \
  --device cuda
