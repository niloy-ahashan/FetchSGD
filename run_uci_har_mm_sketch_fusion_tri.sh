#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR — **three-way sketch fusion** (FetchSGD sketch mode):
#
#   1) Sketch gradient for **img** branch (img_extractor + img_refiner)
#   2) Sketch gradient for **txt** branch (txt_extractor + txt_refiner)
#   3) Sketch gradient for **fusion head** (img_to_txt, txt_to_img,
#      integrator, classifier) — i.e. everything that mixes / concatenates
#      modalities after refinement
#
# Tables are added: t_img + t_txt + t_fusion → same shape as standard
# sketch upload; server path unchanged.
#
# Compare:
#   run_uci_har_mm_train.sh           — single sketch of full grad
#   run_uci_har_mm_sketch_fusion.sh   — --mm_sketch_fusion (two-way)
#   this script                       — --mm_sketch_fusion_tri
#
# Requires: datasets/uci_har_mm/data.npz
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
  --mm_sketch_fusion_tri \
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
  --num_cols 60000 \
  --lr_scale 0.1 \
  --pivot_epoch 15 \
  --num_blocks 1 \
  --num_epochs 100 \
  --device cuda
