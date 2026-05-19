#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# MELD (paper features) + **SketchFusionB** (Fixed Count Sketch + MLP)
#
# Features follow the original MELD paper (Poria et al., 2019):
#   img branch (audio): openSMILE ComParE → selection → 300-dim
#   txt branch (text):  GloVe 300d mean word embeddings → 300-dim
#
# Model-level fusion: fixed random Count Sketch hash functions map
# each modality's features into a shared table (differentiable
# scatter_add). MLP head decodes the fused sketch.
#
# Gradient compression: standard FetchSGD sketch.
#
# Requires: datasets/meld_paper/data.npz (see prepare_meld_paper.py).
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

python3 CommEfficient/CommEfficient/mm_train.py \
  --dataset_dir datasets/meld_paper/ \
  --dataset_name MultiModal \
  --model SketchFusionB \
  --img_dim 300 \
  --txt_dim 300 \
  --feat_dim 512 \
  --sketch_r 4 \
  --sketch_c 128 \
  --mm_dropout 0.3 \
  --num_classes 7 \
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
  --k 50000 \
  --num_rows 3 \
  --num_cols 10000 \
  --lr_scale 0.1 \
  --pivot_epoch 15 \
  --num_blocks 1 \
  --num_epochs 50 \
  --device cuda
