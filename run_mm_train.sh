#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# Multimodal FetchSGD — MIR-Flickr (24-class multi-label)
#
# Data:  PMFH/data/mir_cnn_twt.mat  (image 4096-d CNN, text 1386-d BoW)
#        5 000 train samples, 2 243 test samples, 24 multi-label classes
# Model: MultiModalNet  (PFMH-style feature fusion + classifier)
#        + MFM-style cross-modal predictors for missing modality handling
# Comm:  Sketch (Count-Sketch / Count-Median)
#
# Client partitioning uses Dirichlet (alpha=0.1), matching PMFH.
# 10 clients with 5 workers/round ≈ PMFH's num_users=10, frac=0.5.
#
# Wall-time / local compute (vs PMFH’s many local epochs × large batches):
#   • --mm_local_epochs  >1  → multiple FedSampler passes per display epoch
#   • --num_epochs       ↑   → more federated rounds
#   • --local_batch_size  N>0 (e.g. 32–128) → spe = ceil(N_train / (N * num_workers)),
#     more steps/epoch than local_batch_size=-1 (full client batch); tune for VRAM.
# Similarity-preserving loss (weight=0.01) from PFMH is added
# alongside BCE to directly optimise for retrieval quality (MAP).
# MFM missing-modality loss (weight=1.0) with 20% masking probability
# trains cross-modal predictors for robustness.
# ---------------------------------------------------------------

python3 CommEfficient/CommEfficient/mm_train.py \
  --dataset_dir PMFH/data/ \
  --dataset_name MultiModal \
  --model MultiModalNet \
  --img_dim 4096 \
  --txt_dim 1386 \
  --feat_dim 512 \
  --mm_dropout 0.3 \
  --num_classes 24 \
  --dirichlet_alpha 0.1 \
  --sim_loss_weight 0.01 \
  --missing_loss_weight 1.0 \
  --missing_prob 0.2 \
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
