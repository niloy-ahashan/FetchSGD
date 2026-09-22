#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# mHealth (4 modalities: Acc/Gyro/Mag/ECG) with **Independent Compression**
#
# Multimodal fusion is ordinary summation of refined features:
#     fused = sum(f'_Acc, f'_Gyro, f'_Mag, f'_ECG)
# There is no sketch-based fusion (unlike SketchFusionB4)
# and no cross-modal / missing-modality heads or similarity loss.
#
# The fused model is then trained with original FetchSGD
# (Rothchild et al.): one Count Sketch of the full gradient.
#
# This is the 4-modality sibling of run_uci_har_independent_compression.sh,
# using models/independent_compression_net_4.py::IndependentCompression4 +
# mm_train_independent_4mod.py (mirrors how mm_train_actionsense_4mod.py is
# the 4-modality sibling of mm_train.py).
#
# Requires: <dir>/data.npz (see prepare_mhealth_mm.py). Dims and
# num_classes come from prepare_stats.json, not CLI flags.
#
# prepare_mhealth_mm.py's --test_split toggles between the default fixed
# subject-1-8-train/9-10-test holdout ("subject") and a UCI-HAR-style
# pooled, class-stratified random split ("random") — see its own header
# comment. Point this script at whichever prepared cache you want with
# DATA_PATH, e.g. after building a second cache in its own directory:
#   python3 CommEfficient/CommEfficient/prepare_mhealth_mm.py \
#     --out_dir datasets/mhealth_mm_random --test_split random
#   DATA_PATH=datasets/mhealth_mm_random bash run_mhealth_independent_compression_4mod.sh
# Both caches then coexist on disk so either can be tried without
# overwriting the other.
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

OUT_DIR="${DATA_PATH:-${ROOT}/datasets/mhealth_mm_random_group}"
if [[ ! -f "${OUT_DIR}/data.npz" ]]; then
  echo "Missing ${OUT_DIR}/data.npz — run prepare_mhealth_mm.py first:" >&2
  echo "  python3 CommEfficient/CommEfficient/prepare_mhealth_mm.py" >&2
  exit 1
fi

python3 CommEfficient/CommEfficient/mm_train_independent_4mod.py \
  --dataset_dir "${OUT_DIR}/" \
  --dataset_name MultiModal \
  --model IndependentCompression4 \
  --feat_dim 512 \
  --mm_dropout 0.3 \
  --dirichlet_alpha 0.1 \
  --skip_map \
  --mm_local_epochs 1 \
  --local_batch_size -1 \
  --local_momentum 0.0 \
  --virtual_momentum 0.9 \
  --error_type virtual \
  --mode sketch \
  --num_clients 10 \
  --num_devices 1 \
  --num_workers 10 \
  --share_ps_gpu \
  --k 20000 \
  --num_rows 3 \
  --num_cols 5000 \
  --lr_scale 0.05 \
  --pivot_epoch 15 \
  --num_blocks 1 \
  --num_epochs 42 \
  --device cuda
