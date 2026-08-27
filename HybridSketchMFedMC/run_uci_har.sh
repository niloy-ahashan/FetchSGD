#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR hybrid: SketchFusionB fusion + FetchSGD gradient sketch
# + MFedMC client/modality selection.
#
# All code lives in HybridSketchMFedMC/ (does not modify CommEfficient
# or MFedMC). Extra args are forwarded to main.py, e.g.:
#   ./HybridSketchMFedMC/run_uci_har.sh --client_select random
#   ./HybridSketchMFedMC/run_uci_har.sh --num_select_modalities 1
#   ./HybridSketchMFedMC/run_uci_har.sh --prefer-higher-loss
#
# Requires: datasets/uci_har_mm/data.npz
#   python3 CommEfficient/CommEfficient/prepare_uci_har_mm.py \
#     --uci_root "human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset" \
#     --out_dir "datasets/uci_har_mm"
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/uci_har_mm}"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

cd "${ROOT}"
exec "${PYTHON}" "${HERE}/main.py" \
  --dataset_dir "${DATA_PATH}" \
  --num_classes 6 \
  --num_clients 10 \
  --dirichlet_alpha 0.1 \
  --acc_dim 348 \
  --gyro_dim 213 \
  --feat_dim 512 \
  --sketch_r 4 \
  --sketch_c 128 \
  --mm_dropout 0.3 \
  --num_epochs 70 \
  --local_epochs 5 \
  --local_batch_size -1 \
  --virtual_momentum 0.9 \
  --error_type virtual \
  --mode sketch \
  --k 20000 \
  --num_rows 3 \
  --num_cols 10000 \
  --lr_scale 0.1 \
  --pivot_epoch 20 \
  --num_blocks 1 \
  # --client_select loss \
  --client_select random \
  # --client_select_ratio 1 \
  --num_select_modalities 2 \
  --device cuda \
  "$@"
