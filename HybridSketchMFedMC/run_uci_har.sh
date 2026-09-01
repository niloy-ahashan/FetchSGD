#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR hybrid: SketchFusionB fusion + FetchSGD gradient sketch
# + MFedMC client/modality selection.
#
# Extra args are forwarded to main.py, e.g.:
#   ./HybridSketchMFedMC/run_uci_har.sh --client_select random
#   ./HybridSketchMFedMC/run_uci_har.sh --num_select_modalities 1
#   ./HybridSketchMFedMC/run_uci_har.sh --fusion_mode sum
#   (sum = IndependentCompression additive fusion; sketch = SketchFusionB)
#
# Requires: datasets/uci_har_mm/data.npz
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
  --sketch_r 2 \
  --sketch_c 128 \
  --mm_dropout 0.3 \
  --num_epochs 100 \
  --local_epochs 10 \
  --local_batch_size -1 \
  --virtual_momentum 0.9 \
  --error_type virtual \
  --mode sketch \
  --k 20000 \
  --num_rows 3 \
  --num_cols 50000 \
  --lr_scale 0.1 \
  --pivot_epoch 15 \
  --num_blocks 1 \
  --fusion_mode sketch \
  --client_select loss \
  --client_select_ratio 0.2 \
  --num_select_modalities 1 \
  --device cuda \
  "$@"
