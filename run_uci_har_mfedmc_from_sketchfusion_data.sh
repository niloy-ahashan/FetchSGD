#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# MFedMC on UCI HAR Acc/Gyro features, **ActionSense protocol**:
#   • clients = UCI HAR subjects (30 people, each with all 6 activities)
#   • per-client stratified 80/20 train/test
#   • local RF fusion scored on that client's hold-out (not the official test)
#   • local_epochs=5, client_select_ratio=0.2, top_shap=1
#
# Features still come from datasets/uci_har_mm/data.npz (same Acc/Gyro
# vectors as SketchFusionB). Subject IDs come from the original UCI folder.
#
# Official-test / Dirichlet-10 comparison (old path):
#   ./run_uci_har_mfedmc_from_sketchfusion_data.sh \
#     --partition dirichlet --eval_on_global_test --train_ratio 1.0 \
#     --num_clients 10 --dirichlet_alpha 0.1
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
UCI_HAR="${ROOT}/MFedMC/UCI_HAR"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/uci_har_mm}"
UCI_ROOT="${UCI_ROOT:-${ROOT}/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset}"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

cd "${UCI_HAR}"
exec "${PYTHON}" main.py \
  --dataset_dir "${DATA_PATH}" \
  --uci_root "${UCI_ROOT}" \
  --partition subject \
  --num_classes 6 \
  --acc_dim 348 \
  --gyro_dim 213 \
  --iterations 140 \
  --local_epochs 5 \
  --top_shap 1 \
  --client_select_ratio 0.2 \
  --train_ratio 0.8 \
  --class_non_iid_rate 1 \
  --no-eval_on_global_test \
  "$@"
