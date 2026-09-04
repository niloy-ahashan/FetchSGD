#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# MFedMC on UCI HAR Acc/Gyro features, **SketchFusionB protocol**:
#   • clients = 10-client Dirichlet(alpha=0.1) partition (not subjects)
#   • all of each client's data is training data (no local hold-out)
#   • fusion scored on the official shared UCI HAR test set
#   • local_epochs=5, client_select_ratio=0.2, top_shap=1
#
# This is the sibling of run_uci_har_mfedmc_from_sketchfusion_data.sh's
# ActionSense-protocol default, using the override flags documented
# there so the two are directly comparable to SketchFusionB /
# HybridSketchMFedMC (same partition + same eval split).
#
# Features come from datasets/uci_har_mm/data.npz (same Acc/Gyro
# vectors as SketchFusionB). --uci_root is not needed here since the
# dirichlet partition doesn't use per-subject IDs.
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
UCI_HAR="${ROOT}/MFedMC/UCI_HAR"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/uci_har_mm}"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

cd "${UCI_HAR}"
exec "${PYTHON}" main.py \
  --dataset_dir "${DATA_PATH}" \
  --partition dirichlet \
  --num_clients 10 \
  --dirichlet_alpha 0.1 \
  --num_classes 6 \
  --acc_dim 348 \
  --gyro_dim 213 \
  --iterations 170 \
  --local_epochs 5 \
  --top_shap 1 \
  --client_select_ratio 0.5 \
  --train_ratio 1.0 \
  --eval_on_global_test \
  "$@"
