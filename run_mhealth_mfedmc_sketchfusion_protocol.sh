#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# MFedMC on mHealth Acc/Gyro/Mag/ECG features, **SketchFusionB4 protocol**:
#   • clients = 10-client Dirichlet(alpha=0.1) partition
#   • all of each client's data is training data (no local hold-out)
#   • fusion scored on the official shared mHealth test set
#   • local_epochs=5, client_select_ratio=0.5, top_shap=1
#
# This is the sibling of run_mhealth_mfedmc_from_sketchfusion_data.sh's
# ActionSense-style default, so the two are directly comparable to
# SketchFusionB4 / HybridSketchMFedMC (same partition + same eval split).
# Both mHealth scripts use --partition dirichlet (see that script's header
# comment for why, unlike the UCI HAR pair).
#
# Features come from datasets/mhealth_mm/data.npz (see prepare_mhealth_mm.py).
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
MHEALTH="${ROOT}/MFedMC/mHealth"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/mhealth_mm}"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

cd "${MHEALTH}"
exec "${PYTHON}" main.py \
  --dataset_dir "${DATA_PATH}" \
  --partition dirichlet \
  --num_clients 10 \
  --dirichlet_alpha 0.1 \
  --num_classes 12 \
  --acc_dim 177 \
  --gyro_dim 118 \
  --mag_dim 118 \
  --ecg_dim 43 \
  --iterations 200 \
  --local_epochs 10 \
  --top_shap 2 \
  --client_select_ratio 0.2 \
  --train_ratio 1.0 \
  --eval_on_global_test \
  "$@"
