#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# MFedMC on mHealth Acc/Gyro/Mag/ECG features, **ActionSense-style settings**:
#   • clients = 10-client Dirichlet(alpha=0.1) partition
#     (mHealth has no subject-partition loader yet — unlike its UCI HAR
#     sibling run_uci_har_mfedmc_from_sketchfusion_data.sh, which defaults to
#     --partition subject, this script and
#     run_mhealth_mfedmc_sketchfusion_protocol.sh both use --partition
#     dirichlet; subject-based clients, one per mHealth subject, are a
#     possible follow-up)
#   • per-client stratified 80/20 train/test
#   • local RF fusion scored on that client's hold-out (not the official test)
#   • local_epochs=5, client_select_ratio=0.2, top_shap=1
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
  --iterations 100 \
  --local_epochs 5 \
  --top_shap 2 \
  --client_select_ratio 0.2 \
  --train_ratio 0.8 \
  --no-eval_on_global_test \
  "$@"
