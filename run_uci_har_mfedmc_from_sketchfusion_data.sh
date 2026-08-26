#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# MFedMC on the **same UCI HAR split** as SketchFusionB
# (run_uci_har_sketch_fusion_B.sh).
#
# Data: datasets/uci_har_mm/  (Acc 348-D + Gyro 213-D from prepare_uci_har_mm.py)
# If client0.npz … client9.npz exist, they are reused so the 10-client
# Dirichlet partition matches SketchFusionB exactly.
#
# This does **not** modify ActionSense/main.py.  It runs MFedMC/UCI_HAR/main.py:
#   • two MLP modality encoders (Acc, Gyro) + local RF fusion
#   • joint modality + client selection
#   • logs fusion accuracy, uplink bytes, and selection frequency
#
# Prepare data once (same as SketchFusionB):
#   python3 CommEfficient/CommEfficient/prepare_uci_har_mm.py \
#     --uci_root "human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset" \
#     --out_dir "datasets/uci_har_mm"
#
# Extra args are forwarded to main.py, e.g.:
#   ./run_uci_har_mfedmc_from_sketchfusion_data.sh --iterations 50 --top_shap 1
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
  --num_classes 6 \
  --num_clients 10 \
  --dirichlet_alpha 0.1 \
  --acc_dim 348 \
  --gyro_dim 213 \
  --iterations 50 \
  --local_epochs 5 \
  --top_shap 0 \
  --client_select_ratio 1 \
  --eval_on_global_test \
  "$@"
