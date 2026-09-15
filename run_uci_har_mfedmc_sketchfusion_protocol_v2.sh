#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# MFedMC v2 on UCI HAR Acc/Gyro features, SketchFusionB protocol,
# IC-style per-modality encoder (Level 2):
#   raw Acc/Gyro -> Extractor (in_dim->feat_dim->feat_dim, IC-shaped)
#                -> Refiner (sigmoid gate, same as IC; toggle with
#                   --no-use_refiner)
#                -> small classifier (log-probs)
# No shared/joint fusion is added — each modality is still trained and
# uploaded independently; MFedMC's SHAP-driven selection + RandomForest
# late fusion in federated.py are unchanged from v1.
#
#   • clients = 10-client Dirichlet(alpha=0.1) partition (not subjects)
#   • all of each client's data is training data (no local hold-out)
#   • fusion scored on the official shared UCI HAR test set
#   • local_epochs=5, client_select_ratio=0.5, top_shap=1
#
# Sibling of run_uci_har_mfedmc_sketchfusion_protocol.sh, pointed at
# MFedMC/UCI_HAR_v2 instead of MFedMC/UCI_HAR (v1). Self-contained; does
# not modify MFedMC/UCI_HAR (v1) or its run script.
#
# Features come from datasets/uci_har_mm/data.npz (same Acc/Gyro vectors
# as SketchFusionB). --uci_root is not needed here since the dirichlet
# partition doesn't use per-subject IDs.
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
UCI_HAR="${ROOT}/MFedMC/UCI_HAR_v2"
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
  --feat_dim 512 \
  --use_refiner \
  --iterations 3 \
  --local_epochs 5 \
  --top_shap 1 \
  --client_select_ratio 0.2 \
  --train_ratio 1.0 \
  --eval_on_global_test \
  "$@"
