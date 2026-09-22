#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# MFedMC v2 on mHealth Acc/Gyro/Mag/ECG features, SketchFusionB4 protocol,
# IC-style per-modality encoder (Level 2):
#   raw Acc/Gyro/Mag/ECG -> Extractor (in_dim->feat_dim->feat_dim, IC-shaped)
#                         -> Refiner (sigmoid gate, same as IC; toggle with
#                            --no-use_refiner)
#                         -> small classifier (log-probs)
# No shared/joint fusion is added — each modality is still trained and
# uploaded independently; MFedMC's SHAP-driven selection + RandomForest
# late fusion in federated.py are unchanged from v1.
#
#   • clients = 10-client Dirichlet(alpha=0.1) partition
#   • all of each client's data is training data (no local hold-out)
#   • fusion scored on the official shared mHealth test set
#   • local_epochs=10, client_select_ratio=0.2, top_shap=2
#
# Sibling of run_mhealth_mfedmc_sketchfusion_protocol.sh, pointed at
# MFedMC/mHealth_v2 instead of MFedMC/mHealth (v1). Self-contained; does
# not modify MFedMC/mHealth (v1) or its run script.
#
# Features come from datasets/mhealth_mm/data.npz (see prepare_mhealth_mm.py).
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
MHEALTH="${ROOT}/MFedMC/mHealth_v2"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/mhealth_mm_random_group}"

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
  --feat_dim 512 \
  --use_refiner \
  --iterations 10 \
  --local_epochs 5 \
  --top_shap 2 \
  --client_select_ratio 0.2 \
  --train_ratio 1.0 \
  --eval_on_global_test \
  "$@"
