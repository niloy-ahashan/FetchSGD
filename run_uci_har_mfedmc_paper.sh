#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# MFedMC PAPER REPRODUCTION on UCI HAR — raw Inertial Signals, LSTM encoders,
# natural 30-subject clients. This is NOT the same protocol as
# run_uci_har_mfedmc_sketchfusion_protocol_v2.sh (which trains on SketchFusionB's
# engineered 348/213-D feature cache, a 10-client Dirichlet shard, and a shared
# global test set, for comparability with this repo's other baselines). This
# script targets the paper's (arXiv:2401.16685v2) own setup instead:
#   • Acc = total_acc_{x,y,z} (raw accelerometer, 128x3) — the literal,
#     unfiltered sensor reading per UCI HAR's own README.txt
#   • Gyro = body_gyro_{x,y,z} (128x3)
#   • encoder = single-layer LSTM, 128 hidden units, no dropout (paper Sec 4.2)
#   • clients = the 30 real UCI HAR subjects (paper Table 1 "Natural Distribution")
#   • evaluation = each client's own local holdout (never a shared test set)
#   • selection = paper Eq. (18) lowest-loss clients, Eq. (21) sample-weighted
#     aggregation, Eq. (13)/(14) SHAP+size+recency modality priority
#   • accuracy reported = the paper's own metric: per-client RandomForest
#     fusion accuracy (Stage #2), never the cross-baseline "test_acc" global-
#     ensemble number the other scripts in this repo use
#
# Self-contained in MFedMC/UCI_HAR_paper/; does not import from or modify
# MFedMC/UCI_HAR, MFedMC/UCI_HAR_v2, or any other existing script/package.
# Logging/output deliberately matches upstream's own minimal style (one line
# per round, a final "Results saved to ..." line, no console-log file, no
# hyperparameter dump, no seed-sweep tooling) rather than this repo's other
# baselines' conventions.
#
# Requires: human+activity+recognition+using+smartphones/UCI HAR Dataset/
#   UCI HAR Dataset/{train,test}/{subject_*.txt,Inertial Signals/} (confirmed
#   present on disk; --uci_root overrides auto-discovery if needed).
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
PKG="${ROOT}/MFedMC/UCI_HAR_paper"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

cd "${PKG}"

exec "${PYTHON}" main.py \
  --num_classes 6 \
  --hidden_size 128 \
  --lr 0.1 \
  --batch_size 32 \
  --local_epochs 5 \
  --train_ratio 0.8 \
  --iterations 30 \
  --top_shap 1 \
  --modality_weights 0.3333333333 0.3333333333 0.3333333333 \
  --client_select_ratio 0.2 \
  --seed 46 \
  "$@"
