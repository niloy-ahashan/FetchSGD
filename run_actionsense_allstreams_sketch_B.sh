#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# ActionSense — MFedMC multi-modal federated encoders (LSTM + fusion)
#
# Same *layout* as run_uci_har_sketch_fusion_B.sh: dataset-specific runner
# from repo root with clear defaults and overridable paths.
#
# This invokes MFedMC/ActionSense/main.py (not CommEfficient SketchFusionB).
#
# Default dataset:
#   data_processed_allStreams_10s_10hz_5subj_ex20-20_allActs.hdf5
#   (all streams, 10 s windows, 10 Hz, 5 subjects, ex20-20, all activities)
#
# Examples:
#   ./run_actionsense_allstreams_sketch_B.sh
#   ./run_actionsense_allstreams_sketch_B.sh --iterations 100 --prefer-higher-loss
#   DATA_PATH=/path/to/other.hdf5 ./run_actionsense_allstreams_sketch_B.sh
#   ACTIONSENSE_DIR=/path/to/MFedMC/ActionSense ./run_actionsense_allstreams_sketch_B.sh
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
ACTIONSENSE_DIR="${ACTIONSENSE_DIR:-${ROOT}/MFedMC/ActionSense}"
DATA_PATH="${DATA_PATH:-${ACTIONSENSE_DIR}/data/data_processed_allStreams_10s_10hz_5subj_ex20-20_allActs.hdf5}"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
elif [[ -x "${ACTIONSENSE_DIR}/../.venv/bin/python" ]]; then
  PYTHON="${ACTIONSENSE_DIR}/../.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

cd "${ACTIONSENSE_DIR}"

exec "${PYTHON}" main.py --data_path "${DATA_PATH}" "$@"
