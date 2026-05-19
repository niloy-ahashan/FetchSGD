#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# MFedMC ActionSense main.py on UCI-HAR-derived HDF5 (6 classes).
#
# Build the HDF5 first (once):
#   ./run_prepare_uci_har_mfedmc_hdf5.sh
#
# Default data file:
#   MFedMC/ActionSense/data/uci_har_mfedmc.hdf5
#
# Extra args are forwarded to main.py, e.g.:
#   ./run_actionsense_uci_har_mfedmc.sh --iterations 100 --top_shap 2
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
ACTIONSENSE="${ROOT}/MFedMC/ActionSense"
DATA_PATH="${DATA_PATH:-${ACTIONSENSE}/data/uci_har_mfedmc.hdf5}"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

cd "${ACTIONSENSE}"
exec "${PYTHON}" main.py \
  --data_path "${DATA_PATH}" \
  --num_classes 6 \
  "$@"
