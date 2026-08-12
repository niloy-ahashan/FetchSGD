#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# WISDM raw sensors → ActionSense-compatible HDF5 for MFedMC main.py
#
# Writes: MFedMC/ActionSense/data/wisdm_mfedmc.hdf5
#
# Then train:
#   ./run_actionsense_wisdm_mfedmc.sh
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WISDM="${ROOT}/datasets/wisdm/wisdm-dataset"
OUT_DEFAULT="${ROOT}/MFedMC/ActionSense/data/wisdm_mfedmc.hdf5"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

WISDM_ROOT="${WISDM_ROOT:-$DEFAULT_WISDM}"
OUTPUT="${OUTPUT:-$OUT_DEFAULT}"

"${PYTHON}" "${ROOT}/MFedMC/ActionSense/prepare_wisdm_mfedmc_hdf5.py" \
  --wisdm_root "${WISDM_ROOT}" \
  --output "${OUTPUT}" \
  "$@"
