#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR (raw Inertial Signals) → ActionSense-compatible HDF5 for MFedMC main.py
#
# Writes: MFedMC/ActionSense/data/uci_har_mfedmc.hdf5 (override with --output)
#
# Point --uci_root at the *inner* folder that contains train/ and test/:
#   .../UCI HAR Dataset/UCI HAR Dataset
#
# Then train:
#   ./run_actionsense_uci_har_mfedmc.sh
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_UCI="${ROOT}/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
OUT_DEFAULT="${ROOT}/MFedMC/ActionSense/data/uci_har_mfedmc.hdf5"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

UCI_ROOT="${UCI_ROOT:-$DEFAULT_UCI}"
OUTPUT="${OUTPUT:-$OUT_DEFAULT}"

"${PYTHON}" "${ROOT}/MFedMC/ActionSense/prepare_uci_har_mfedmc_hdf5.py" \
  --uci_root "${UCI_ROOT}" \
  --output "${OUTPUT}" \
  "$@"
