#!/usr/bin/env bash
set -euo pipefail

# MFedMC on the SketchFusionB UCI HAR Acc/Gyro split (datasets/uci_har_mm).
# Algorithm matches GitHub ActionSense MFedMC; only the dataset/encoders differ.

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/uci_har_mm/data.npz}"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

cd "${HERE}"
exec "${PYTHON}" main.py --data_path "${DATA_PATH}" "$@"
