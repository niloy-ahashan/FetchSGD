#!/usr/bin/env bash
# Merge all ActionNet streamLogs under MFedMC/dataset into one ActionSense HDF5.
# Subject ids (S00–S09) are parsed from each filename by preprocess_actionnet_streamlog.py.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/../dataset}"

# Quote the glob so the shell does not expand it; Python expands it once.
python "${SCRIPT_DIR}/preprocess_actionnet_streamlog.py" \
  --input-glob "${DATA_DIR}/*streamLog_actionNet-wearables_*.hdf5" \
  --output "${SCRIPT_DIR}/data/actionnet_merged_multisession.hdf5"

echo
echo "Train with:"
echo "  cd ${SCRIPT_DIR} && python main.py --data_path ./data/actionnet_merged_multisession.hdf5"
