#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR with **FLASH, widths /8** (Salehi et al., INFOCOM'22),
# PyTorch port of FLASH/code/federated/ -- the smaller-model variant,
# like PMFH's --hidden_dim / --common_dim reduction.
#
#   • data: same SketchFusionB split (Acc 348-D + Gyro 213-D,
#     10-client Dirichlet(0.1), official UCI HAR test set)
#   • model: FLASH GPS (coord_mlp) branch per modality -> concat
#     (64*M) -> FLASH fusion head; every layer kept, hidden widths /8:
#       branch Dense 1024,512,256      -> 128,64,32
#       fusion Dense 1024,512,256,128  -> 128,64,32,16
#     419,110 params, 1.68 MB per upload (full widths: 5.07 M, 20.28 MB)
#   • training flags mirror run_uci_har.sh, so the two differ only in
#     the model
#   • log -> FLASH/torch_port/result_updated/FLASH_..._bd128-64-32_fd128-64-32-16_....txt
#
# Extra args are forwarded, e.g.:
#   ./FLASH/torch_port/run_uci_har_small.sh --rounds 2 --local_epochs 2
#   ./FLASH/torch_port/run_uci_har_small.sh --branch_dims 256,128,64 --fusion_dims 256,128,64,32   # widths /4
#
# Requires: datasets/uci_har_mm/data.npz
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/uci_har_mm}"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

cd "${ROOT}"
exec "${PYTHON}" "${HERE}/main.py" \
  --dataset uci_har \
  --dataset_dir "${DATA_PATH}" \
  --num_clients 10 \
  --dirichlet_alpha 0.1 \
  --rounds 10 \
  --local_epochs 5 \
  --lr 1e-4 \
  --bs 32 \
  --patience 5 \
  --val_frac 0.0 \
  --policy all \
  --seed 42 \
  --branch gps \
  --branch_dims 128,64,32 \
  --fusion_dims 128,64,32,16 \
  --device cuda \
  "$@"
