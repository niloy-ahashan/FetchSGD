#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR with **PMFH** (Prototype-based Federated Multi-modal
# Hashing, upstream github.com/vindahi/PMFH) as a baseline next to
# IC / MFedMC v2 / Hybrid v2.
#
#   • same 10-client Dirichlet(alpha=0.1) cache (datasets/uci_har_mm)
#   • Acc -> PMFH "image" input, Gyro -> "text" input
#   • upstream PMFH training (LocalUpdate from main.py), flickr.sh
#     rounds/batch, upstream lr / local epochs / loss weights
#   • test_acc = kNN (Hamming) on hash codes, official test set;
#     mAP logged too
#   • log -> PMFH/result_updated/PMFH_..._<cum_up>MB_....txt
#
# Extra args are forwarded, e.g.:
#   ./PMFH/run_uci_har_pmfh.sh --rounds 2 --train_ep 1 --train_ep_private 1
#
# Requires: datasets/uci_har_mm/data.npz
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/uci_har_mm}"

if [[ -x "${ROOT}/venv/bin/python" ]]; then
  PYTHON="${ROOT}/venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

cd "${HERE}"
exec "${PYTHON}" "${HERE}/main_uci_har.py" \
  --dataset_dir "${DATA_PATH}" \
  --num_clients 10 \
  --dirichlet_alpha 0.1 \
  --classes 6 \
  --image_dim 348 \
  --text_dim 213 \
  --bit 32 \
  --hidden_dim 256 \
  --common_dim 128 \
  --rounds 5 \
  --batch_size 64 \
  --train_ep 5 \
  --train_ep_private 5 \
  --knn_k 10 \
  --seed 42 \
  --device cuda \
  "$@"
