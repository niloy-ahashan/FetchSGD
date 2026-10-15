#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR with **PMFH + federated head** (option B): official PMFH
# (github.com/vindahi/PMFH) plus a classifier head trained during
# federated training, like IC / MFedMC v2 / Hybrid v2.
#
#   • PMFH itself untouched: upstream Layers (gate, sizes), upstream
#     LocalUpdate losses, clipping, prototypes, FedAvg
#   • head = Linear(common_dim -> 6) on nec_vec.detach(): trained with
#     CrossEntropy on each client, uploaded and FedAvg'd with the
#     global model; its gradient never reaches PMFH's parameters
#   • test_acc = trained head argmax, 8-sample shard mean (as the
#     baselines)
#   • flags mirror run_uci_har_pmfh.sh so the two are directly comparable
#   • log -> PMFH/result_updated/PMFHfh_..._<cum_up>MB_....txt
#
# Extra args are forwarded, e.g.:
#   ./PMFH/run_uci_har_pmfh_fedhead.sh --rounds 2 --train_ep 1 --train_ep_private 1
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
exec "${PYTHON}" "${HERE}/main_uci_har_v2.py" \
  --variant fedhead \
  --clf_weight 1.0 \
  --head_lr 1e-3 \
  --dataset_dir "${DATA_PATH}" \
  --num_clients 10 \
  --dirichlet_alpha 0.1 \
  --classes 6 \
  --image_dim 348 \
  --text_dim 213 \
  --bit 64 \
  --hidden_dim 128 \
  --common_dim 32 \
  --rounds 5 \
  --batch_size 64 \
  --train_ep 10 \
  --train_ep_private 10 \
  --seed 46 \
  --device cuda \
  "$@"
