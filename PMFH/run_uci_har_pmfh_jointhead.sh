#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR with **PMFH + jointly trained head** (option C): PMFH
# (github.com/vindahi/PMFH) plus a classifier head trained during
# federated training, like IC / MFedMC v2 / Hybrid v2.
#
#   • same as run_uci_har_pmfh_fedhead.sh (upstream Layers, gate, sizes,
#     upstream losses, prototypes, FedAvg) except the head is joint:
#   • head = Linear(common_dim -> 6) on nec_vec (no detach): trained with
#     CrossEntropy on each client, uploaded and FedAvg'd with the
#     global model; its CE gradient also trains PMFH's network, like the
#     baselines' jointly trained classifier (PMFH's features change)
#   • test_acc = trained head argmax, 8-sample shard mean (as the
#     baselines)
#   • flags mirror run_uci_har_pmfh_fedhead.sh (detached head) for a direct comparison
#   • log -> PMFH/result_updated/PMFHjh_..._<cum_up>MB_....txt
#
# Extra args are forwarded, e.g.:
#   ./PMFH/run_uci_har_pmfh_jointhead.sh --rounds 2 --train_ep 1 --train_ep_private 1
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
  --head_grad joint \
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
  --seed 42 \
  --device cuda \
  "$@"
