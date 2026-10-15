#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR with **PMFH-v2**: PMFH (github.com/vindahi/PMFH) with the
# IC / MFedMC v2 / Hybrid v2 per-modality gate and a trained
# classifier head, next to IC / MFedMC v2 / Hybrid v2.
#
#   • model: layers_uci.LayersUCI
#       gate  = Linear(c,c) -> ReLU -> Linear(c,c) -> Sigmoid (FeaRefiner)
#       head  = Linear(common_dim -> 6) on nec_vec (neck output)
#   • training: upstream LocalUpdate + clf_weight * CrossEntropy(head)
#     on the global model only; private model, KD, ConLoss and
#     prototypes as upstream
#   • test_acc = trained head argmax, 8-sample shard mean (as the
#     baselines)
#   • log -> PMFH/result_updated/PMFHv2_..._<cum_up>MB_....txt
#   • official PMFH stays in run_uci_har_pmfh.sh / main_uci_har.py
#
# Extra args are forwarded, e.g.:
#   ./PMFH/run_uci_har_pmfh_v2.sh --rounds 2 --train_ep 1 --train_ep_private 1
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
  --dataset_dir "${DATA_PATH}" \
  --num_clients 10 \
  --dirichlet_alpha 0.1 \
  --classes 6 \
  --image_dim 348 \
  --text_dim 213 \
  --bit 64 \
  --hidden_dim 512 \
  --common_dim 128 \
  --rounds 2 \
  --batch_size 64 \
  --train_ep 10 \
  --train_ep_private 10 \
  --clf_weight 1.0 \
  --seed 42 \
  --device cuda \
  "$@"
