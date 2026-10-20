#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR with **FL-FD** (Qi et al., Information Fusion 2023), next to
# FLASH / IC / MFedMC v2 / PMFH v2 / Hybrid v2.
#
#   • data: same SketchFusionB split (10-client Dirichlet(0.1), official
#     UCI HAR test set), read from datasets/uci_har_lstm_mm, which holds
#     the same rows as uci_har_mm but as raw body_acc / body_gyro windows
#   • input: per modality |x,y,z| -> GAF (difference, as released) ->
#     70 x 70; one channel per modality (no camera), so 2 x 70 x 70
#   • algorithm: unchanged from FL-FD/fixed/ (FLFDNet, local_train,
#     fedavg): 8 of 10 clients per round (paper: 12 of 15), 3 local
#     epochs, SGD lr 1e-3, momentum 1e-4, bs 32, equal-weight average
#   • comm: K x full float state_dict down and up per round; round-1
#     download (shared init) not billed; MB = 1e6 bytes
#   • test_acc = 8-sample shard mean (as the baselines)
#   • log -> FL-FD/uci_har/result_updated/FLFD_..._<cum_up>MB_....txt
#
# Extra args are forwarded, e.g.:
#   ./FL-FD/uci_har/run_uci_har.sh --rounds 2 --device cpu
#   ./FL-FD/uci_har/run_uci_har.sh --gaf_method summation   # the paper's formula
#   ./FL-FD/uci_har/run_uci_har.sh --clients_per_round 10
#
# Requires: datasets/uci_har_lstm_mm/data.npz
# (see CommEfficient/CommEfficient/prepare_uci_har_lstm_mm.py)
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/uci_har_lstm_mm}"
PYTHON="${PYTHON:-${ROOT}/venv/bin/python}"   # needs pyts + Pillow (MFedMC/.venv lacks them)

cd "${ROOT}"
exec "${PYTHON}" "${HERE}/main.py" \
  --dataset_dir "${DATA_PATH}" \
  --num_clients 10 \
  --dirichlet_alpha 0.1 \
  --rounds 350 \
  --clients_per_round 8 \
  --local_epochs 3 \
  --lr 1e-3 \
  --momentum 1e-4 \
  --bs 32 \
  --gaf_method difference \
  --size 70 \
  --valid_batch_size 8 \
  --seed 42 \
  --device cuda \
  "$@"
