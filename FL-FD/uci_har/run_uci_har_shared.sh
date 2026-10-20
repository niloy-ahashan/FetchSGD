#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR with **FL-FD** (Qi et al., Information Fusion 2023), shared-encoder
# variant -- FL-FD's counterpart of FLASH's --branch shared.
#
#   • data: datasets/uci_har_mm -- the 348-D acc / 213-D gyro feature
#     vectors FLASH / Hybrid v2 / MFedMC v2 / PMFH v2 read; same
#     10-client Dirichlet(0.1) split and official UCI HAR test set
#   • model: per modality the Hybrid v2 encoder (FeaExtractor, gated by
#     FeaRefiner) -> concatenate -> FL-FD's one-Linear head (fc1).
#     No GAF / CNN: the shared encoder replaces FL-FD's image encoder
#   • algorithm: unchanged from FL-FD/fixed/ (local_train, fedavg,
#     evaluate): 8 of 10 clients per round (paper: 12 of 15), 3 local
#     epochs, SGD lr 1e-3, momentum 1e-4, bs 32, equal-weight average
#   • width: --feat_dim 64 (as the FLASH shared runs); 512 = Hybrid v2
#   • comm: K x full float state_dict down and up per round; round-1
#     download (shared init) not billed; MB = 1e6 bytes.
#     fd64: 61,766 params, 1.977 MB up per round; fd512: 1,870,342
#     params, 59.85 MB up per round
#   • test_acc = 8-sample shard mean (as the baselines)
#   • log -> FL-FD/uci_har/result_updated/FLFD_uci_har_mm_mm_<cum_up>MB_..._shared_fd64_....txt
#   • npz -> FL-FD/uci_har/results/FLFD_uci_har_mm_K8_E3_R..._shared_fd64_seed42.npz
#
# Extra args are forwarded, e.g.:
#   ./FL-FD/uci_har/run_uci_har_shared.sh --rounds 2 --device cpu
#   ./FL-FD/uci_har/run_uci_har_shared.sh --feat_dim 512   # Hybrid v2 width
#
# Requires: datasets/uci_har_mm/data.npz
# (see CommEfficient/CommEfficient/prepare_uci_har_mm.py)
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/uci_har_mm}"
PYTHON="${PYTHON:-${ROOT}/venv/bin/python}"   # needs pyts + Pillow (MFedMC/.venv lacks them)

cd "${ROOT}"
exec "${PYTHON}" "${HERE}/main.py" \
  --dataset_dir "${DATA_PATH}" \
  --encoder shared \
  --feat_dim 64 \
  --mm_dropout 0.3 \
  --num_clients 10 \
  --dirichlet_alpha 0.1 \
  --rounds 25 \
  --clients_per_round 8 \
  --local_epochs 3 \
  --lr 1e-3 \
  --momentum 1e-4 \
  --bs 32 \
  --valid_batch_size 8 \
  --seed 42 \
  --device cuda \
  "$@"
