#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR with **FLASH** (Salehi et al., INFOCOM'22), PyTorch port of
# FLASH/code/federated/ next to IC / MFedMC v2 / PMFH v2 / Hybrid v2.
#
#   • data: same SketchFusionB split (Acc 348-D + Gyro 213-D,
#     10-client Dirichlet(0.1), official UCI HAR test set)
#   • model: one FLASH GPS (coord_mlp) branch per modality -> concat
#     (64*M) -> unchanged FLASH fusion head
#   • recipe: FLASH's own -- all clients every round, Adam 1e-4, bs 32,
#     <= local_epochs with early stopping (patience 5) on a 10% local
#     val split, unweighted mean of uploaded models
#   • policy: all (released run_flash.sh); paper's random branch via
#     --policy p_acc,p_gyro,p_fusion
#   • test_acc = 8-sample shard mean (as the baselines)
#   • log -> FLASH/torch_port/result_updated/FLASH_..._<cum_up>MB_....txt
#
# Extra args are forwarded, e.g.:
#   ./FLASH/torch_port/run_uci_har.sh --rounds 2 --local_epochs 2
#   ./FLASH/torch_port/run_uci_har.sh --branch shared   # Hybrid v2 / IC encoder per modality
#   ./FLASH/torch_port/run_uci_har.sh --branch gps --branch_dims 128,64,32 --fusion_dims 128,64,32,16   # widths /8
#   ./FLASH/torch_port/run_uci_har.sh --policy 0.3,0.3,0.4
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
  --rounds 3 \
  --local_epochs 10 \
  --lr 1e-4 \
  --bs 32 \
  --patience 5 \
  --val_frac 0.0 \
  --policy all \
  --seed 42 \
  --branch shared \
  --fusion_dims 128,64,32,16 \
  --feat_dim 64 \
  --device cuda \
  "$@"
