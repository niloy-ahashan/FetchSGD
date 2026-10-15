#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# mHealth with **FLASH** (Salehi et al., INFOCOM'22), PyTorch port of
# FLASH/code/federated/ next to IC / MFedMC v2 / Hybrid v2 (4 modalities).
#
#   • data: same SketchFusionB4 split as the other mHealth baselines
#     (Acc 177-D + Gyro 118-D + Mag 118-D + ECG 43-D, 12 classes,
#     10-client Dirichlet(0.1), datasets/mhealth_mm_random_group)
#   • model: one FLASH GPS (coord_mlp) branch per modality -> concat
#     (64*4 = 256) -> unchanged FLASH fusion head
#   • recipe: FLASH's own -- all clients every round, Adam 1e-4, bs 32,
#     <= local_epochs with early stopping (patience 5) on a 10% local
#     val split, unweighted mean of uploaded models
#   • policy: all (released run_flash.sh); paper's random branch via
#     --policy p_acc,p_gyro,p_mag,p_ecg,p_fusion
#   • test_acc = 8-sample shard mean (as the baselines)
#   • log -> FLASH/torch_port/result_updated/FLASH_..._<cum_up>MB_....txt
#
# Extra args are forwarded, e.g.:
#   ./FLASH/torch_port/run_mhealth.sh --rounds 2 --local_epochs 2
#   ./FLASH/torch_port/run_mhealth.sh --branch shared   # Hybrid v2 / IC encoder per modality
#   ./FLASH/torch_port/run_mhealth.sh --branch gps --branch_dims 128,64,32 --fusion_dims 128,64,32,16   # widths /8
#   ./FLASH/torch_port/run_mhealth.sh --policy 0.15,0.15,0.15,0.15,0.4
#
# Requires: datasets/mhealth_mm_random_group/data.npz
# (see CommEfficient/CommEfficient/prepare_mhealth_mm.py)
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/mhealth_mm_random_group}"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

cd "${ROOT}"
exec "${PYTHON}" "${HERE}/main.py" \
  --dataset mhealth \
  --dataset_dir "${DATA_PATH}" \
  --num_clients 10 \
  --dirichlet_alpha 0.1 \
  --rounds 200 \
  --local_epochs 100 \
  --lr 1e-4 \
  --bs 32 \
  --patience 5 \
  --val_frac 0.1 \
  --policy all \
  --seed 42 \
  --device cuda \
  "$@"
