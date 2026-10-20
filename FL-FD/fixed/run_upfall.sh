#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UP-Fall with **FL-FD** (Qi et al., Information Fusion 2023), fixed PyTorch
# port of FL-FD/{data_processing,main}.py.
#
#   • data: CompleteDataSet.csv + Camera2 frames (FL-FD/dataset/), subjects
#     5 and 9 dropped -> 15 clients, one per subject; trials 1-2 train,
#     trial 3 test (global test set)
#   • input: [GAF(acc), GAF(gyro), camera frame-difference] -> 3 x 70 x 70,
#     one sample per (trial, wearable sensor)
#   • recipe: FedAvg, 12 of 15 clients per round, 3 local epochs, SGD lr 1e-3,
#     bs 32; 200 rounds (activity, 11-class) / 100 rounds (fall, binary)
#   • step 1 builds/refreshes the cache (no-op if the trial set is unchanged;
#     trials without camera frames are skipped and listed)
#   • log -> FL-FD/fixed/result_updated/FLFD_UPFall_....txt
#
# Extra args go to main.py, e.g.:
#   ./FL-FD/fixed/run_upfall.sh --task fall
#   ./FL-FD/fixed/run_upfall.sh --sensors Belt --rounds 5
#   ./FL-FD/fixed/run_upfall.sh --seed 1
#
# Requires: FL-FD/dataset/CompleteDataSet.csv, FL-FD/dataset/camera/
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT}/FL-FD/dataset}"
CACHE="${CACHE:-${DATA_DIR}/flfd_upfall_camera2_70.npz}"
PYTHON="${PYTHON:-${ROOT}/venv/bin/python}"

cd "${ROOT}"
"${PYTHON}" "${HERE}/prepare_upfall.py" \
  --csv "${DATA_DIR}/CompleteDataSet.csv" \
  --camera_dir "${DATA_DIR}/camera" \
  --camera 2 \
  --out "${CACHE}" \
  --window 140 \
  --size 70 \
  --gaf_method difference

exec "${PYTHON}" "${HERE}/main.py" \
  --cache "${CACHE}" \
  --task activity \
  --sensors all \
  --clients_per_round 12 \
  --local_epochs 3 \
  --lr 1e-3 \
  --momentum 1e-4 \
  --bs 32 \
  --seed 42 \
  --device cuda \
  "$@"
