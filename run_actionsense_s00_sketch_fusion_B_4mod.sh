#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# ActionSense S00 — **four** fed-multimodal inputs + SketchFusionB4 + FetchSGD sketch
#
# Modalities (see prepare_actionsense_4mod_mm.py):
#   m0 Eye | m1 EMG L+R | m2 Tactile L+R | m3 IMU
#
# Prep (writes datasets/actionsense_s00_4mod_mm/data.npz + prepare_stats.json):
#   python3 CommEfficient/CommEfficient/prepare_actionsense_s00_4mod_mm.py
#
# Override paths:
#   HDF5_PATH=... ACTIONSENSE_MM_DIR=... ./run_actionsense_s00_sketch_fusion_B_4mod.sh
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

DATASETS_HDF5="${ROOT}/datasets/S00_preprocessed.hdf5"
MFEDMC_HDF5="${ROOT}/MFedMC/ActionSense/data/S00_preprocessed.hdf5"
if [[ -f "${DATASETS_HDF5}" ]]; then
  DEFAULT_HDF5="${DATASETS_HDF5}"
elif [[ -f "${MFEDMC_HDF5}" ]]; then
  DEFAULT_HDF5="${MFEDMC_HDF5}"
else
  DEFAULT_HDF5="${MFEDMC_HDF5}"
fi

REQUESTED_HDF5="${HDF5_PATH:-}"
if [[ -n "${REQUESTED_HDF5}" ]] && [[ -f "${REQUESTED_HDF5}" ]]; then
  HDF5_PATH="${REQUESTED_HDF5}"
elif [[ -n "${REQUESTED_HDF5}" ]] && [[ ! -f "${REQUESTED_HDF5}" ]]; then
  echo "HDF5_PATH is set but file not found: ${REQUESTED_HDF5}" >&2
  if [[ -f "${DEFAULT_HDF5}" ]]; then
    echo "Falling back to: ${DEFAULT_HDF5}" >&2
    HDF5_PATH="${DEFAULT_HDF5}"
  else
    exit 1
  fi
else
  HDF5_PATH="${DEFAULT_HDF5}"
fi

OUT_DIR="${ACTIONSENSE_MM_DIR:-${ROOT}/datasets/actionsense_s00_4mod_mm}"
STATS_JSON="${OUT_DIR}/prepare_stats.json"
DATA_NPZ="${OUT_DIR}/data.npz"

if [[ ! -f "${DATA_NPZ}" ]]; then
  if [[ ! -f "${HDF5_PATH}" ]]; then
    echo "Missing ${DATA_NPZ} and no HDF5 at:" >&2
    echo "  ${HDF5_PATH}" >&2
    exit 1
  fi
  echo "Preparing 4-modality npz from ${HDF5_PATH} → ${OUT_DIR} ..."
  python3 CommEfficient/CommEfficient/prepare_actionsense_4mod_mm.py \
    --hdf5_path "${HDF5_PATH}" \
    --out_dir "${OUT_DIR}"
fi

if [[ ! -f "${STATS_JSON}" ]]; then
  echo "Expected ${STATS_JSON} after prepare." >&2
  exit 1
fi

eval "$(
  python3 - <<PY
import json
with open("${STATS_JSON}") as f:
    d = json.load(f)
if "num_classes" not in d:
    raise SystemExit("prepare_stats.json missing num_classes")
print(f"export NUM_CLASSES={d['num_classes']}")
PY
)"

# shellcheck disable=SC2154
: "${NUM_CLASSES:?}"

if [[ -z "${DEVICE:-}" ]]; then
  if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE=cuda
  else
    DEVICE=cpu
    echo "CUDA not available — using --device cpu (set DEVICE=cuda to force)." >&2
  fi
fi

python3 CommEfficient/CommEfficient/mm_train_actionsense_4mod.py \
  --dataset_dir "${OUT_DIR}" \
  --dataset_name MultiModal \
  --model SketchFusionB4 \
  --feat_dim 512 \
  --sketch_r 4 \
  --sketch_c 128 \
  --mm_dropout 0.3 \
  --num_classes "${NUM_CLASSES}" \
  --dirichlet_alpha 0.1 \
  --sim_loss_weight 0.0 \
  --missing_loss_weight 0.0 \
  --missing_prob 0.0 \
  --skip_map \
  --mm_local_epochs 10 \
  --local_batch_size -1 \
  --local_momentum 0.0 \
  --virtual_momentum 0.9 \
  --error_type virtual \
  --mode sketch \
  --num_clients 10 \
  --num_devices 1 \
  --num_workers 5 \
  --share_ps_gpu \
  --k 20000 \
  --num_rows 3 \
  --num_cols 10000 \
  --lr_scale 0.001 \
  --pivot_epoch 10 \
  --num_blocks 1 \
  --num_epochs 25 \
  --device "${DEVICE}"
