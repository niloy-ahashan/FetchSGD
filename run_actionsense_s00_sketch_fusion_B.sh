#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# ActionSense S00 (S00_preprocessed.hdf5) → **SketchFusionB** + FetchSGD sketch
#
# Same training recipe as run_uci_har_sketch_fusion_B.sh: mm_train.py,
# SketchFusionB, gradient compression --mode sketch.
#
# One-time prep (writes datasets/actionsense_s00_mm/data.npz +
# prepare_stats.json with img_dim / txt_dim / num_classes):
#   python3 CommEfficient/CommEfficient/prepare_actionsense_s00_mm.py
#
# Or with explicit paths:
#   python3 CommEfficient/CommEfficient/prepare_actionsense_allstreams_mm.py \
#     --hdf5_path /path/to/S00_preprocessed.hdf5 \
#     --out_dir datasets/actionsense_s00_mm
#
# Override HDF5 / output dir for prepare:
#   HDF5_PATH=/path/to/file.hdf5 ACTIONSENSE_MM_DIR=/path/to/out ./run_actionsense_s00_sketch_fusion_B.sh
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

DEFAULT_HDF5="${ROOT}/MFedMC/ActionSense/data/S00_preprocessed.hdf5"
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

OUT_DIR="${ACTIONSENSE_MM_DIR:-${ROOT}/datasets/actionsense_s00_mm}"
STATS_JSON="${OUT_DIR}/prepare_stats.json"
DATA_NPZ="${OUT_DIR}/data.npz"

if [[ ! -f "${DATA_NPZ}" ]]; then
  if [[ ! -f "${HDF5_PATH}" ]]; then
    echo "Missing ${DATA_NPZ} and no HDF5 at:" >&2
    echo "  ${HDF5_PATH}" >&2
    echo "Place S00_preprocessed.hdf5 there or set HDF5_PATH." >&2
    exit 1
  fi
  echo "Preparing FedMultiModal npz from ${HDF5_PATH} → ${OUT_DIR} ..."
  python3 CommEfficient/CommEfficient/prepare_actionsense_allstreams_mm.py \
    --hdf5_path "${HDF5_PATH}" \
    --out_dir "${OUT_DIR}"
fi

if [[ ! -f "${STATS_JSON}" ]]; then
  echo "Expected ${STATS_JSON} after prepare. Re-run prepare or check OUT_DIR." >&2
  exit 1
fi

eval "$(
  python3 - <<PY
import json
with open("${STATS_JSON}") as f:
    d = json.load(f)
for k in ("img_dim", "txt_dim", "num_classes"):
    print(f"export {k.upper()}={d[k]}")
PY
)"

# shellcheck disable=SC2154
: "${IMG_DIM:?}" "${TXT_DIM:?}" "${NUM_CLASSES:?}"

# Default: use CUDA when PyTorch sees a GPU; otherwise CPU (override with DEVICE=cuda|cpu).
if [[ -z "${DEVICE:-}" ]]; then
  if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE=cuda
  else
    DEVICE=cpu
    echo "CUDA not available — using --device cpu (set DEVICE=cuda to force)." >&2
  fi
fi

python3 CommEfficient/CommEfficient/mm_train.py \
  --dataset_dir "${OUT_DIR}" \
  --dataset_name MultiModal \
  --model SketchFusionB \
  --img_dim "${IMG_DIM}" \
  --txt_dim "${TXT_DIM}" \
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
  --num_cols 5000 \
  --lr_scale 0.1 \
  --pivot_epoch 15 \
  --num_blocks 1 \
  --num_epochs 40 \
  --device "${DEVICE}"
