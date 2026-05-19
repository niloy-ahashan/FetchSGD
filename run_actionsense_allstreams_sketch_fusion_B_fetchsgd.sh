#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# ActionSense (all-streams HDF5) → FetchSGD **SketchFusionB**
#
# Same recipe as run_uci_har_sketch_fusion_B.sh: mm_train.py + SketchFusionB +
# FetchSGD sketch compression (--mode sketch).
#
# Default HDF5 (first path that exists wins):
#   datasets/data_processed_allStreams_10s_10hz_5subj_ex20-20_allActs.hdf5
#   MFedMC/ActionSense/data/data_processed_allStreams_10s_10hz_5subj_ex20-20_allActs.hdf5
#
# One-time prep (writes datasets/actionsense_allstreams_mm/data.npz +
# prepare_stats.json with img_dim / txt_dim / num_classes):
#   python3 CommEfficient/CommEfficient/prepare_actionsense_allstreams_mm.py \
#     --hdf5_path "${HDF5_PATH}" \
#     --out_dir datasets/actionsense_allstreams_mm
#
# Override HDF5 location:
#   HDF5_PATH=/path/to/file.hdf5 ./run_actionsense_allstreams_sketch_fusion_B_fetchsgd.sh
#
# MFedMC ActionSense trainer (different codebase) stays in:
#   ./run_actionsense_allstreams_sketch_B.sh
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

# Prefer explicit HDF5_PATH only when that file exists; ignore bad/placeholder env (e.g. /path/to/...).
DATASETS_HDF5="${ROOT}/datasets/data_processed_allStreams_10s_10hz_5subj_ex20-20_allActs.hdf5"
MFEDMC_HDF5="${ROOT}/MFedMC/ActionSense/data/data_processed_allStreams_10s_10hz_5subj_ex20-20_allActs.hdf5"
if [[ -f "${DATASETS_HDF5}" ]]; then
  DEFAULT_HDF5="${DATASETS_HDF5}"
elif [[ -f "${MFEDMC_HDF5}" ]]; then
  DEFAULT_HDF5="${MFEDMC_HDF5}"
else
  DEFAULT_HDF5="${DATASETS_HDF5}"
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
    echo "Unset HDF5_PATH or point it to your .hdf5, e.g.:" >&2
    echo "  unset HDF5_PATH" >&2
    echo "  export HDF5_PATH=${DATASETS_HDF5}" >&2
    exit 1
  fi
else
  HDF5_PATH="${DEFAULT_HDF5}"
fi

OUT_DIR="${ACTIONSENSE_MM_DIR:-${ROOT}/datasets/actionsense_allstreams_mm}"
STATS_JSON="${OUT_DIR}/prepare_stats.json"
DATA_NPZ="${OUT_DIR}/data.npz"

if [[ ! -f "${DATA_NPZ}" ]]; then
  if [[ ! -f "${HDF5_PATH}" ]]; then
    echo "Missing ${DATA_NPZ} and no HDF5 file found at:" >&2
    echo "  ${HDF5_PATH}" >&2
    echo "Place the ActionSense .hdf5 under datasets/ (or MFedMC/ActionSense/data/) or run:" >&2
    echo "  python3 CommEfficient/CommEfficient/prepare_actionsense_allstreams_mm.py \\" >&2
    echo "    --hdf5_path /absolute/path/to/your.hdf5 --out_dir ${OUT_DIR}" >&2
    exit 1
  fi
  echo "Preparing FedMultiModal npz from ${HDF5_PATH} ..."
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
  --device cuda
