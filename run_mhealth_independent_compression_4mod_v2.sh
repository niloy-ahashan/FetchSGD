#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# mHealth (4 modalities: Acc/Gyro/Mag/ECG) with **Independent Compression**,
# run and logged the same way as run_uci_har_independent_compression.sh.
#
# Same method and flags as run_mhealth_independent_compression_4mod.sh
# (summation fusion of the 4 refined features, original FetchSGD: one
# Count Sketch of the full gradient), but through
# mm_train_independent_4mod_v2.py, which adds:
#   - the compact per-round table (iter/test_acc/up_MB/cum_up_MB/...),
#   - the console log tee'd to
#     CommEfficient/result_updated/IC_<dataset>_mm_<cum_up>MB_..._seed<N>.txt,
#   - --seed 42 by default, extra args forwarded, PYTHON override,
#   - the opt-in multi-seed sweep + summarize_seeds.py mean.
# The original 4mod script is kept unchanged for prior runs.
#
# Extra args are forwarded, e.g.:
#   ./run_mhealth_independent_compression_4mod_v2.sh --num_epochs 2 --device cpu
#
# Optional multi-seed sweep (opt-in; default behavior above is unchanged):
# set NUM_SEEDS and/or SEEDS in the environment to run several seeds back to
# back and print/save their mean, e.g.:
#   NUM_SEEDS=3 SEEDS="42 43 44" ./run_mhealth_independent_compression_4mod_v2.sh
#   SEEDS="7 8 9" ./run_mhealth_independent_compression_4mod_v2.sh
# Do NOT pass --seed yourself as an extra arg in sweep mode: "$@" is appended
# after --seed "${s}" on every iteration, and argparse keeps the last
# occurrence of a flag, so a user-supplied --seed would silently override
# every seed of the sweep.
# A failed seed is skipped (the remaining seeds still run) and listed at the
# end, with a final nonzero exit if anything failed.
# After the seeds, summarize_seeds.py prints/saves their mean in the
# Results-sheet columns: CommEfficient/result_updated/IC_<dataset>_..._seeds<...>_summary.txt
#
# Requires: <dir>/data.npz + prepare_stats.json (see prepare_mhealth_mm.py).
# Dims and num_classes come from prepare_stats.json, not CLI flags.
# Pick a different prepared cache with DATA_PATH, e.g.
#   DATA_PATH=datasets/mhealth_mm_random bash run_mhealth_independent_compression_4mod_v2.sh
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-python3}"

# OUT_DIR="${DATA_PATH:-${ROOT}/datasets/mhealth_mm_random_group}"
OUT_DIR="${DATA_PATH:-${ROOT}/datasets/mhealth_mm}"

OUT_DIR="${OUT_DIR%/}"
if [[ ! -f "${OUT_DIR}/data.npz" ]]; then
  echo "Missing ${OUT_DIR}/data.npz — run prepare_mhealth_mm.py first:" >&2
  echo "  python3 CommEfficient/CommEfficient/prepare_mhealth_mm.py" >&2
  exit 1
fi
# dataset dir name as it appears in the log file names (for summarize_seeds.py)
DATASET_NAME="$(basename "${OUT_DIR}")"

# Flags shared by every run; --seed is appended separately below (per branch).
COMMON_ARGS=(
  CommEfficient/CommEfficient/mm_train_independent_4mod_v2.py
  --dataset_dir "${OUT_DIR}/"
  --dataset_name MultiModal
  --model IndependentCompression4
  --feat_dim 512
  --mm_dropout 0.3
  --dirichlet_alpha 0.1
  --skip_map
  --mm_local_epochs 1
  --local_batch_size -1
  --local_momentum 0.0
  --virtual_momentum 0.9
  --error_type virtual
  --mode sketch
  --num_clients 10
  --num_devices 1
  --num_workers 10
  --share_ps_gpu
  --k 20000
  --num_rows 3
  --num_cols 5000
  --lr_scale 0.05
  --pivot_epoch 10
  --num_blocks 1
  --num_epochs 42
  --device cuda
)

# if [[ -z "${NUM_SEEDS:-}" && -z "${SEEDS:-}" ]]; then
#   # Default path: single exec, --seed 42.
#   exec "${PYTHON}" "${COMMON_ARGS[@]}" --seed 42 "$@"
# fi

# --- Opt-in multi-seed sweep -------------------------------------------------
NUM_SEEDS="${NUM_SEEDS:-5}"         # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-47 48 44 45 46}"    # seed values, space-separated

read -r -a SEED_LIST <<< "${SEEDS}"
FAILED=()
DONE=()
START="$(date +%s)"
for s in "${SEED_LIST[@]:0:${NUM_SEEDS}}"; do
  echo "=== seed ${s} ==="
  if ! "${PYTHON}" "${COMMON_ARGS[@]}" --seed "${s}" "$@"; then
    echo "!!! seed ${s} failed; continuing with the next seed" >&2
    FAILED+=("${s}")
  else
    DONE+=("${s}")
  fi
done

# mean over the finished seeds, in the Results-sheet columns
if (( ${#DONE[@]} )); then
  "${PYTHON}" "${ROOT}/CommEfficient/CommEfficient/summarize_seeds.py" \
    --dataset "${DATASET_NAME}" --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
