#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR with **Independent Compression**
#
# Multimodal fusion is ordinary summation of refined features:
#     fused = f'_acc + f'_gyro
# There is no sketch-based fusion (unlike SketchFusion A/B/C)
# and no cross-modal / missing-modality heads.
#
# The fused model is then trained with original FetchSGD
# (Rothchild et al.): one Count Sketch of the full gradient.
#
# Extra args are forwarded, e.g.:
#   ./run_uci_har_independent_compression.sh --num_epochs 2 --device cpu
#
# Optional multi-seed sweep (opt-in; default behavior above is unchanged):
# set NUM_SEEDS and/or SEEDS in the environment to run several seeds back to
# back and print/save their mean, e.g.:
#   NUM_SEEDS=3 SEEDS="42 43 44" ./run_uci_har_independent_compression.sh
#   SEEDS="7 8 9" ./run_uci_har_independent_compression.sh
# Do NOT pass --seed yourself as an extra arg in sweep mode: "$@" is appended
# after --seed "${s}" on every iteration, and argparse keeps the last
# occurrence of a flag, so a user-supplied --seed would silently override
# every seed of the sweep.
# A failed seed is skipped (the remaining seeds still run) and listed at the
# end, with a final nonzero exit if anything failed. Each seed's console log
# already has its own seed-tagged name (result_updated/IC_..._seed<s>.txt),
# so unlike the MFedMC v2 / Hybrid v2 run scripts, no rescue-copy step is
# needed here.
# After the seeds, summarize_seeds.py prints/saves their mean in the
# Results-sheet columns: CommEfficient/result_updated/IC_..._seeds<...>_summary.txt
#
# Requires: datasets/uci_har_mm/data.npz (see prepare_uci_har_mm.py).
# Cache still uses img_*/txt_* keys; they are acc/gyro features.
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-python3}"

# Flags shared by every run; --seed is appended separately below (per branch).
COMMON_ARGS=(
  CommEfficient/CommEfficient/mm_train_independent.py
  --dataset_dir datasets/uci_har_mm/
  --dataset_name MultiModal
  --model IndependentCompression
  --acc_dim 348
  --gyro_dim 213
  --feat_dim 512
  --mm_dropout 0.3
  --num_classes 6
  --dirichlet_alpha 0.1
  --local_batch_size -1
  --local_momentum 0.0
  --virtual_momentum 0.9
  --error_type virtual
  --share_ps_gpu
  --mode sketch
  --k 20000
  --num_rows 3
  --num_cols 5000
  --num_epochs 85
  --pivot_epoch 20
  --mm_local_epochs 1
  --num_clients 10
  --num_workers 10
  --lr_scale 0.05
  --num_blocks 1
  --num_devices 1
  --device cuda
)

if [[ -z "${NUM_SEEDS:-}" && -z "${SEEDS:-}" ]]; then
  # Default path: unchanged today's behavior. Single exec, --seed 42.
  exec "${PYTHON}" "${COMMON_ARGS[@]}" --seed 42 "$@"
fi

# --- Opt-in multi-seed sweep -------------------------------------------------
NUM_SEEDS="${NUM_SEEDS:-5}"         # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-42 43 44 45 46}"    # seed values, space-separated

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
  "${PYTHON}" "${ROOT}/CommEfficient/CommEfficient/summarize_seeds.py" --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
