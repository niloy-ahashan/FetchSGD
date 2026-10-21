#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR hybrid v2: FetchSGD engine (example-weighted gradient sketch)
# + MFedMC client/modality selection. Independent of HybridSketchMFedMC/.
#
# Extra args are forwarded to main.py, e.g.:
#   ./HybridSketchMFedMC_v2/run_uci_har.sh --upload_object delta --local_epochs 5
#   ./HybridSketchMFedMC_v2/run_uci_har.sh --client_select random
#   ./HybridSketchMFedMC_v2/run_uci_har.sh --fusion_mode sketch
#
# Optional multi-seed sweep (opt-in; default behavior above is unchanged):
# set NUM_SEEDS and/or SEEDS in the environment to run several seeds back to
# back and print/save their mean, e.g.:
#   NUM_SEEDS=3 SEEDS="42 43 44" ./HybridSketchMFedMC_v2/run_uci_har.sh
#   SEEDS="7 8 9" ./HybridSketchMFedMC_v2/run_uci_har.sh
# Do NOT pass --seed yourself as an extra arg in sweep mode: "$@" is appended
# after --seed "${s}" on every iteration, and argparse keeps the last
# occurrence of a flag, so a user-supplied --seed would silently override
# every seed of the sweep.
# A failed seed is skipped (the remaining seeds still run) and listed at the
# end, with a final nonzero exit if anything failed. Each seed's npz is also
# copied to results/HybridV2_..._seed<s>.npz (the runner's own npz name has no
# seed, so the next seed's run would otherwise silently overwrite it).
# After the seeds, summarize_seeds.py prints/saves their mean in the
# Results-sheet columns: result_updated/HybridV2_..._seeds<...>_summary.txt
#
# Requires: datasets/uci_har_mm/data.npz
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/uci_har_mm}"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

# Flags shared by every run; --seed is appended separately below (per branch).
COMMON_ARGS=(
  --dataset_dir "${DATA_PATH}"
  --num_classes 6
  --num_clients 10
  --dirichlet_alpha 0.1
  --acc_dim 348
  --gyro_dim 213
  --feat_dim 512
  --fusion_mode sketch
  --sketch_r 2
  --sketch_c 128
  --mm_dropout 0.3
  --local_batch_size -1
  --upload_object gradient
  --weight_decay 5e-4
  --virtual_momentum 0.9
  --error_type virtual
  --lr_scale 0.05
  --mode sketch
  --k 20000
  --num_rows 3
  --num_cols 20000
  --num_epochs 53
  --pivot_epoch 15
  --local_epochs 1
  --num_blocks 1
  --num_select_modalities 1
  --client_select_ratio 0.4
  --client_select loss
  --device cuda
)

cd "${ROOT}"

if [[ -z "${NUM_SEEDS:-}" && -z "${SEEDS:-}" ]]; then
  # Default path: unchanged today's behavior. Single exec, --seed 42.
  exec "${PYTHON}" "${HERE}/main.py" "${COMMON_ARGS[@]}" --seed 42 "$@"
fi

# --- Opt-in multi-seed sweep -------------------------------------------------
NUM_SEEDS="${NUM_SEEDS:-5}"         # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-42 43 44 45 46}"    # seed values, space-separated

read -r -a SEED_LIST <<< "${SEEDS}"
RESULTS_DIR="${HERE}/results"
FAILED=()
DONE=()
START="$(date +%s)"
for s in "${SEED_LIST[@]:0:${NUM_SEEDS}}"; do
  echo "=== seed ${s} ==="
  SEED_START="$(date +%s)"
  if ! "${PYTHON}" "${HERE}/main.py" "${COMMON_ARGS[@]}" --seed "${s}" "$@"; then
    echo "!!! seed ${s} failed; continuing with the next seed" >&2
    FAILED+=("${s}")
  else
    DONE+=("${s}")
    # the runner's npz name has no seed (the next seed's run would overwrite
    # it): rescue a per-seed copy before the next iteration starts.
    find "${RESULTS_DIR}" -maxdepth 1 -name 'HybridV2_*.npz' ! -name '*_seed*' \
      -newermt "@${SEED_START}" | while read -r f; do cp "${f}" "${f%.npz}_seed${s}.npz"; done
  fi
done

# mean over the finished seeds, in the Results-sheet columns
if (( ${#DONE[@]} )); then
  "${PYTHON}" "${HERE}/summarize_seeds.py" --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
