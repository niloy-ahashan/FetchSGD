#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# MFedMC v2 on UCI HAR Acc/Gyro features, SketchFusionB protocol,
# IC-style per-modality encoder (Level 2):
#   raw Acc/Gyro -> Extractor (in_dim->feat_dim->feat_dim, IC-shaped)
#                -> Refiner (sigmoid gate, same as IC; toggle with
#                   --no-use_refiner)
#                -> small classifier (log-probs)
# No shared/joint fusion is added — each modality is still trained and
# uploaded independently; MFedMC's SHAP-driven selection + RandomForest
# late fusion in federated.py are unchanged from v1.
#
#   • clients = 10-client Dirichlet(alpha=0.1) partition (not subjects)
#   • all of each client's data is training data (no local hold-out)
#   • fusion scored on the official shared UCI HAR test set
#   • local_epochs=5, client_select_ratio=0.1, top_shap=1
#
# Sibling of run_uci_har_mfedmc_sketchfusion_protocol.sh, pointed at
# MFedMC/UCI_HAR_v2 instead of MFedMC/UCI_HAR (v1). Self-contained; does
# not modify MFedMC/UCI_HAR (v1) or its run script.
#
# Features come from datasets/uci_har_mm/data.npz (same Acc/Gyro vectors
# as SketchFusionB). --uci_root is not needed here since the dirichlet
# partition doesn't use per-subject IDs.
#
# Optional multi-seed sweep (opt-in; default behavior above is unchanged):
# set NUM_SEEDS and/or SEEDS in the environment to run several seeds back to
# back and print/save their mean, e.g.:
#   NUM_SEEDS=3 SEEDS="42 43 44" ./run_uci_har_mfedmc_sketchfusion_protocol_v2.sh
#   SEEDS="7 8 9" ./run_uci_har_mfedmc_sketchfusion_protocol_v2.sh
# Do NOT pass --seed yourself as an extra arg in sweep mode: "$@" is appended
# after --seed "${s}" on every iteration, and argparse keeps the last
# occurrence of a flag, so a user-supplied --seed would silently override
# every seed of the sweep.
# A failed seed is skipped (the remaining seeds still run) and listed at the
# end, with a final nonzero exit if anything failed. Each seed's npz is also
# copied to results/MFedMC_UCI_HAR_v2_..._seed<s>.npz (the runner's own npz
# name has no seed, so the next seed's run would otherwise silently overwrite
# it).
# After the seeds, summarize_seeds.py prints/saves their mean in the
# Results-sheet columns: result_updated/MFedMC_UCI_HAR_v2_..._seeds<...>_summary.txt
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
UCI_HAR="${ROOT}/MFedMC/UCI_HAR_v2"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/uci_har_mm}"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

# Flags shared by every run; --seed is appended separately below (per branch).
COMMON_ARGS=(
  --dataset_dir "${DATA_PATH}"
  --partition dirichlet
  --dirichlet_alpha 0.1
  --num_classes 6
  --acc_dim 348
  --gyro_dim 213
  --feat_dim 512
  --use_refiner
  --batch_size 32
  --iterations 14
  --local_epochs 5
  --num_clients 10
  --client_select_ratio 0.1
  --top_shap 1
  --train_ratio 1.0
  --eval_on_global_test
)

cd "${UCI_HAR}"

if [[ -z "${NUM_SEEDS:-}" && -z "${SEEDS:-}" ]]; then
  # Default path: unchanged today's behavior. Single exec, --seed 42.
  exec "${PYTHON}" main.py "${COMMON_ARGS[@]}" --seed 42 "$@"
fi

# --- Opt-in multi-seed sweep -------------------------------------------------
NUM_SEEDS="${NUM_SEEDS:-5}"         # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-42 43 44 45 46}"    # seed values, space-separated

read -r -a SEED_LIST <<< "${SEEDS}"
RESULTS_DIR="${UCI_HAR}/results"
FAILED=()
DONE=()
START="$(date +%s)"
for s in "${SEED_LIST[@]:0:${NUM_SEEDS}}"; do
  echo "=== seed ${s} ==="
  SEED_START="$(date +%s)"
  if ! "${PYTHON}" main.py "${COMMON_ARGS[@]}" --seed "${s}" "$@"; then
    echo "!!! seed ${s} failed; continuing with the next seed" >&2
    FAILED+=("${s}")
  else
    DONE+=("${s}")
    # file_name (main.py) has no seed component: rescue a per-seed copy
    # before the next iteration can overwrite it.
    find "${RESULTS_DIR}" -maxdepth 1 -name 'MFedMC_UCI_HAR_v2_mm_*.npz' ! -name '*_seed*' \
      -newermt "@${SEED_START}" | while read -r f; do cp "${f}" "${f%.npz}_seed${s}.npz"; done
  fi
done

# mean over the finished seeds, in the Results-sheet columns
if (( ${#DONE[@]} )); then
  "${PYTHON}" "${UCI_HAR}/summarize_seeds.py" --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
