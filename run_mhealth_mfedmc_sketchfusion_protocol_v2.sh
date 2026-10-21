#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# MFedMC v2 on mHealth Acc/Gyro/Mag/ECG features, SketchFusionB4 protocol,
# IC-style per-modality encoder (Level 2):
#   raw Acc/Gyro/Mag/ECG -> Extractor (in_dim->feat_dim->feat_dim, IC-shaped)
#                         -> Refiner (sigmoid gate, same as IC; toggle with
#                            --no-use_refiner)
#                         -> small classifier (log-probs)
# No shared/joint fusion is added — each modality is still trained and
# uploaded independently with MFedMC's SHAP-driven modality selection and
# loss-driven client selection (highest-loss clients, as in UCI HAR v2 /
# HybridSketchMFedMC_v2; --no-prefer-higher-loss for the paper's Eq. 18).
# Logged test_acc = argmax of summed per-modality log-probs on the shared
# test set, averaged over 8-sample shards (IC convention); the per-client
# RandomForest late-fusion eval is disabled in federated.py.
#
#   • clients = 10-client Dirichlet(alpha=0.1) partition
#   • all of each client's data is training data (no local hold-out)
#   • fusion scored on the official shared mHealth test set
#   • local_epochs=5, client_select_ratio=0.2, top_shap=2
#
# Sibling of run_mhealth_mfedmc_sketchfusion_protocol.sh, pointed at
# MFedMC/mHealth_v2 instead of MFedMC/mHealth (v1). Self-contained; does
# not modify MFedMC/mHealth (v1) or its run script.
#
# Features come from datasets/mhealth_mm_random_group/data.npz (see
# prepare_mhealth_mm.py).
#
# Optional multi-seed sweep (opt-in; default behavior above is unchanged):
# set NUM_SEEDS and/or SEEDS in the environment to run several seeds back to
# back and print/save their mean, e.g.:
#   NUM_SEEDS=3 SEEDS="42 43 44" ./run_mhealth_mfedmc_sketchfusion_protocol_v2.sh
#   SEEDS="7 8 9" ./run_mhealth_mfedmc_sketchfusion_protocol_v2.sh
# Do NOT pass --seed yourself as an extra arg in sweep mode: "$@" is appended
# after --seed "${s}" on every iteration, and argparse keeps the last
# occurrence of a flag, so a user-supplied --seed would silently override
# every seed of the sweep.
# A failed seed is skipped (the remaining seeds still run) and listed at the
# end, with a final nonzero exit if anything failed. Each seed's npz is also
# copied to results/MFedMC_mHealth_v2_..._seed<s>.npz (the runner's own npz
# name has no seed, so the next seed's run would otherwise silently overwrite
# it).
# After the seeds, summarize_seeds.py prints/saves their mean in the
# Results-sheet columns: result_updated/MFedMC_mHealth_v2_..._seeds<...>_summary.txt
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
MHEALTH="${ROOT}/MFedMC/mHealth_v2"
# DATA_PATH="${DATA_PATH:-${ROOT}/datasets/mhealth_mm_random_group}"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/mhealth_mm}"


if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

# Flags shared by every run; --seed is appended separately below (per branch).
COMMON_ARGS=(
  --dataset_dir "${DATA_PATH}"
  --partition dirichlet
  --num_clients 10
  --dirichlet_alpha 0.1
  --num_classes 12
  --acc_dim 177
  --gyro_dim 118
  --mag_dim 118
  --ecg_dim 43
  --feat_dim 512
  --use_refiner
  --iterations 10
  --local_epochs 5
  --top_shap 2
  --client_select_ratio 0.2
  --train_ratio 1.0
  --eval_on_global_test
)

cd "${MHEALTH}"

if [[ -z "${NUM_SEEDS:-}" && -z "${SEEDS:-}" ]]; then
  # Default path: single exec, --seed 42.
  exec "${PYTHON}" main.py "${COMMON_ARGS[@]}" --seed 42 "$@"
fi

# --- Opt-in multi-seed sweep -------------------------------------------------
NUM_SEEDS="${NUM_SEEDS:-1}"         # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-42 43 44 45 46}"    # seed values, space-separated

read -r -a SEED_LIST <<< "${SEEDS}"
RESULTS_DIR="${MHEALTH}/results"
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
    find "${RESULTS_DIR}" -maxdepth 1 -name 'MFedMC_mHealth_v2_mm_*.npz' ! -name '*_seed*' \
      -newermt "@${SEED_START}" | while read -r f; do cp "${f}" "${f%.npz}_seed${s}.npz"; done
  fi
done

# mean over the finished seeds, in the Results-sheet columns
if (( ${#DONE[@]} )); then
  "${PYTHON}" "${MHEALTH}/summarize_seeds.py" --since "${START}" --seeds "${DONE[@]}" \
    --dataset "$(basename "${DATA_PATH}")" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
