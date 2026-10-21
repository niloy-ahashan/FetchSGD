#!/usr/bin/env bash
set -uo pipefail

# ---------------------------------------------------------------
# UCI HAR: HybridSketchMFedMC v2's algorithm with FL-FD's low-parameter model.
# Nothing in HybridSketchMFedMC_v2/ is modified: main.py here imports and runs
# Hybrid v2's own code and swaps only the model and the input step (in memory).
#
#   • algorithm: Hybrid v2 unchanged -- loss-based client selection (0.4 -> 4 of
#     10), SHAP modality selection (1 of 2), gradient masking, FetchSGD Count
#     Sketch + virtual momentum + error feedback, LR schedule, wd 5e-4
#   • model (--model branch): per modality FL-FD's 3 conv blocks on a 70 x 70 GAF
#     image (900 params each) -> Hybrid's 2 x 128 Count-Sketch fusion -> FL-FD's
#     single Linear(256, 6): 3,342 params (Hybrid v2's own model: 2,261,510)
#     --model flfd = FL-FD's network unchanged, 4,461 params
#   • data: datasets/uci_har_mm, same split and test set as Hybrid v2
#   • sketch: 3 x 500, k = 500 (Hybrid's 3 x 20000 would be larger than the
#     model). Upload per client = 3 x 500 x 4 B = 6,000 B; 4 clients per round
#     = 0.024 MB per round. Rounds (--num_epochs) per budget:
#         10 MB: 417   20 MB: 834   30 MB: 1250   40 MB: 1667   50 MB: 2084
#   • logs -> HybridSketchMFedMC_v2_flfd/result_updated/HybridV2FD_branch_..._seed<s>.txt
#   • npz  -> HybridSketchMFedMC_v2_flfd/results/HybridV2FD_branch_..._seed<s>.npz
#   • after the seeds: summarize_seeds.py writes the Results-sheet row
#
# Seeds: edit NUM_SEEDS / SEEDS / NUM_EPOCHS below, or override per call, e.g.
#   NUM_SEEDS=3 SEEDS="7 8 9" NUM_EPOCHS=417 ./HybridSketchMFedMC_v2_flfd/run_uci_har.sh
# Extra args go to every seed, e.g. --num_cols 1000 --k 1000, --model flfd,
# --fusion_mode sum. Do not pass --seed or --num_epochs as extra args (use the
# variables). A failed seed is skipped (the next seeds still run) and listed.
#
# Requires: datasets/uci_har_mm/data.npz; root venv (pyts, Pillow, shap, sklearn)
# ---------------------------------------------------------------

NUM_SEEDS="${NUM_SEEDS:-1}"            # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-42 43 44 45 46}"       # seed values, space-separated
NUM_EPOCHS="${NUM_EPOCHS:-417}"       # rounds; 2084 = 50 MB with 3 x 500 and 4 clients

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/uci_har_mm}"
PYTHON="${PYTHON:-${ROOT}/venv/bin/python}"   # MFedMC/.venv (Hybrid's usual) lacks pyts + Pillow

# Hybrid v2's run_uci_har.sh flags, except: no --feat_dim / --mm_dropout (the
# FL-FD model has its own widths and Dropout2d 0.2), --num_cols 500 --k 500, and
# --num_epochs from NUM_EPOCHS; plus FL-FD's input flags.
COMMON_ARGS=(
  --dataset_dir "${DATA_PATH}"
  --num_classes 6
  --num_clients 10
  --dirichlet_alpha 0.1
  --acc_dim 348
  --gyro_dim 213
  --fusion_mode sketch
  --sketch_r 2
  --sketch_c 128
  --local_batch_size -1
  --upload_object gradient
  --weight_decay 5e-4
  --virtual_momentum 0.9
  --error_type virtual
  --lr_scale 0.05
  --mode sketch
  --k 500
  --num_rows 3
  --num_cols 500
  --num_epochs "${NUM_EPOCHS}"
  --pivot_epoch 15
  --local_epochs 1
  --num_blocks 1
  --num_select_modalities 1
  --client_select_ratio 0.4
  --client_select loss
  --device cuda
  --model branch
  --gaf_method difference
  --size 70
)

cd "${ROOT}"
read -r -a SEED_LIST <<< "${SEEDS}"
FAILED=()
DONE=()
START="$(date +%s)"
for s in "${SEED_LIST[@]:0:${NUM_SEEDS}}"; do
  echo "=== seed ${s} ==="
  if ! "${PYTHON}" "${HERE}/main.py" "${COMMON_ARGS[@]}" --seed "${s}" "$@"; then
    echo "!!! seed ${s} failed; continuing with the next seed" >&2
    FAILED+=("${s}")
  else
    DONE+=("${s}")
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
