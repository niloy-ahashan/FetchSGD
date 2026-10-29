#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# mHealth with **FedMobile** (Liu, Wang, Yuan, WWW'25), PyTorch port of
# FedMobile/upstream/USC-HAD/MMFL_main next to IC / MFedMC v2 / Hybrid v2 / FLASH.
#
#   • data: same split as the other mHealth baselines
#     (Acc 177 + Gyro 118 + Mag 118 + ECG 43, 12 classes, 10-client Dirichlet(0.1), official test set)
#   • model: Hybrid v2 Extractor+Refiner per modality (64-D) -> concat
#     -> one Linear (released MyMMModel.classifier); generator G(eps, y)
#   • recipe (released): all clients every round, 1 local epoch, Adam 1e-3,
#     wd 1e-4, bs 16, rounds as set below (released: 150); KD from the generator, contribution-aware
#     model aggregation, k-means + Shapley generator aggregation
#   • missing modality: released setting, modality 0 (Acc) zeroed in 80% of each batch
#   • test_acc = 8-sample shard mean (as the baselines)
#   • log -> FedMobile/torch_port/result_updated/FedMobile_..._<cum_up>MB_....txt
#
# Extra args are forwarded, e.g.:
#   ./FedMobile/torch_port/run_mhealth_miss.sh --rounds 2
#   ./FedMobile/torch_port/run_mhealth_miss.sh --distill 0          # released model_agg_org (plain FedAvg)
#   ./FedMobile/torch_port/run_mhealth_miss.sh --train_generator 1  # also step the generators (paper intent)
# Seeds: edit NUM_SEEDS / SEEDS below, or override per call, e.g.
#   NUM_SEEDS=1 ./FedMobile/torch_port/run_mhealth_miss.sh             # seed 42 only
#   NUM_SEEDS=3 SEEDS="7 8 9" ./FedMobile/torch_port/run_mhealth_miss.sh
# A failed seed is skipped (the next seeds still run) and listed at the end.
# After the seeds, summarize_seeds.py prints/saves their mean in the Results-sheet
# columns: result_updated/FedMobile_<data>_mm_<up>MB_it<R>_K10_E1[_tags]_seeds<...>_summary.txt
#
# Requires: datasets/mhealth_mm_random_group/data.npz
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
# DATA_PATH="${DATA_PATH:-${ROOT}/datasets/mhealth_mm_random_group}"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/mhealth_mm}"


if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
NUM_SEEDS="${NUM_SEEDS:-5}"         # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-42 43 44 45 46}"    # seed values, space-separated

cd "${ROOT}"
read -r -a SEED_LIST <<< "${SEEDS}"
FAILED=()
DONE=()
START="$(date +%s)"
for s in "${SEED_LIST[@]:0:${NUM_SEEDS}}"; do
  echo "=== seed ${s} ==="
  if ! "${PYTHON}" "${HERE}/main.py" \
    --dataset mhealth \
    --dataset_dir "${DATA_PATH}" \
    --num_clients 10 \
    --dirichlet_alpha 0.1 \
    --rounds 3 \
    --local_epochs 5 \
    --feat_dim 64 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --bs 16 \
    --miss_rate 0.8 \
    --miss_modality 0 \
    --seed "${s}" \
    --device cuda \
    "$@"; then
    echo "!!! seed ${s} failed; continuing with the next seed" >&2
    FAILED+=("${s}")
  else
    DONE+=("${s}")
  fi
done

# mean over the finished seeds, in the Results-sheet columns
if (( ${#DONE[@]} )); then
  "${PYTHON}" "${HERE}/summarize_seeds.py" --dataset "$(basename "${DATA_PATH}")" \
    --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
