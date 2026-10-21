#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR with **PMFH + federated head** (option B): official PMFH
# (github.com/vindahi/PMFH) plus a classifier head trained during
# federated training, like IC / MFedMC v2 / Hybrid v2.
#
#   • PMFH itself untouched: upstream Layers (gate, sizes), upstream
#     LocalUpdate losses, clipping, prototypes, FedAvg
#   • head = Linear(common_dim -> 6) on nec_vec.detach(): trained with
#     CrossEntropy on each client, uploaded and FedAvg'd with the
#     global model; its gradient never reaches PMFH's parameters
#   • test_acc = trained head argmax, 8-sample shard mean (as the
#     baselines)
#   • flags mirror run_uci_har_pmfh.sh so the two are directly comparable
#   • log -> PMFH/result_updated/PMFHfh_..._<cum_up>MB_....txt
#
# Extra args are forwarded, e.g.:
#   ./PMFH/run_uci_har_pmfh_fedhead.sh --rounds 2 --train_ep 1 --train_ep_private 1
# Seeds: edit NUM_SEEDS / SEEDS below, or override per call, e.g.
#   NUM_SEEDS=3 SEEDS="7 8 9" ./PMFH/run_uci_har_pmfh_fedhead.sh
# (don't pass --seed as an extra arg: it would override every seed of the loop)
# A failed seed is skipped (the next seeds still run) and listed at the end.
# Each seed's npz is also copied to results/PMFHfh_UCI_HAR_bit<b>_h<h>_seed<s>.npz.
# After the seeds, summarize_seeds.py prints/saves their mean in the Results-sheet
# columns: result_updated/PMFHfh_uci_har_mm_mm_<up>MB_it<R>_bit<b>_h<h>_c<c>_seeds<...>_summary.txt
#
# Requires: datasets/uci_har_mm/data.npz
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/uci_har_mm}"

if [[ -x "${ROOT}/venv/bin/python" ]]; then
  PYTHON="${ROOT}/venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

NUM_SEEDS="${NUM_SEEDS:-5}"         # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-42 43 44 45 46}"    # seed values, space-separated

cd "${HERE}"
read -r -a SEED_LIST <<< "${SEEDS}"
FAILED=()
DONE=()
START="$(date +%s)"
for s in "${SEED_LIST[@]:0:${NUM_SEEDS}}"; do
  echo "=== seed ${s} ==="
  SEED_START="$(date +%s)"
  if ! "${PYTHON}" "${HERE}/main_uci_har_v2.py" \
  --variant fedhead \
  --clf_weight 1.0 \
  --head_lr 1e-3 \
  --dataset_dir "${DATA_PATH}" \
  --num_clients 10 \
  --dirichlet_alpha 0.1 \
  --classes 6 \
  --image_dim 348 \
  --text_dim 213 \
  --bit 64 \
  --hidden_dim 128 \
  --common_dim 32 \
  --rounds 10 \
  --batch_size 64 \
  --train_ep 10 \
  --train_ep_private 10 \
  --seed "${s}" \
  --device cuda \
  "$@"; then
    echo "!!! seed ${s} failed; continuing with the next seed" >&2
    FAILED+=("${s}")
  else
    DONE+=("${s}")
    # the runner's npz name has no seed (next seed would overwrite it): keep a per-seed copy
    find "${HERE}/results" -maxdepth 1 -name 'PMFHfh_UCI_HAR_bit*_h*.npz' ! -name '*_seed*' \
      -newermt "@${SEED_START}" | while read -r f; do cp "${f}" "${f%.npz}_seed${s}.npz"; done
  fi
done

# mean over the finished seeds, in the Results-sheet columns
if (( ${#DONE[@]} )); then
  "${PYTHON}" "${HERE}/summarize_seeds.py" --prefix PMFHfh --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
