#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# mHealth with **PMFH + federated head**, nmod version: same setup
# as run_uci_har_pmfh_fedhead.sh (official PMFH training + Linear
# head on nec_vec.detach(), CE-trained, FedAvg'd with the global
# model; test_acc = head argmax, 8-sample shard mean), on the shared
# mHealth split (10 clients, Dirichlet 0.1, 12 classes).
#
#   • modalities: one PMFH encoder + gate per modality (Acc, Gyro,
#     Mag, ECG), hash of the summed gated features (layers_nmod.py)
#   • log -> PMFH/result_updated/PMFHfhN_mhealth_mm_random_group_mm_<cum_up>MB_....txt
#
# Extra args are forwarded, e.g.:
#   ./PMFH/run_mhealth_pmfh_fedhead_nmod.sh --rounds 2 --train_ep 1 --train_ep_private 1
# Seeds: NUM_SEEDS=3 SEEDS="7 8 9" ./PMFH/run_mhealth_pmfh_fedhead_nmod.sh
# (don't pass --seed as an extra arg: it would override every seed of the loop)
# A failed seed is skipped (the next seeds still run) and listed at the end.
# Each seed's npz is also copied to results/PMFHfhN_mHealth_bit<b>_h<h>_seed<s>.npz.
# After the seeds, summarize_seeds_mhealth.py prints/saves their mean:
#   result_updated/PMFHfhN_mhealth_mm_random_group_mm_<up>MB_it<R>_bit<b>_h<h>_c<c>_seeds<...>_summary.txt
#
# Requires: datasets/mhealth_mm_random_group/data.npz
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
# DATA_PATH="${DATA_PATH:-${ROOT}/datasets/mhealth_mm_random_group}"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/mhealth_mm}"


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
  if ! "${PYTHON}" "${HERE}/main_mhealth_pmfh.py" \
  --mod_mode nmod \
  --clf_weight 1.0 \
  --head_lr 1e-3 \
  --dataset_dir "${DATA_PATH}" \
  --num_clients 10 \
  --dirichlet_alpha 0.1 \
  --classes 12 \
  --bit 64 \
  --hidden_dim 64 \
  --common_dim 32 \
  --rounds 1 \
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
    find "${HERE}/results" -maxdepth 1 -name 'PMFHfhN_mHealth_bit*_h*.npz' ! -name '*_seed*' \
      -newermt "@${SEED_START}" | while read -r f; do cp "${f}" "${f%.npz}_seed${s}.npz"; done
  fi
done

# mean over the finished seeds, in the Results-sheet columns
if (( ${#DONE[@]} )); then
  "${PYTHON}" "${HERE}/summarize_seeds_mhealth.py" --prefix PMFHfhN --dataset mhealth_mm --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
