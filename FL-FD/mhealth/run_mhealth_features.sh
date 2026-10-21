#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# mHealth with **FL-FD** (Qi et al., Information Fusion 2023), feature-vector
# variant: the mHealth counterpart of FL-FD/uci_har/run_uci_har_features.sh.
# main.py here is a wrapper that runs FL-FD/uci_har/main.py unchanged with
# HybridSketchMFedMC_v2's load_mhealth_mm as the loader.
#
#   • data: datasets/mhealth_mm -- the Acc 177 / Gyro 118 / Mag 118 / ECG 43
#     feature vectors Hybrid v2 reads (its run_mhealth.sh); same 10-client
#     Dirichlet(0.1) split and test set, 12 classes
#   • input: per modality feature vector -> GAF (difference, as released)
#     -> 70 x 70; the magnitude step is skipped (no x/y/z per time step
#     to combine); one channel per modality, so 4 x 70 x 70
#   • algorithm: unchanged from FL-FD/fixed/ (FLFDNet, local_train,
#     fedavg): 8 of 10 clients per round (paper: 12 of 15), 3 local
#     epochs, SGD lr 1e-3, momentum 1e-4, bs 32, equal-weight average
#   • comm: K x full float state_dict down and up per round; round-1
#     download (shared init) not billed; MB = 1e6 bytes. 32,364 B per
#     model (4-ch first conv, 12-class fc1) -> ~0.259 MB upload per round;
#     97 rounds ~ 25.1 MB, matching Hybrid v2 mHealth's 25.20 MB
#   • test_acc = 8-sample shard mean (as the baselines)
#   • log -> FL-FD/mhealth/result_updated/FLFD_mhealth_mm_mm_<cum_up>MB_....txt
#   • npz -> FL-FD/mhealth/results/FLFD_mhealth_mm_K8_E3_R97_seed42.npz
#
# Extra args are forwarded, e.g.:
#   ./FL-FD/mhealth/run_mhealth_features.sh --rounds 2 --device cpu
#   ./FL-FD/mhealth/run_mhealth_features.sh --gaf_method summation   # the paper's formula
# Seeds: edit NUM_SEEDS / SEEDS below, or override per call, e.g.
#   NUM_SEEDS=3 SEEDS="7 8 9" ./FL-FD/mhealth/run_mhealth_features.sh
# A failed seed is skipped (the next seeds still run) and listed at the end.
# After the seeds, FL-FD/uci_har/summarize_seeds.py prints/saves their mean in the
# Results-sheet columns: result_updated/FLFD_mhealth_mm_mm_<up>MB_it<R>_K<K>_seeds<...>_summary.txt
#
# Requires: datasets/mhealth_mm/data.npz
# (see CommEfficient/CommEfficient/prepare_mhealth_mm.py)
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/mhealth_mm}"
PYTHON="${PYTHON:-${ROOT}/venv/bin/python}"   # needs pyts + Pillow (MFedMC/.venv lacks them)
NUM_SEEDS="${NUM_SEEDS:-1}"         # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-42 43 44 45 46}"    # seed values, space-separated

cd "${ROOT}"
read -r -a SEED_LIST <<< "${SEEDS}"
FAILED=()
DONE=()
START="$(date +%s)"
for s in "${SEED_LIST[@]:0:${NUM_SEEDS}}"; do
  echo "=== seed ${s} ==="
  if ! "${PYTHON}" "${HERE}/main.py" \
    --dataset_dir "${DATA_PATH}" \
    --num_clients 10 \
    --dirichlet_alpha 0.1 \
    --rounds 97 \
    --clients_per_round 8 \
    --local_epochs 3 \
    --lr 1e-3 \
    --momentum 1e-4 \
    --bs 32 \
    --gaf_method difference \
    --size 70 \
    --valid_batch_size 8 \
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
  "${PYTHON}" "${ROOT}/FL-FD/uci_har/summarize_seeds.py" --dataset "$(basename "${DATA_PATH}")" \
    --results_dir "${HERE}/results" --log_dir "${HERE}/result_updated" \
    --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
