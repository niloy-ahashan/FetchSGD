# FetchSGD: Communication-Efficient Multimodal Federated Learning

Research code for **multimodal federated learning under tight communication budgets**. It combines two ideas:

- **FetchSGD** ([Rothchild et al., 2020](https://arxiv.org/abs/2007.07682)): clients compress gradients into a Count Sketch before uploading, and the server recovers an approximate update from the aggregated sketch.
- **MFedMC** ([arXiv:2401.16685](https://arxiv.org/abs/2401.16685)): each round, only a subset of clients and modalities upload, chosen by a SHAP-, model-size- and recency-based priority score.

On top of these, the repo adds **SketchFusion**, which reuses the Count Sketch structure to fuse modality features inside the model. It also includes **HybridSketchMFedMC**, which drives FetchSGD-compressed training with MFedMC-style client and modality selection.

## Methods compared

| Method | Where | Fusion | Upload |
|---|---|---|---|
| SketchFusion A / B / C | `CommEfficient/` | Count Sketch of modality features (A: learned hashes, B: fixed random hashes + MLP head, C: hybrid) | FetchSGD gradient sketch |
| IndependentCompression | `CommEfficient/` | Element-wise sum of refined modality features | FetchSGD gradient sketch |
| FedAvg / Top-K | `CommEfficient/` (`--mode fedavg` / `--mode topk`) | Any of the above | Full update / top-K sparsified |
| MFedMC (v1, v2) | `MFedMC/` | Per-modality encoders + local RandomForest late fusion | Full weights for selected clients/modalities |
| HybridSketchMFedMC (v1, v2) | `HybridSketchMFedMC/`, `HybridSketchMFedMC_v2/` | `--fusion_mode sketch` (SketchFusionB) or `sum` (IndependentCompression) | FetchSGD sketch for MFedMC-selected clients/modalities |

`MFedMC/*_v2` swaps in IndependentCompression-shaped encoders (Extractor + optional Refiner). `HybridSketchMFedMC_v2` uses FetchSGD-style example-weighted sketch aggregation, and uploads gradients by default (`--upload_object gradient|delta`).

## Repository layout

```
FetchSGD/
├── CommEfficient/CommEfficient/     # FetchSGD engine + SketchFusion / IndependentCompression
│   ├── mm_train.py                  # 2-modality entry point (SketchFusionA/B/C, MultiModal)
│   ├── mm_train_actionsense_4mod.py # 4-modality entry point (SketchFusionB4)
│   ├── mm_train_independent.py      # IndependentCompression, 2 modalities
│   ├── mm_train_independent_4mod.py # IndependentCompression, 4 modalities
│   ├── fed_aggregator.py            # Server: sketch unrolling, virtual momentum, error feedback
│   ├── fed_worker.py                # Clients (one torch.multiprocessing process each)
│   ├── models/                      # sketch_fusion_nets, independent_compression_net(_4), multimodal_net, ResNets
│   ├── data_utils/                  # Fed* dataset classes + Dirichlet FedSampler
│   ├── prepare_*.py                 # Raw data → feature caches under datasets/
│   └── utils.py                     # Shared CLI (parse_args) and param-vector helpers
├── MFedMC/                          # Self-contained MFedMC loops, one folder per dataset
│   ├── UCI_HAR/, UCI_HAR_v2/
│   ├── mHealth/, mHealth_v2/
│   └── ActionSense/                 # Original upstream MFedMC implementation
├── HybridSketchMFedMC/              # Hybrid v1 (local SGD, client-weighted delta sketches)
├── HybridSketchMFedMC_v2/           # Hybrid v2 (FetchSGD-style gradient sketches)
├── csh/                             # csvec Count Sketch library
├── PMFH/, factorized/, ActionNet/   # Reference code from other papers / dataset tooling
└── run_*.sh                         # One launcher per (dataset × method) config
```

## Installation

Requires Python 3.10+ and a CUDA GPU; the run scripts default to `--device cuda`.

```bash
git clone https://github.com/niloy-ahashan/FetchSGD.git
cd FetchSGD

python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

pip install -e csh             # csvec Count Sketch library
pip install -e CommEfficient   # CommEfficient package
```

MFedMC and HybridSketchMFedMC scripts use `MFedMC/.venv/bin/python` if it exists and otherwise fall back to `python3`. To use a separate environment for them:

```bash
python -m venv MFedMC/.venv
MFedMC/.venv/bin/pip install -r MFedMC/requirements.txt
```

## Data preparation

Datasets and feature caches live under `datasets/`, which is gitignored.

**UCI HAR** (Accelerometer 348-D + Gyroscope 213-D, 6 classes):

```bash
python CommEfficient/CommEfficient/prepare_uci_har_mm.py \
  --uci_root "human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset" \
  --out_dir  datasets/uci_har_mm
```

**mHealth** (Acc / Gyro / Mag / ECG, 12 classes). `--test_split` chooses how windows are split between train and test:

| `--test_split` | Behavior |
|---|---|
| `random` (default) | Class-stratified random split over pooled windows. Windows overlap by 50%, so near-duplicate windows can end up on both sides of the split. |
| `random_group` | Random split over whole activity runs, so overlapping windows always stay on the same side. |
| `subject` | Fixed holdout: subjects 1–8 train, 9–10 test. |

Most mHealth run scripts default to the `random_group` cache:

```bash
python CommEfficient/CommEfficient/prepare_mhealth_mm.py \
  --mhealth_root datasets/mhealth+dataset/MHEALTHDATASET \
  --out_dir      datasets/mhealth_mm_random_group \
  --test_split   random_group
```

To use a different cache, set `DATA_PATH`, for example `DATA_PATH=datasets/mhealth_mm bash run_mhealth_...sh`. `run_mhealth_sketch_fusion_B_4mod.sh` reads from `datasets/mhealth_mm`.

Other `prepare_*.py` scripts (ActionSense, MELD, PTB-XL, VQA, WISDM, and others) are kept for reference, but they no longer have launchers.

## Running experiments

Always launch through a root `run_*.sh` script. Each one sets `PYTHONPATH` and pins the full set of flags for its configuration. Extra arguments are forwarded to the underlying `main.py` / `mm_train*.py`.

### UCI HAR

| Script | Method |
|---|---|
| `run_uci_har_sketch_fusion_{A,B,C}.sh` | SketchFusion A / B / C + FetchSGD |
| `run_uci_har_independent_compression.sh` | IndependentCompression + FetchSGD |
| `run_uci_har_fedavg.sh` | FedAvg baseline (no compression) |
| `run_uci_har_mfedmc_sketchfusion_protocol{,_v2}.sh` | MFedMC v1 / v2 on the SketchFusion split (10 Dirichlet clients, official test set) |
| `run_uci_har_mfedmc_from_sketchfusion_data.sh` | MFedMC v1, ActionSense-style protocol (one client per subject, local 80/20 split) |
| `run_uci_har_hybrid_sketch_mfedmc{,_v2}.sh` | HybridSketchMFedMC v1 / v2 |
| `run_uci_har_comm_budget.sh` | Test accuracy under a total communication budget (`COMM_MB=5`) |
| `run_uci_har_mm_comm_overhead.sh` | Communication (MiB) needed to reach a target test accuracy |

### mHealth

| Script | Method |
|---|---|
| `run_mhealth_sketch_fusion_B_4mod.sh` | SketchFusionB4 + FetchSGD |
| `run_mhealth_independent_compression_4mod.sh` | IndependentCompression4 + FetchSGD |
| `run_mhealth_mfedmc_sketchfusion_protocol{,_v2}.sh` | MFedMC v1 / v2 |
| `run_mhealth_mfedmc_from_sketchfusion_data.sh` | MFedMC v1, alternate protocol |
| `run_mhealth_hybrid_sketch_mfedmc{,_v2}.sh` | HybridSketchMFedMC v1 / v2 |

### Quick start

```bash
bash run_uci_har_sketch_fusion_B.sh
bash run_uci_har_hybrid_sketch_mfedmc_v2.sh --client_select random
bash run_uci_har_mfedmc_sketchfusion_protocol_v2.sh --iterations 50
```

### Key CommEfficient flags

| Flag | Meaning |
|---|---|
| `--mode {sketch,fedavg,topk}` | Upload compression strategy |
| `--num_rows`, `--num_cols`, `--k` | Count Sketch size and number of recovered coordinates |
| `--num_clients`, `--num_workers` | Total clients / clients per round |
| `--dirichlet_alpha` | Non-IID label skew (default experiments use 0.1) |
| `--mm_local_epochs` | Local SGD epochs before one compressed upload; does not change rounds per epoch |
| `--max_comm_megabytes`, `--target_test_acc` | Communication-budget and communication-to-target experiments |

When adding a new configuration, create a new `run_*.sh` rather than editing an existing one. Existing scripts record the settings of earlier runs.

## Outputs

Each subproject writes to its own `results/` directory (gitignored), with hyperparameters encoded in the filename, for example `MFedMC/UCI_HAR/results/MFedMC_UCI_HAR_mm_dirichlet_Top_1_...npz`. Result files hold per-round accuracy, upload and download bytes, elapsed time, and client/modality selection history.

## Tests

```bash
python -m unittest csh/csvec/test_csvec.py
```

`CommEfficient/CommEfficient/{grad_test,unit_test,test_gradients,pytorch_test}.py` are standalone diagnostic scripts, not a test suite.

## Citation

```bibtex
@article{niloy2024fetchsgd,
  title  = {Communication-Efficient Multimodal Federated Learning with Sketch-Based Gradient Compression},
  author = {Niloy, Ahashan Habib},
  year   = {2024}
}
```

This work builds on FetchSGD (Rothchild et al., ICML 2020) and MFedMC (arXiv:2401.16685); please cite those as well.

## License

See the per-directory licenses: `csh/LICENSE`, `MFedMC/LICENSE`, `ActionNet/LICENSE`, `factorized/LICENSE`.
