# FetchSGD: Communication-Efficient Multimodal Federated Learning with Sketch-Based Gradient Compression

This repository implements **sketch-based gradient compression** for multimodal federated learning. It extends the FetchSGD framework with novel fusion strategies (SketchFusion) that use Count Sketch data structures for both gradient communication and modality fusion.

## Project Structure

```
FetchSGD/
├── CommEfficient/          # Core federated learning engine
│   └── CommEfficient/
│       ├── mm_train.py              # Multimodal training entry point
│       ├── mm_sketch_fusion.py      # Sketch-based fusion training
│       ├── fed_aggregator.py        # Server-side aggregation
│       ├── fed_worker.py            # Client-side training
│       ├── models/                  # Neural network architectures
│       │   ├── multimodal_net.py
│       │   └── sketch_fusion_nets.py
│       ├── data_utils/              # Dataset loaders and partitioning
│       └── utils.py                 # Shared utilities and CLI args
├── csh/                    # Count Sketch Vector (CSVec) library
├── MFedMC/                 # MFedMC: Joint Modality and Client Selection
├── ActionNet/              # ActionSense dataset recording/parsing tools
├── PMFH/                   # Pairwise Multimodal Federated Hashing
├── factorized/             # Factorized multimodal models
├── run_*.sh                # Experiment launch scripts
└── requirements.txt        # Python dependencies
```

## Supported Datasets

| Dataset | Modalities | Script Examples |
|---------|-----------|-----------------|
| UCI HAR | Accelerometer + Gyroscope | `run_uci_har_sketch_fusion_B.sh` |
| ActionSense | Eye, EMG, Tactile, IMU | `run_actionsense_s00_sketch_fusion_B.sh` |
| MELD | Audio + Text (sentiment) | `run_meld_sketch_fusion_B.sh` |
| PTB-XL | ECG leads | `run_ptbxl_sketch_fusion_B.sh` |
| Atrial Fibrillation | ECG signals | `run_atrial_fibrillation_sketch_fusion_B.sh` |
| Heartbeat | ECG signals | `run_heartbeat_sketch_fusion_B.sh` |
| Face Detection | EEG + EMG | `run_face_detection_sketch_fusion_B.sh` |
| Spoken Arabic Digits | MFCCs | `run_spoken_arabic_digits_sketch_fusion_B.sh` |
| VQA | Image + Text | `run_vqa_sketch_fusion_B.sh` |

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/niloy-ahashan/FetchSGD.git
cd FetchSGD

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Install the Count Sketch library (required for sketch mode)
cd csh
pip install -e .
cd ..

# Install CommEfficient package
cd CommEfficient
pip install -e .
cd ..
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.5.1+cu121 | Deep learning framework |
| torchvision | 0.20.1+cu121 | Vision models and transforms |
| torchaudio | 2.5.1+cu121 | Audio processing |
| numpy | 2.1.2 | Numerical computation |
| scipy | 1.15.3 | Scientific computing |
| scikit-learn | 1.7.2 | ML utilities and metrics |
| h5py | 3.15.1 | HDF5 dataset I/O |
| pandas | 2.3.3 | Data manipulation |
| matplotlib | 3.10.7 | Plotting |
| librosa | 0.11.0 | Audio feature extraction |
| transformers | 5.5.0 | NLP models (MELD, VQA) |
| tensorboard | 2.20.0 | Training visualization |
| wfdb | 4.3.1 | PhysioNet ECG data |
| opensmile | 2.6.0 | Audio feature extraction |

## Usage

### Quick Start: UCI HAR with SketchFusion

```bash
# Prepare the dataset
python CommEfficient/CommEfficient/prepare_uci_har_mm.py

# Run training
bash run_uci_har_sketch_fusion_B.sh
```

### General Pattern

Each `run_*.sh` script launches a federated learning experiment. Key arguments:

```bash
python CommEfficient/CommEfficient/mm_train.py \
  --dataset_dir <path>        # Path to prepared dataset
  --model SketchFusionB       # Fusion model (SketchFusionA/B/C, MultiModal)
  --mode sketch               # Communication mode (sketch, fedavg, topk)
  --num_clients 10            # Number of federated clients
  --num_epochs 40             # Training rounds
  --num_rows 3                # Sketch rows (for gradient compression)
  --num_cols 5000             # Sketch columns
  --device cuda               # Device (cuda or cpu)
```

### Training Modes

- **`sketch`** — FetchSGD: compress gradients via Count Sketch before upload
- **`fedavg`** — Standard Federated Averaging (full gradient upload)
- **`topk`** — Top-K sparsification

### Fusion Models

- **`SketchFusionA`** — Learned sketch hash functions for fusion
- **`SketchFusionB`** — Fixed random Count Sketch + MLP head (recommended)
- **`SketchFusionC`** — Hybrid sketch fusion variant
- **`MultiModal`** — Standard concatenation-based multimodal fusion

## Citation

If you find this work useful, please cite:

```bibtex
@article{niloy2024fetchsgd,
  title={Communication-Efficient Multimodal Federated Learning with Sketch-Based Gradient Compression},
  author={Niloy, Ahashan Habib},
  year={2024}
}
```

## License

See individual subdirectory licenses (`ActionNet/LICENSE`, `MFedMC/LICENSE`, `csh/LICENSE`, `factorized/LICENSE`).
