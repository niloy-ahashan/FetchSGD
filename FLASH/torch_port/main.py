"""FLASH baseline (PyTorch port) on the SketchFusionB-style feature splits.

Data, client partition, test metric and console table are shared with
Hybrid v2 / IC / MFedMC v2 / PMFH v2 (imported from HybridSketchMFedMC_v2).
"""

import argparse
import io
import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(ROOT, "HybridSketchMFedMC_v2"))

import numpy as np
import torch

from dataset import load_mhealth_mm, load_uci_har_mm  # HybridSketchMFedMC_v2
from flash_federated import run_flash
from flash_model import BRANCH_DIMS, FUSION_DIMS, FlashNet
from utils import Timer  # HybridSketchMFedMC_v2


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def args_parser():
    p = argparse.ArgumentParser(description="FLASH (INFOCOM'22) federated baseline, PyTorch port")
    p.add_argument("--dataset", choices=["uci_har", "mhealth"], default="uci_har")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--dirichlet_alpha", type=float, default=0.1)
    # FLASH recipe (federated_learning.py / run_flash.sh)
    p.add_argument("--rounds", type=int, default=200, help="comms_round")
    p.add_argument("--local_epochs", type=int, default=100, help="max epochs per round (early stopping)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--shuffle", type=int, default=1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--val_frac", type=float, default=0.1, help="local validation split for early stopping")
    p.add_argument(
        "--policy", type=str, default="all",
        help="'all' = sync every branch (released run_flash.sh); or M+1 comma-separated "
             "probabilities for [mod_0..mod_{M-1}, fusion] (paper's random-branch policy)",
    )
    p.add_argument("--weighted_avg", action="store_true", help="sample-weighted average (FLASH: unweighted)")
    p.add_argument(
        "--branch", choices=["gps", "shared"], default="gps",
        help="'gps' = FLASH's coord_mlp branch per modality (as published); 'shared' = the "
             "Hybrid v2 / IC encoder (FeaExtractor + FeaRefiner) per modality",
    )
    p.add_argument("--feat_dim", type=int, default=512, help="--branch shared: encoder width (Hybrid v2: 512)")
    p.add_argument("--mm_dropout", type=float, default=0.3, help="--branch shared: encoder dropout (Hybrid v2: 0.3)")
    p.add_argument(
        "--branch_dims", type=str, default="1024,512,256",
        help="--branch gps: widths of the 3 hidden Dense layers (FLASH: 1024,512,256)",
    )
    p.add_argument(
        "--fusion_dims", type=str, default="1024,512,256,128",
        help="widths of the 4 hidden fusion-head layers (FLASH: 1024,512,256,128)",
    )
    p.add_argument("--valid_batch_size", type=int, default=8, help="test shard size (as the baselines)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--results_dir", type=str, default=os.path.join(HERE, "results"))
    return p.parse_args()


def _policy_tag(policy):
    return "PAll" if policy == "all" else "P" + "-".join(f"{float(x):g}" for x in policy.split(","))


def _dims(s):
    return tuple(int(x) for x in str(s).split(","))


def _width_tags(args):
    """[] at FLASH's widths; else e.g. ['bd128-64-32', 'fd128-64-32-16']."""
    tags = []
    if _dims(args.branch_dims) != BRANCH_DIMS:
        tags.append("bd" + "-".join(str(d) for d in _dims(args.branch_dims)))
    if _dims(args.fusion_dims) != FUSION_DIMS:
        tags.append("fd" + "-".join(str(d) for d in _dims(args.fusion_dims)))
    return tags


def _txt_stem(args, cum_up_mb):
    dataset = os.path.basename(os.path.normpath(args.dataset_dir))
    up_str = f"{cum_up_mb:.2f}MB" if cum_up_mb is not None else "NAMB"
    return (
        f"FLASH_{dataset}_mm_{up_str}_dirichlet{args.dirichlet_alpha:g}_it{args.rounds}_"
        f"{_policy_tag(args.policy)}_E{args.local_epochs}_lr{args.lr:g}_"
        f"{'wavg_' if args.weighted_avg else ''}"
        f"{'shared_' if args.branch == 'shared' else ''}"
        f"{''.join(t + '_' for t in _width_tags(args))}"
        f"clients{args.num_clients}_seed{args.seed}"
    )


def _run(args):
    timer = Timer()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print(f"Requested {args.device} but CUDA is unavailable — using {device}")

    if args.dataset == "mhealth":
        print("Loading SketchFusionB4 mHealth split (Acc/Gyro/Mag/ECG feature vectors)...")
        loader = load_mhealth_mm
    else:
        print("Loading SketchFusionB UCI HAR split (Acc / Gyro feature vectors)...")
        loader = load_uci_har_mm
    clients, global_test, meta = loader(
        args.dataset_dir,
        num_clients=args.num_clients,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed,
    )
    modalities = meta["modalities"]

    model = FlashNet(
        meta["mod_dims"], meta["num_classes"],
        branch=args.branch, feat_dim=args.feat_dim, dropout=args.mm_dropout,
        branch_dims=_dims(args.branch_dims), fusion_dims=_dims(args.fusion_dims),
    ).to(device)

    print("Finished initializing in {:.2f} seconds".format(timer()))
    print("Hyperparameters (resolved):")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")

    history = run_flash(args, model, clients, global_test, modalities, device)

    print(
        f"Final test accuracy: {float(history['test_acc'][-1]):.4f} "
        f"({100.0 * float(history['test_acc'][-1]):.2f}%)"
    )

    os.makedirs(args.results_dir, exist_ok=True)
    dataset_label = "mHealth" if args.dataset == "mhealth" else "UCI_HAR"
    file_name = os.path.join(
        args.results_dir,
        f"FLASH_{dataset_label}_{_policy_tag(args.policy)}_E{args.local_epochs}_R{args.rounds}"
        f"{'_wavg' if args.weighted_avg else ''}"
        f"{'_shared' if args.branch == 'shared' else ''}"
        f"{''.join('_' + t for t in _width_tags(args))}_seed{args.seed}.npz",
    )
    np.savez(
        file_name,
        train_loss=history["train_loss"],
        train_acc=history["train_acc"],
        test_loss=history["test_loss"],
        test_acc=history["test_acc"],
        upload_bytes=history["upload_bytes"],
        download_bytes=history["download_bytes"],
        epochs_run=history["epochs_run"],
        branch_synced=history["branch_synced"],
        branch_names=np.array(history["branch_names"]),
        modalities=np.array(modalities),
        dataset_dir=np.array(meta["dataset_dir"]),
        samples_per_client=np.array(meta["samples_per_client"]),
        total_download_mib=history["total_download_mib"],
        total_upload_mib=history["total_upload_mib"],
        n_params=history["n_params"],
        policy=np.array(args.policy),
        branch=np.array(args.branch),
        branch_dims=np.array(model.branch_dims),
        fusion_dims=np.array(model.fusion_dims),
        local_epochs=np.array(args.local_epochs),
    )
    print(f"Results saved to {file_name}")
    return history["total_upload_mib"]


def main():
    args = args_parser()
    log_buf = io.StringIO()
    stdout_orig = sys.stdout
    sys.stdout = _Tee(stdout_orig, log_buf)
    cum_up_mb = None
    try:
        cum_up_mb = _run(args)
    finally:
        sys.stdout = stdout_orig
        updated_dir = os.path.join(HERE, "result_updated")
        os.makedirs(updated_dir, exist_ok=True)
        txt_path = os.path.join(updated_dir, _txt_stem(args, cum_up_mb) + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(log_buf.getvalue())
        print(f"Console log saved to {txt_path}")


if __name__ == "__main__":
    main()
