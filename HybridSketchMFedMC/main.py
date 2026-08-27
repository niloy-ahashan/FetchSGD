import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch

from dataset import load_uci_har_mm
from federated import run_federated
from models import SketchFusionBNet
from options import args_parser
from utils import Timer


def main():
    args = args_parser()
    print(args)

    timer = Timer()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    if str(device) != args.device:
        print(f"Requested {args.device} but CUDA is unavailable — using {device}")

    print("Loading SketchFusionB UCI HAR split (Acc / Gyro feature vectors)...")
    clients, global_test, meta = load_uci_har_mm(
        args.dataset_dir,
        num_clients=args.num_clients,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=42,
    )
    modalities = list(meta["modalities"])
    mod_dims = list(meta["mod_dims"])
    if mod_dims[0] != args.acc_dim or mod_dims[1] != args.gyro_dim:
        print(
            f"Warning: data dims Acc={mod_dims[0]} Gyro={mod_dims[1]} "
            f"but args Acc={args.acc_dim} Gyro={args.gyro_dim}. Using data dims."
        )
        args.acc_dim, args.gyro_dim = mod_dims
    if meta["num_classes"] != args.num_classes:
        args.num_classes = int(meta["num_classes"])

    model_config = {
        "mod_dims": mod_dims,
        "feat_dim": args.feat_dim,
        "num_classes": args.num_classes,
        "dropout": args.mm_dropout,
        "sketch_r": args.sketch_r,
        "sketch_c": args.sketch_c,
    }
    print(f"SketchFusionB config: {model_config}")
    model = SketchFusionBNet(**model_config).to(device)

    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), results_dir)
    os.makedirs(results_dir, exist_ok=True)

    history = run_federated(args, model, clients, global_test, modalities, device)
    _ = timer()

    client_select_freq = history["client_selected"].mean(axis=0)
    modality_select_freq = history["modality_selected"].mean(axis=(0, 1))
    selected_mask = history["client_selected"][:, :, None]
    denom = np.maximum(selected_mask.sum(axis=(0, 1)), 1)
    modality_select_freq_given_client = (
        (history["modality_selected"] * selected_mask).sum(axis=(0, 1)) / denom
    )

    print("\n=== Selection frequency ===")
    print(f"Client select freq (fraction of rounds): {np.round(client_select_freq, 4).tolist()}")
    print(
        "Modality select freq over all (round, client): "
        + ", ".join(f"{m}={f:.4f}" for m, f in zip(modalities, modality_select_freq))
    )
    print(
        "Modality select freq given client was selected: "
        + ", ".join(
            f"{m}={f:.4f}" for m, f in zip(modalities, modality_select_freq_given_client)
        )
    )
    print(
        f"Final test accuracy: {float(history['test_acc'][-1]):.4f} "
        f"({100.0 * float(history['test_acc'][-1]):.2f}%)"
    )

    mw_str = "_".join(f"{w:.1f}" for w in args.modality_weights)
    file_name = os.path.join(
        results_dir,
        f"Hybrid_UCI_HAR_Top_{args.num_select_modalities}_"
        f"ShapCommRec_{mw_str}_Client_{args.client_select}_"
        f"{args.client_select_ratio:.1f}.npz",
    )
    np.savez(
        file_name,
        train_loss=history["train_loss"],
        train_acc=history["train_acc"],
        test_loss=history["test_loss"],
        test_acc=history["test_acc"],
        upload_bytes=history["upload_bytes"],
        download_bytes=history["download_bytes"],
        client_selected=history["client_selected"],
        modality_selected=history["modality_selected"],
        client_select_freq=client_select_freq,
        modality_select_freq=modality_select_freq,
        modality_select_freq_given_client=modality_select_freq_given_client,
        lr=history["lr"],
        modalities=np.array(modalities),
        reused_sketchfusion_cache=np.array(meta["reused_sketchfusion_cache"]),
        dataset_dir=np.array(meta["dataset_dir"]),
        samples_per_client=np.array(meta["samples_per_client"]),
        total_download_mib=history["total_download_mib"],
        total_upload_mib=history["total_upload_mib"],
    )
    print(f"Results saved to {file_name}")


if __name__ == "__main__":
    main()
