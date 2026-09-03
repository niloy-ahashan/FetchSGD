import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch

from options import args_parser
from dataset import (
    MODALITIES,
    attach_global_test,
    dirichlet_partition_data,
    load_from_sketchfusion_dir,
    load_subject_clients,
    stratified_split_client_data,
)
from federated import federated_learning
from models import VectorMLP


def _default_dataset_dir():
    here = os.path.abspath(os.path.dirname(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(repo, "datasets", "uci_har_mm")


def main():
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = args.dataset_dir or _default_dataset_dir()
    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), results_dir)
    os.makedirs(results_dir, exist_ok=True)

    print("Loading UCI HAR Acc/Gyro features...")
    if args.partition == "subject":
        client_data, global_test, meta = load_subject_clients(
            dataset_dir, uci_root=args.uci_root or None
        )
        args.num_clients = meta["num_clients"]
        if args.class_non_iid_rate < 1.0:
            print(f"Using Dirichlet partitioning (alpha={args.class_non_iid_rate})...")
            client_data = dirichlet_partition_data(
                client_data, alpha=args.class_non_iid_rate, seed=args.seed
            )
    else:
        client_data, global_test, meta = load_from_sketchfusion_dir(
            dataset_dir,
            num_clients=args.num_clients,
            dirichlet_alpha=args.dirichlet_alpha,
            seed=args.seed,
        )
    if meta["acc_dim"] != args.acc_dim or meta["gyro_dim"] != args.gyro_dim:
        print(
            f"Warning: data dims Acc={meta['acc_dim']} Gyro={meta['gyro_dim']} "
            f"but args Acc={args.acc_dim} Gyro={args.gyro_dim}. Using data dims."
        )
        args.acc_dim = meta["acc_dim"]
        args.gyro_dim = meta["gyro_dim"]

    client_data_train, client_data_local_test = stratified_split_client_data(
        client_data, train_ratio=args.train_ratio, seed=args.seed
    )
    if args.eval_on_global_test:
        print("Evaluating fusion on the official UCI HAR test set.")
        client_data_test = attach_global_test(client_data_train.keys(), global_test)
    else:
        print(
            f"Evaluating fusion on per-client held-out split "
            f"(train_ratio={args.train_ratio}, ActionSense protocol)."
        )
        client_data_test = client_data_local_test

    k = args.num_classes
    global_modality_encoders = [
        VectorMLP(args.acc_dim, k, args.hidden, args.mm_dropout),
        VectorMLP(args.gyro_dim, k, args.hidden, args.mm_dropout),
    ]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for m in global_modality_encoders:
        m.to(device)

    print("Starting MFedMC federated learning on Acc + Gyro...")
    (
        acc,
        mod_counts,
        upload_bytes_round,
        client_selected,
        modality_selected,
        elapsed_seconds_round,
    ) = federated_learning(
        args=args,
        client_data_train=client_data_train,
        client_data_test=client_data_test,
        global_models=global_modality_encoders,
        modalities=MODALITIES,
        device=device,
    )
    upload_bytes_cumulative = np.cumsum(upload_bytes_round)
    mean_fusion_acc = np.nanmean(acc[:, :, -1], axis=1)
    client_select_freq = client_selected.mean(axis=0)
    # Fraction of (round, client) pairs that uploaded each modality.
    modality_select_freq = modality_selected.mean(axis=(0, 1))
    # Among selected clients, how often each modality was uploaded.
    selected_mask = client_selected[:, :, None]
    denom = np.maximum(selected_mask.sum(axis=(0, 1)), 1)
    modality_select_freq_given_client = (
        (modality_selected * selected_mask).sum(axis=(0, 1)) / denom
    )

    print("\n=== Selection frequency (for later compression-ratio plots) ===")
    print(f"Client select freq (fraction of rounds): {np.round(client_select_freq, 4).tolist()}")
    print(
        f"Modality select freq over all (round, client): "
        + ", ".join(f"{m}={f:.4f}" for m, f in zip(MODALITIES, modality_select_freq))
    )
    print(
        f"Modality select freq given client was selected: "
        + ", ".join(f"{m}={f:.4f}" for m, f in zip(MODALITIES, modality_select_freq_given_client))
    )
    print(
        f"Final mean fusion accuracy: {float(mean_fusion_acc[-1]):.2f}% | "
        f"Cumulative uplink: {int(upload_bytes_cumulative[-1]) / 1e6:.6f} MB"
    )

    mw_str = "_".join([f"{w:.1f}" for w in args.modality_weights])
    file_name = os.path.join(
        results_dir,
        f"MFedMC_UCI_HAR_mm_{args.partition}_Top_{args.top_shap}_ShapCommRec_{mw_str}_"
        f"Client_{args.client_select_ratio:.1f}.npz",
    )
    np.savez(
        file_name,
        acc=acc,
        mod=mod_counts,
        upload_bytes_round=upload_bytes_round,
        upload_bytes_cumulative=upload_bytes_cumulative,
        elapsed_seconds_round=elapsed_seconds_round,
        mean_fusion_acc=mean_fusion_acc,
        client_selected=client_selected,
        modality_selected=modality_selected,
        client_select_freq=client_select_freq,
        modality_select_freq=modality_select_freq,
        modality_select_freq_given_client=modality_select_freq_given_client,
        modalities=np.array(MODALITIES),
        client_ids=np.array(list(client_data_train.keys())),
        reused_sketchfusion_cache=np.array(meta.get("reused_sketchfusion_cache", False)),
        partition=np.array(args.partition),
        dataset_dir=np.array(meta["dataset_dir"]),
        samples_per_client=np.array(meta["samples_per_client"]),
    )
    print(f"Results saved to {file_name}")


if __name__ == "__main__":
    main()
