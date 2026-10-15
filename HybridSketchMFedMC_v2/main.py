import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch

from dataset import load_mhealth_mm, load_uci_har_mm
from federated import run_federated
from models import SketchFusionBNet
from options import args_parser
from utils import Timer


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


def _txt_stem(args, cum_up_mb):
    dataset = os.path.basename(os.path.normpath(args.dataset_dir or "unknown"))
    up_str = f"{cum_up_mb:.2f}MB" if cum_up_mb is not None else "NAMB"
    alpha = getattr(args, "dirichlet_alpha", 0.1)
    epochs = float(args.num_epochs)
    it = int(epochs) if epochs == int(epochs) else epochs
    n_workers = max(1, int(round(args.client_select_ratio * args.num_clients)))
    n_workers = min(n_workers, args.num_clients)
    loss_str = "highloss" if args.prefer_higher_loss else "lowloss"
    return (
        f"HybridV2_{dataset}_mm_{up_str}_dirichlet{alpha:g}_it{it}_"
        f"r{args.num_rows}xc{args.num_cols}_k{args.k}_"
        f"lr{args.lr_scale:g}_"
        f"Top_{args.num_select_modalities}_"
        f"Client_{args.client_select_ratio:.1f}_"
        f"{loss_str}_"
        f"clients{args.num_clients}_workers{n_workers}_seed{args.seed}"
    )


def _result_updated_dir():
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(here, "result_updated")


def _run(args):
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

    if args.dataset == "mhealth":
        print("Loading SketchFusionB4 mHealth split (Acc/Gyro/Mag/ECG feature vectors)...")
        clients, global_test, meta = load_mhealth_mm(
            args.dataset_dir,
            num_clients=args.num_clients,
            dirichlet_alpha=args.dirichlet_alpha,
            seed=42,
        )
    else:
        print("Loading SketchFusionB UCI HAR split (Acc / Gyro feature vectors)...")
        clients, global_test, meta = load_uci_har_mm(
            args.dataset_dir,
            num_clients=args.num_clients,
            dirichlet_alpha=args.dirichlet_alpha,
            seed=42,
        )
    modalities = list(meta["modalities"])
    mod_dims = list(meta["mod_dims"])
    mismatches = []
    for name, actual_dim in zip(modalities, mod_dims):
        attr = f"{name.lower()}_dim"
        arg_dim = getattr(args, attr, None)
        if arg_dim != actual_dim:
            mismatches.append(f"{name}={actual_dim} (arg {attr}={arg_dim})")
            setattr(args, attr, actual_dim)
    if mismatches:
        print("Warning: data mod_dims differ from args: " + ", ".join(mismatches)
              + ". Using data dims.")
    if meta["num_classes"] != args.num_classes:
        args.num_classes = int(meta["num_classes"])

    model_config = {
        "mod_dims": mod_dims,
        "feat_dim": args.feat_dim,
        "num_classes": args.num_classes,
        "dropout": args.mm_dropout,
        "sketch_r": args.sketch_r,
        "sketch_c": args.sketch_c,
        "fusion_mode": args.fusion_mode,
    }
    print(f"Hybrid v2 fusion config: {model_config}")
    model = SketchFusionBNet(**model_config).to(device)

    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), results_dir)
    os.makedirs(results_dir, exist_ok=True)

    print("Finished initializing in {:.2f} seconds".format(timer()))
    print("Hyperparameters (resolved):")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")

    history = run_federated(args, model, clients, global_test, modalities, device)

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
    dataset_label = "mHealth" if args.dataset == "mhealth" else "UCI_HAR"
    upload_object = getattr(args, "effective_upload_object", args.upload_object)
    file_name = os.path.join(
        results_dir,
        f"HybridV2_{dataset_label}_{args.fusion_mode}_{upload_object}_"
        f"Top_{args.num_select_modalities}_"
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
        fusion_mode=np.array(args.fusion_mode),
        upload_object=np.array(upload_object),
        local_epochs=np.array(args.local_epochs),
        weight_decay=np.array(args.weight_decay),
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
        updated_dir = _result_updated_dir()
        os.makedirs(updated_dir, exist_ok=True)
        txt_path = os.path.join(updated_dir, _txt_stem(args, cum_up_mb) + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(log_buf.getvalue())
        print(f"Console log saved to {txt_path}")


if __name__ == "__main__":
    main()
