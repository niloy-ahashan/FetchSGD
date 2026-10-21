"""PMFH + federated head on mHealth (4 modalities: Acc, Gyro, Mag, ECG).

mHealth counterpart of ``main_uci_har_v2.py --variant fedhead`` (option B):
official PMFH training plus a classifier head on ``nec_vec.detach()``, trained
with CE, uploaded and FedAvg'd with the global model; ``test_acc`` = head
argmax, 8-sample shard mean, like the baselines. PMFH takes two inputs, so
``--mod_mode`` picks how the four modalities are fed in:

* ``grouped`` (prefix ``PMFHfhG``): IMU = Acc+Gyro+Mag concatenated -> PMFH
  "image", ECG -> "text". Models are upstream ``Layers`` / ``LayersFedHead``,
  i.e. PMFH unchanged. Label results "PMFH + federated head (IMU | ECG)".
* ``nmod`` (prefix ``PMFHfhN``): one PMFH encoder + gate per modality, hash of
  the sum of the four gated features (``layers_nmod.LayersNMod``; equal to
  ``Layers`` for two modalities). Label results "PMFH-N + federated head".

Data is the shared mHealth split (``HybridSketchMFedMC_v2/dataset.py``
``load_mhealth_mm``: 10 clients, Dirichlet(0.1), official test set). Local
training is ``main_uci_har_v2.LocalUpdateUCI`` and everything else (client
exclusion, drop-last-batch-of-1, FastConLoss, comm accounting, log table) is the
same as ``main_uci_har_v2.py``, whose helpers are imported.
"""

import argparse
import copy
import io
import os
import sys
import time

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import numpy as np
import torch
from torch.utils.data import DataLoader

from layers import Layers
from layers_nmod import LayersNMod, LayersNModFedHead
from layers_uci import LayersFedHead
from main_uci_har import (
    FastConLoss,
    TableLogger,
    _Tee,
    _load_hybrid_v2_dataset_module,
    _nbytes,
    _one_hot,
    _shard_mean_acc,
)
from main_uci_har_v2 import LocalUpdateUCI, _head_logits
from utils import (
    agg_func,
    average_weights,
    proto_aggregation,
    seed_setting,
)

PREFIX = {"grouped": "PMFHfhG", "nmod": "PMFHfhN"}


def args_parser():
    parser = argparse.ArgumentParser()
    # data / federation
    parser.add_argument("--dataset_dir", type=str,
                        default=os.path.join(ROOT, "datasets", "mhealth_mm_random_group"))
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mod_mode", type=str, default="grouped", choices=["grouped", "nmod"],
                        help="grouped: Acc+Gyro+Mag -> image, ECG -> text (official PMFH); "
                             "nmod: one PMFH encoder + gate per modality")
    # PMFH (same names and defaults as main_uci_har_v2.py)
    parser.add_argument("--mlpdrop", type=float, default=0.01)
    parser.add_argument("--classes", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--train_ep", type=int, default=10)
    parser.add_argument("--train_ep_private", type=int, default=10)
    parser.add_argument("--bit", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--common_dim", type=int, default=128)
    parser.add_argument("--aa", type=float, default=0.000001)
    parser.add_argument("--bb", type=float, default=0.01)
    parser.add_argument("--cc", type=float, default=0.001)
    parser.add_argument("--dd", type=float, default=1)
    parser.add_argument("--ee", type=float, default=0.001)
    parser.add_argument("--ff", type=float, default=1)
    # federated head
    parser.add_argument("--head_grad", type=str, default="detached", choices=["detached", "joint"],
                        help="detached = head cannot change PMFH (option B); "
                             "joint = head's CE also trains PMFH's network (option C)")
    parser.add_argument("--head_lr", type=float, default=1e-3,
                        help="Adam lr of the classifier head (PMFH keeps --learning_rate)")
    parser.add_argument("--clf_weight", type=float, default=1.0,
                        help="weight of the CrossEntropy head loss added to the global model")
    # evaluation
    parser.add_argument("--valid_batch_size", type=int, default=8,
                        help="shard size for the IC-style shard-mean accuracy")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--slow_upstream", action="store_true",
                        help="use upstream's loop ConLoss and autograd anomaly mode (same numbers, much slower)")
    args = parser.parse_args()
    args.variant = "fedhead"  # read by LocalUpdateUCI
    return args


def _prefix(args):
    return PREFIX[args.mod_mode] + ("jh" if args.head_grad == "joint" else "")


def _txt_stem(args, cum_up_mb, n_active):
    dataset = os.path.basename(os.path.normpath(args.dataset_dir or "unknown"))
    up_str = f"{cum_up_mb:.2f}MB" if cum_up_mb is not None else "NAMB"
    return (
        f"{_prefix(args)}_{dataset}_mm_{up_str}_dirichlet{args.dirichlet_alpha:g}_"
        f"it{args.rounds}_bit{args.bit}_h{args.hidden_dim}_c{args.common_dim}_"
        f"clf{args.clf_weight:g}_lr{args.learning_rate:g}_"
        f"E{args.train_ep}_Ep{args.train_ep_private}_"
        f"clients{args.num_clients}_workers{n_active}_seed{args.seed}"
    )


def _split_inputs(args, xs):
    """mHealth modality list -> PMFH's (image, text) arrays for ``args.mod_mode``."""
    if args.mod_mode == "grouped":
        return np.concatenate(xs[:3], axis=1), xs[3]
    return np.concatenate(xs, axis=1), np.empty((len(xs[0]), 0), dtype=np.float32)


def _run(args):
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"Requested {args.device} but CUDA is unavailable — using {device}")
    args.device = device
    seed_setting(args.seed)
    if not args.slow_upstream:
        # Upstream calls torch.autograd.set_detect_anomaly(True) every local update.
        # Anomaly mode only adds NaN checks / stack traces to backward; it changes no values.
        torch.autograd.set_detect_anomaly = lambda *a, **k: None
    timer_start = time.perf_counter()

    print("Loading SketchFusionB4 mHealth split (Acc / Gyro / Mag / ECG feature vectors)...")
    ds = _load_hybrid_v2_dataset_module()
    clients, global_test, meta = ds.load_mhealth_mm(
        args.dataset_dir,
        num_clients=args.num_clients,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=42,
    )
    args.mod_dims = [int(d) for d in meta["mod_dims"]]
    if meta["num_classes"] != args.classes:
        args.classes = int(meta["num_classes"])

    # Clients too small for PMFH's positional per-class prototypes (n // classes == 0).
    n_per = [int(len(c["y"])) for c in clients]
    active = [cid for cid, n in enumerate(n_per) if n >= args.classes]
    excluded = [cid for cid, n in enumerate(n_per) if n < args.classes]
    if excluded:
        print(
            f"Excluding clients {excluded} (samples {[n_per[c] for c in excluded]} < "
            f"classes={args.classes}): PMFH prototypes would be NaN."
        )
    n_active = len(active)
    args.num_users = n_active  # upstream enables the contrastive loss when all users have protos

    # Upstream datasetall layout; client j owns a contiguous index range.
    split = [_split_inputs(args, clients[c]["xs"]) for c in active]
    train_x = np.concatenate([s[0] for s in split]).astype(np.float32)
    train_y = np.concatenate([s[1] for s in split]).astype(np.float32)
    train_L = _one_hot(np.concatenate([clients[c]["y"] for c in active]), args.classes)
    args.image_dim, args.text_dim = int(train_x.shape[1]), int(train_y.shape[1])
    empty_x = np.empty((0, args.image_dim), dtype=np.float32)
    empty_y = np.empty((0, args.text_dim), dtype=np.float32)
    empty_L = np.empty((0, args.classes), dtype=np.int64)
    datasetall = [train_L, train_x, train_y, empty_L, empty_x, empty_y, empty_L, empty_x, empty_y]
    offsets = np.cumsum([0] + [n_per[c] for c in active])
    client_idcs_train = [list(range(offsets[j], offsets[j + 1])) for j in range(n_active)]
    client_idcs_empty = [[] for _ in range(n_active)]

    q_img, q_txt = _split_inputs(args, global_test["xs"])
    q_lab = np.asarray(global_test["y"], dtype=np.int64)

    if args.mod_mode == "grouped":
        args.img_hidden_dim = [args.image_dim, args.hidden_dim, args.common_dim]
        args.txt_hidden_dim = [args.text_dim, args.hidden_dim, args.common_dim]
        # Same construction order as main_uci_har_v2.py fedhead; only the global model gets the head.
        private_modal_list = [Layers(args=args) for _ in range(n_active)]
        global_modal = LayersFedHead(args=args)
    else:
        private_modal_list = [LayersNMod(args=args) for _ in range(n_active)]
        global_modal = LayersNModFedHead(args=args)

    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(HERE, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    model_bytes = _nbytes(global_modal.state_dict())
    n_params = sum(p.numel() for p in global_modal.parameters())
    print("Finished initializing in {:.2f} seconds".format(time.perf_counter() - timer_start))
    print("Hyperparameters (resolved):")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    head_in = "nec_vec (joint)" if args.head_grad == "joint" else "detached nec_vec"
    if args.mod_mode == "grouped":
        print(f"PMFH + federated head (IMU | ECG): official Layers bit={args.bit} "
              f"image=Acc+Gyro+Mag {args.img_hidden_dim} text=ECG {args.txt_hidden_dim}, "
              f"head={args.common_dim}->{args.classes} on {head_in}")
    else:
        print(f"PMFH-N + federated head: LayersNMod bit={args.bit} "
              f"encoders={dict(zip(meta['modalities'], [[d, args.hidden_dim, args.common_dim] for d in args.mod_dims]))}, "
              f"fusion = hash(sum of gated features), head={args.common_dim}->{args.classes} on {head_in}")
    print(f"Total params: {n_params}  (state_dict bytes per model: {model_bytes})")
    print(f"Active clients: {active} (n={n_active}), samples: {[n_per[c] for c in active]}")
    print(f"Engine: PMFH training + federated head (CE x {args.clf_weight:g}, head lr {args.head_lr:g}, "
          f"{'CE gradient also trains PMFH' if args.head_grad == 'joint' else 'gradient stops at nec_vec'}; "
          f"head uploaded and FedAvg'd with the global model)")
    print("Speed: " + ("upstream ConLoss loop + anomaly mode" if args.slow_upstream
                       else "FastConLoss (same values) + anomaly mode off"))
    print(f"Eval: test_acc = trained head argmax on test ({len(q_lab)}); shard={args.valid_batch_size}")

    logger = TableLogger()
    history = {k: [] for k in (
        "test_acc", "private_loss", "global_loss",
        "upload_bytes", "download_bytes",
    )}
    global_avg_protos = {}
    global_protos = {}
    local_protos = {}
    total_upload = 0.0
    total_download = 0.0
    train_start = time.perf_counter()
    prev_elapsed = 0.0

    for rnd in range(args.rounds):
        # Download billed before local training; round 1 is shared init (IC convention).
        if rnd == 0:
            download_bytes = 0
        else:
            download_bytes = n_active * (
                _nbytes(global_modal.state_dict())
                + _nbytes(global_protos)
                + _nbytes(global_avg_protos)
            )

        local_weights = []
        private_weights = []
        upload_bytes = 0
        total_private_loss = 0.0
        total_global_loss = 0.0

        for idx in range(n_active):
            local_model = LocalUpdateUCI(
                args=args, id=idx, dataset=datasetall, idxs=client_idcs_train[idx],
                idcs_retrieval=client_idcs_empty, idcs_query=client_idcs_empty,
            )
            if not args.slow_upstream:
                local_model.criterion_CL = FastConLoss(temperature=0.07)
            # Same loader as upstream, but never leave a final batch of 1 (BatchNorm1d).
            train_ds = local_model.dataloader_train.dataset
            local_model.dataloader_train = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                drop_last=(len(train_ds) % args.batch_size == 1),
            )
            w_global, w_private, protos, private_loss, global_loss = local_model.update_weights_UCCH_new(
                args, global_protos, global_avg_protos,
                model_img=global_modal, private_modal=private_modal_list[idx],
            )
            total_private_loss += private_loss
            total_global_loss += global_loss

            agg_protos = agg_func(protos)
            local_protos[idx] = copy.deepcopy(agg_protos)
            local_weights.append(w_global)
            private_weights.append(w_private)
            upload_bytes += _nbytes(w_global) + _nbytes(agg_protos)

        local_weights_list = average_weights(local_weights)
        global_avg_protos = proto_aggregation(local_protos)
        global_protos = copy.deepcopy(local_protos)
        for idx in range(n_active):
            private_modal_list[idx].load_state_dict(private_weights[idx])
        global_modal.load_state_dict(local_weights_list[0], strict=True)

        # ---- eval: trained head, read out like IC / MFedMC v2 / Hybrid v2 ----
        pred = _head_logits(global_modal, q_img, q_txt, device).argmax(dim=1).cpu().numpy()
        test_acc = _shard_mean_acc((pred == q_lab).astype(np.float64), args.valid_batch_size)
        global_modal.cpu()

        elapsed_s = time.perf_counter() - train_start
        round_s = elapsed_s - prev_elapsed
        prev_elapsed = elapsed_s
        upload_mb = upload_bytes / (1000 * 1000)
        download_mb = download_bytes / (1000 * 1000)
        total_upload += upload_mb
        total_download += download_mb

        logger.append({
            "iter": rnd + 1,
            "test_acc": test_acc,
            "up_MB": upload_mb,
            "cum_up_MB": total_upload,
            "down_MB": download_mb,
            "cum_down_MB": total_download,
            "round_s": round_s,
            "time_s": elapsed_s,
        })
        history["test_acc"].append(test_acc)
        history["private_loss"].append(total_private_loss / n_active)
        history["global_loss"].append(total_global_loss / n_active)
        history["upload_bytes"].append(upload_bytes)
        history["download_bytes"].append(download_bytes)

    print("Total Download (MiB): {:0.2f}".format(total_download))
    print("Total Upload (MiB): {:0.2f}".format(total_upload))
    print("Avg Download Per Client: {:0.2f}".format(total_download / max(args.num_clients, 1)))
    print("Avg Upload Per Client: {:0.2f}".format(total_upload / max(args.num_clients, 1)))
    print("Cumulative train comm (download+upload): {:.4f} MiB".format(total_download + total_upload))
    print(
        f"Final test accuracy: {history['test_acc'][-1]:.4f} "
        f"({100.0 * history['test_acc'][-1]:.2f}%)"
    )

    file_name = os.path.join(results_dir, f"{_prefix(args)}_mHealth_bit{args.bit}_h{args.hidden_dim}.npz")
    np.savez(
        file_name,
        test_acc=np.asarray(history["test_acc"]),
        private_loss=np.asarray(history["private_loss"]),
        global_loss=np.asarray(history["global_loss"]),
        upload_bytes=np.asarray(history["upload_bytes"]),
        download_bytes=np.asarray(history["download_bytes"]),
        total_upload_mib=total_upload,
        total_download_mib=total_download,
        samples_per_client=np.asarray(n_per),
        active_clients=np.asarray(active),
        excluded_clients=np.asarray(excluded, dtype=np.int64),
        dataset_dir=np.array(meta["dataset_dir"]),
        mod_mode=np.array(args.mod_mode),
        mod_dims=np.asarray(args.mod_dims),
        bit=np.array(args.bit),
        hidden_dim=np.array(args.hidden_dim),
        common_dim=np.array(args.common_dim),
        clf_weight=np.array(args.clf_weight),
        head_lr=np.array(args.head_lr),
        head_grad=np.array(args.head_grad),
    )
    print(f"Results saved to {file_name}")
    return total_upload, n_active


def main():
    args = args_parser()
    log_buf = io.StringIO()
    stdout_orig = sys.stdout
    sys.stdout = _Tee(stdout_orig, log_buf)
    cum_up_mb, n_active = None, args.num_clients
    try:
        cum_up_mb, n_active = _run(args)
    finally:
        sys.stdout = stdout_orig
        updated_dir = os.path.join(HERE, "result_updated")
        os.makedirs(updated_dir, exist_ok=True)
        txt_path = os.path.join(updated_dir, _txt_stem(args, cum_up_mb, n_active) + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(log_buf.getvalue())
        print(f"Console log saved to {txt_path}")


if __name__ == "__main__":
    main()
