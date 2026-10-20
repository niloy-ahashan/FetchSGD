"""FL-FD (Qi et al., Information Fusion 2023) on UCI HAR, next to FLASH / Hybrid v2 / MFedMC v2 / PMFH v2.

The FL-FD algorithm is imported unchanged from FL-FD/fixed/ (FLFDNet, local_train, fedavg, evaluate,
and the GAF resize). Only the input side is new:
  * data: datasets/uci_har_lstm_mm, read with HybridSketchMFedMC_v2's loader, so the 10-client
    Dirichlet(0.1) split, labels and official test set are the ones the other baselines use
    (uci_har_mm holds the same rows as feature vectors; this cache holds them as raw 128 x 3 windows)
  * per modality (body_acc, body_gyro): magnitude sqrt(x^2+y^2+z^2) over the 128 steps -> GAF -> 70 x 70,
    as FL-FD/fixed/prepare_upfall.py does per wearable sensor; no camera, so the fused image has one
    channel per modality and FLFDNet's first conv takes M input channels instead of 3

Communication (paper Alg. 1 / Fig. 3; released FL-FD/main.py l.134, l.154-155): each round the K selected
clients download the global model and upload a full model (the release uploads a state_dict-sized diff).
Counted as 4 bytes x every floating-point state_dict entry (weights + BatchNorm running stats, as FLASH's
_state_bytes); round-1 download is the shared init and is not billed (FLASH / PMFH v2 convention).
Reported in MB = 1e6 bytes, as the baselines' tables.
"""

import argparse
import importlib.util
import io
import json
import os
import random
import sys
import time

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIXED = os.path.join(ROOT, "FL-FD", "fixed")
sys.path.append(os.path.join(ROOT, "HybridSketchMFedMC_v2"))

import numpy as np
import torch
from pyts.image import GramianAngularField
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import TensorDataset

from dataset import load_uci_har_mm  # HybridSketchMFedMC_v2
from federated import evaluate_model_ic_shards  # HybridSketchMFedMC_v2
from models import FeaExtractor, FeaRefiner  # HybridSketchMFedMC_v2 (as FLASH's --branch shared)
from utils import TableLogger, Timer  # HybridSketchMFedMC_v2


def _load_fixed(name, filename):
    """Import a FL-FD/fixed/ module by path, under a name that can't clash with HybridSketchMFedMC_v2."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(FIXED, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flfd = _load_fixed("flfd_fixed_main", "main.py")
flfd_prep = _load_fixed("flfd_fixed_prepare_upfall", "prepare_upfall.py")

UCI_HAR_ACTIVITIES = ["Walking", "Walking upstairs", "Walking downstairs", "Sitting", "Standing", "Laying"]


def args_parser():
    p = argparse.ArgumentParser(description="FL-FD (Inf. Fusion'23) on UCI HAR, algorithm from FL-FD/fixed/")
    p.add_argument("--dataset_dir", type=str, required=True, help="raw-window cache, e.g. datasets/uci_har_lstm_mm")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--dirichlet_alpha", type=float, default=0.1)
    # FL-FD recipe (paper Sec. 4.3; FL-FD/fixed/main.py)
    p.add_argument("--rounds", type=int, default=200, help="paper: 200 for activity recognition")
    p.add_argument("--clients_per_round", type=int, default=8, help="paper: 12 of 15 clients (80%%)")
    p.add_argument("--local_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=1e-4, help="released code: 0.0001")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--shuffle", type=int, default=1)
    p.add_argument("--gaf_method", choices=["difference", "summation"], default="difference",
                   help="released code: difference; paper's formula: summation")
    p.add_argument("--size", type=int, default=70, help="image side (paper: 70)")
    p.add_argument("--encoder", choices=["flfd", "shared"], default="flfd",
                   help="flfd = GAF images -> FLFDNet (FL-FD); shared = Hybrid v2 encoder per modality on "
                        "feature vectors -> concat -> FL-FD's one Linear head (like FLASH --branch shared)")
    p.add_argument("--feat_dim", type=int, default=64,
                   help="--encoder shared: encoder width (FLASH shared runs: 64; Hybrid v2: 512)")
    p.add_argument("--mm_dropout", type=float, default=0.3, help="--encoder shared: encoder dropout (Hybrid v2: 0.3)")
    p.add_argument("--valid_batch_size", type=int, default=8, help="test shard size (as the baselines)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--results_dir", type=str, default=os.path.join(HERE, "results"))
    return p.parse_args()


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


class _TupleOut(nn.Module):
    """evaluate_model_ic_shards expects ``logits, _ = model(*xs)``."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x), None


class FLFDSharedNet(nn.Module):
    """--encoder shared: Hybrid v2 encoder per modality -> concat -> FL-FD's one-Linear head.

    Each modality: f = FeaExtractor(x), gated as f * FeaRefiner(f) (Hybrid v2 / FLASH SharedBranch).
    The encodings are concatenated (the feature-level counterpart of FL-FD stacking its channels) and
    ``fc1`` classifies them, as FLFDNet.fc1 does. Takes the concatenated feature vector as one tensor,
    so FL-FD's local_train / evaluate are used unchanged.
    """

    def __init__(self, mod_dims, num_classes, feat_dim=64, dropout=0.3):
        super().__init__()
        self.mod_dims = [int(d) for d in mod_dims]
        self.extractors = nn.ModuleList([FeaExtractor(d, feat_dim, dropout) for d in self.mod_dims])
        self.refiners = nn.ModuleList([FeaRefiner(feat_dim) for _ in self.mod_dims])
        for m in self.modules():  # Hybrid v2's init for the encoder; fc1 keeps PyTorch's default, as FL-FD's
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.fc1 = nn.Linear(feat_dim * len(self.mod_dims), num_classes)

    def forward(self, x):
        feats = []
        for xm, ext, ref in zip(torch.split(x, self.mod_dims, dim=1), self.extractors, self.refiners):
            f = ext(xm)
            feats.append(f * ref(f))
        return self.fc1(torch.cat(feats, dim=1))


def to_flfd_images(mods, method, size, chunk=1024):
    """[(N, T, 3) or (N, D) per modality] -> (N, M, size, size): [magnitude ->] GAF -> bilinear resize.

    Raw windows get FL-FD's magnitude step; feature vectors (the baselines' uci_har_mm) skip it.
    """
    gaf = GramianAngularField(method=method)
    n = len(mods[0])
    out = np.zeros((n, len(mods), size, size), dtype=np.float32)
    for m, x in enumerate(mods):
        if x.ndim == 3 and x.shape[2] == 3:      # raw window (uci_har_lstm_mm): FL-FD's magnitude step
            mag = np.sqrt((x.astype(np.float64) ** 2).sum(axis=2))  # (N, T)
        elif x.ndim == 2:                         # feature vector (uci_har_mm): no axes to combine
            mag = x.astype(np.float64)            # (N, D)
        else:
            raise ValueError(f"modality {m}: expected (N, T, 3) or (N, D), got {x.shape}")
        for s in range(0, n, chunk):
            imgs = gaf.fit_transform(mag[s:s + chunk])
            for i, img in enumerate(imgs):
                out[s + i, m] = flfd_prep._resize(img, size)
    if not np.isfinite(out).all():
        raise ValueError("non-finite GAF values (a constant-magnitude window?)")
    return out


def _model_bytes(model):
    state = model.state_dict()
    return 4 * sum(v.numel() for v in state.values() if v.is_floating_point())


def _check_split(meta):
    stats_path = os.path.join(meta["dataset_dir"], "stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path, encoding="utf-8") as f:
            expected = json.load(f).get("images_per_client")
        if expected is not None and list(expected) != list(meta["samples_per_client"]):
            raise RuntimeError(f"client sizes {meta['samples_per_client']} != stats.json {expected}")


def _run(args):
    timer = Timer()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print(f"Requested {args.device} but CUDA is unavailable — using {device}")

    print(f"Loading SketchFusionB UCI HAR split from {os.path.basename(os.path.normpath(args.dataset_dir))}...")
    clients_raw, test_raw, meta = load_uci_har_mm(
        args.dataset_dir, num_clients=args.num_clients, dirichlet_alpha=args.dirichlet_alpha, seed=args.seed,
    )
    _check_split(meta)
    modalities = meta["modalities"]
    num_classes = meta["num_classes"]

    shapes = [tuple(x.shape[1:]) for x in test_raw["xs"]]
    if args.encoder == "shared":
        if any(len(s) != 1 for s in shapes):
            raise ValueError(f"--encoder shared needs feature vectors (e.g. datasets/uci_har_mm), got {shapes}")
        print(f"Building shared-encoder inputs: feature vectors {shapes} concatenated per sample "
              f"(--gaf_method / --size unused)...")
        build = lambda xs: np.concatenate(xs, axis=1).astype(np.float32)
    else:
        raw = test_raw["xs"][0].ndim == 3
        print(f"Building FL-FD inputs ({'raw windows' if raw else 'feature vectors, no magnitude step'}): "
              f"{'|x,y,z| -> ' if raw else ''}GAF ({args.gaf_method}) -> {args.size} x {args.size}, "
              f"one channel per modality {modalities}, input shapes {shapes}...")
        build = lambda xs: to_flfd_images(xs, args.gaf_method, args.size)
    clients = []
    for c in clients_raw:
        x = build(c["xs"])
        clients.append(TensorDataset(torch.from_numpy(x), torch.from_numpy(c["y"])))
    x_test = build(test_raw["xs"])
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(test_raw["y"]))
    test_shards = {"xs": [x_test], "y": test_raw["y"]}
    print(f"  input per sample: {tuple(x_test.shape[1:])}; train={sum(len(c) for c in clients)}, "
          f"test={len(test_ds)}")

    n_clients = len(clients)
    k = min(args.clients_per_round, n_clients)
    if k != args.clients_per_round:
        print(f"WARNING: only {n_clients} clients — selecting {k} per round")
        args.clients_per_round = k

    if args.encoder == "shared":
        model = FLFDSharedNet(meta["mod_dims"], num_classes, args.feat_dim, args.mm_dropout)
        model_desc = (f"FLFDSharedNet, Hybrid v2 encoder per modality {meta['mod_dims']} -> {args.feat_dim}, "
                      f"concat -> fc1 Linear({args.feat_dim * len(modalities)}, {num_classes})")
    else:
        model = flfd.FLFDNet(num_classes, args.size)
        if len(modalities) != 3:  # no camera: one input channel per modality, rest of FLFDNet unchanged
            old = model.features[0]
            model.features[0] = nn.Conv2d(len(modalities), old.out_channels, old.kernel_size, old.stride)
        model_desc = f"FLFDNet, first conv {len(modalities)} -> 3 channels"
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    model_bytes = _model_bytes(model)

    print(f"Clients: {n_clients}, selected per round: {k}")
    print(f"  samples/client: {meta['samples_per_client']}")
    print(f"Model: {model_desc}; params {n_params}; "
          f"bytes per model transfer {model_bytes} (float state_dict, BatchNorm stats included if any)")
    print("Finished initializing in {:.2f} seconds".format(timer()))
    print("Hyperparameters (resolved):")
    for key, v in sorted(vars(args).items()):
        print(f"  {key}: {v}")

    history = {key: [] for key in ("test_acc", "test_loss", "upload_bytes", "download_bytes", "time_s")}
    selected_mask = np.zeros((args.rounds, n_clients), dtype=np.int8)
    best_acc, best_state, best_round = -1.0, None, -1
    total_up = total_down = 0.0
    logger = TableLogger()
    train_start = time.perf_counter()
    prev_elapsed = 0.0

    for r in range(args.rounds):
        selected = random.sample(range(n_clients), k)
        selected_mask[r, selected] = 1
        down_bytes = 0 if r == 0 else k * model_bytes  # round 1: shared init, not billed
        states = [flfd.local_train(model, clients[c], args, device) for c in selected]
        model.load_state_dict(flfd.fedavg(states))
        up_bytes = k * model_bytes

        test_loss, test_acc = evaluate_model_ic_shards(
            _TupleOut(model), test_shards, device, shard_size=args.valid_batch_size
        )
        if test_acc > best_acc:
            best_acc, best_state, best_round = test_acc, {n: t.clone() for n, t in model.state_dict().items()}, r + 1

        elapsed = time.perf_counter() - train_start
        round_s, prev_elapsed = elapsed - prev_elapsed, elapsed
        up_mb, down_mb = up_bytes / 1e6, down_bytes / 1e6
        total_up += up_mb
        total_down += down_mb
        logger.append({
            "iter": r + 1,
            "test_acc": test_acc,
            "up_MB": up_mb,
            "cum_up_MB": total_up,
            "down_MB": down_mb,
            "cum_down_MB": total_down,
            "round_s": round_s,
            "time_s": elapsed,
        })
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)
        history["upload_bytes"].append(up_bytes)
        history["download_bytes"].append(down_bytes)
        history["time_s"].append(elapsed)

    print("Total Download (MB): {:0.2f}".format(total_down))
    print("Total Upload (MB): {:0.2f}".format(total_up))
    print("Avg Download Per Client: {:0.4f}".format(total_down / max(n_clients, 1)))
    print("Avg Upload Per Client: {:0.4f}".format(total_up / max(n_clients, 1)))
    print("Cumulative train comm (download+upload): {:.4f} MB".format(total_down + total_up))
    print(f"Final test accuracy: {float(history['test_acc'][-1]):.4f} "
          f"({100.0 * float(history['test_acc'][-1]):.2f}%)")

    # FL-FD's own report, on the full test set (not shard-averaged)
    final_acc, _, final_pred, ys = flfd.evaluate(model, test_ds, device)
    final_state = {n: t.clone() for n, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    _, _, best_pred, _ = flfd.evaluate(model, test_ds, device)
    model.load_state_dict(final_state)
    labels = list(range(num_classes))
    # Printing of P/R/F1, the classification report and the confusion matrix is switched off: only the
    # shard test_acc is compared. They are still computed for the npz (final_acc_full, confusion_*).
    # names = UCI_HAR_ACTIVITIES if num_classes == len(UCI_HAR_ACTIVITIES) else [str(i) for i in labels]
    # for tag, pred in ((f"final, round {args.rounds}", final_pred), (f"best-round, round {best_round}", best_pred)):
    #     p, rc, f1, _ = precision_recall_fscore_support(ys, pred, labels=labels, average="weighted", zero_division=0)
    #     print(f"\n[{tag} global model] acc {100 * (pred == ys).mean():.3f}  precision {100 * p:.3f}  "
    #           f"recall {100 * rc:.3f}  F1 {100 * f1:.3f}  (weighted, full test set)")
    # print("\nFinal model:\n" + classification_report(ys, final_pred, labels=labels, target_names=names,
    #                                                   digits=4, zero_division=0))
    # print("Confusion matrix (final, rows = true):\n" + str(confusion_matrix(ys, final_pred, labels=labels)))

    os.makedirs(args.results_dir, exist_ok=True)
    file_name = os.path.join(
        args.results_dir,
        f"FLFD_{os.path.basename(os.path.normpath(args.dataset_dir))}_K{k}_E{args.local_epochs}_R{args.rounds}"
        f"{_variant_tag(args, '_')}_seed{args.seed}.npz",
    )
    np.savez(
        file_name,
        test_loss=np.array(history["test_loss"]),
        test_acc=np.array(history["test_acc"]),
        upload_bytes=np.array(history["upload_bytes"]),
        download_bytes=np.array(history["download_bytes"]),
        time_s=np.array(history["time_s"]),  # cumulative training time per round (log's time_s)
        selected=selected_mask,
        modalities=np.array(modalities),
        dataset_dir=np.array(meta["dataset_dir"]),
        samples_per_client=np.array(meta["samples_per_client"]),
        total_download_mib=total_down,  # MB (1e6), key name as FLASH's npz
        total_upload_mib=total_up,
        n_params=n_params,
        model_bytes=model_bytes,
        clients_per_round=k,
        local_epochs=np.array(args.local_epochs),
        gaf_method=np.array(args.gaf_method),
        encoder=np.array(args.encoder),
        feat_dim=np.array(args.feat_dim if args.encoder == "shared" else 0),
        final_acc_full=final_acc,
        best_round=best_round,
        confusion_final=confusion_matrix(ys, final_pred, labels=labels),
        confusion_best=confusion_matrix(ys, best_pred, labels=labels),
    )
    print(f"Results saved to {file_name}")
    return total_up


def _variant_tag(args, before="", after=""):
    """'' for GAF FL-FD with the released difference field; else 'summation' or 'shared_fd<feat_dim>'."""
    if args.encoder == "shared":
        tag = f"shared_fd{args.feat_dim}"
    elif args.gaf_method == "summation":
        tag = "summation"
    else:
        return ""
    return before + tag + after


def _txt_stem(args, cum_up_mb):
    dataset = os.path.basename(os.path.normpath(args.dataset_dir))
    up_str = f"{cum_up_mb:.2f}MB" if cum_up_mb is not None else "NAMB"
    return (
        f"FLFD_{dataset}_mm_{up_str}_dirichlet{args.dirichlet_alpha:g}_it{args.rounds}_"
        f"K{args.clients_per_round}_E{args.local_epochs}_lr{args.lr:g}_"
        f"{_variant_tag(args, '', '_')}"
        f"clients{args.num_clients}_seed{args.seed}"
    )


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
