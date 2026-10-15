"""PMFH on the SketchFusionB UCI HAR split, logged like IC / MFedMC v2 / Hybrid v2.

Training is upstream PMFH unchanged: ``LocalUpdate`` from ``main.py`` plus
``agg_func`` / ``proto_aggregation`` / ``average_weights`` from ``utils.py``,
driven by the same round loop as ``main.py``. What this file adds:

* data: the 10-client Dirichlet(0.1) cache in ``datasets/uci_har_mm`` (same
  clients as the baselines), Acc -> PMFH "image" input, Gyro -> "text" input;
* eval: kNN (Hamming) accuracy on the official UCI HAR test set, plus mAP;
  also ``test_acc_probe`` -- a linear probe (Linear(nbit -> classes), argmax,
  shard mean) fitted each round on the frozen global model's continuous
  training-set codes. It reads accuracy out the way IC / MFedMC v2 / Hybrid v2
  do (a trained linear head), without touching PMFH training: upstream's own
  ``classify`` head is never trained (``clf_loss`` is computed, never added);
* comm accounting (bytes / 1e6, same unit as the baselines);
* console log in ``result_updated/`` with the cumulative upload MB in the name.

Two UCI HAR edge cases that would break upstream code are handled here:
clients with fewer than ``classes`` samples are excluded (their positional
prototypes would be empty means -> NaN for everyone), and a client whose last
batch would be a single sample drops it (BatchNorm1d fails on batch size 1).
"""

import argparse
import importlib.util
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
from loss import ConLoss
from main import LocalUpdate
from utils import (
    CustomDataSet,
    agg_func,
    average_weights,
    calculate_map,
    proto_aggregation,
    seed_setting,
)
import copy


class FastConLoss(ConLoss):
    """Upstream ``ConLoss`` with the anchor matrix built by one gather.

    Upstream fills ``anchor_feature`` row by row in a Python loop
    (batch_size * anchor_count = 64 * 128 iterations, each a GPU sync).
    Row i gets ``global_protos[argmax(labels[i % batch_size])]``; here the same
    rows are gathered at once. Values, loss and gradients are identical; every
    other line is upstream ``loss.py`` verbatim.
    """

    def forward(self, features, labels=None, global_protos=None):
        device = torch.device('cuda' if features.is_cuda else 'cpu')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is None or global_protos is None:
            raise ValueError('Both `labels` and `global_protos` must be provided')

        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_count = contrast_count
            # was: zeros_like + per-row loop over global_protos[class_index]
            dim = contrast_feature.shape[1]
            class_index = labels.argmax(dim=1)
            table = torch.zeros(int(class_index.max().item()) + 1, dim,
                                dtype=contrast_feature.dtype, device=contrast_feature.device)
            for c in class_index.unique().tolist():
                table[c] = global_protos[c]
            anchor_feature = table[class_index.repeat(anchor_count)]
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create mask based on one-hot labels
        mask = torch.matmul(labels, labels.T).float().to(device)
        mask = mask.repeat(anchor_count, contrast_count)

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def _load_hybrid_v2_dataset_module():
    """Import HybridSketchMFedMC_v2/dataset.py by path (its name would clash otherwise)."""
    path = os.path.join(ROOT, "HybridSketchMFedMC_v2", "dataset.py")
    spec = importlib.util.spec_from_file_location("hybrid_v2_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


class TableLogger:
    """Same console table as CommEfficient ``mm_train.py`` / Hybrid v2."""

    def append(self, output):
        if not hasattr(self, "keys"):
            self.keys = output.keys()
            print(*("{:>12s}".format(k) for k in self.keys))
        filtered = [output[k] for k in self.keys]
        print(
            *(
                "{:12.4f}".format(v)
                if isinstance(v, (float, np.float32, np.float64))
                else "{:12}".format(v)
                for v in filtered
            ),
            flush=True,
        )


def args_parser():
    parser = argparse.ArgumentParser()
    # data / federation
    parser.add_argument("--dataset_dir", type=str,
                        default=os.path.join(ROOT, "datasets", "uci_har_mm"))
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    # PMFH (names and defaults as upstream main.py; flickr.sh rounds/batch)
    parser.add_argument("--mlpdrop", type=float, default=0.01)
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--train_ep", type=int, default=10)
    parser.add_argument("--train_ep_private", type=int, default=10)
    parser.add_argument("--bit", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_dim", type=int, default=348, help="Acc feature dim")
    parser.add_argument("--text_dim", type=int, default=213, help="Gyro feature dim")
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--common_dim", type=int, default=128)
    parser.add_argument("--aa", type=float, default=0.000001)
    parser.add_argument("--bb", type=float, default=0.01)
    parser.add_argument("--cc", type=float, default=0.001)
    parser.add_argument("--dd", type=float, default=1)
    parser.add_argument("--ee", type=float, default=0.001)
    parser.add_argument("--ff", type=float, default=1)
    # evaluation
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--probe_l2", type=float, default=1e-4,
                        help="L2 penalty of the linear probe (fixed, not tuned on test)")
    parser.add_argument("--probe_iters", type=int, default=500,
                        help="max L-BFGS iterations for the linear probe")
    parser.add_argument("--valid_batch_size", type=int, default=8,
                        help="shard size for the IC-style shard-mean accuracy")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--slow_upstream", action="store_true",
                        help="use upstream's loop ConLoss and autograd anomaly mode (same numbers, much slower)")
    return parser.parse_args()


def _txt_stem(args, cum_up_mb, n_active):
    dataset = os.path.basename(os.path.normpath(args.dataset_dir or "unknown"))
    up_str = f"{cum_up_mb:.2f}MB" if cum_up_mb is not None else "NAMB"
    return (
        f"PMFH_{dataset}_mm_{up_str}_dirichlet{args.dirichlet_alpha:g}_"
        f"it{args.rounds}_bit{args.bit}_lr{args.learning_rate:g}_"
        f"E{args.train_ep}_Ep{args.train_ep_private}_"
        f"clients{args.num_clients}_workers{n_active}_seed{args.seed}"
    )


def _nbytes(obj):
    """Bytes of every tensor in a (nested) state_dict / prototype container."""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(v) for v in obj)
    return 0


def _one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes), dtype=np.int64)
    out[np.arange(len(y)), y] = 1
    return out


@torch.no_grad()
def _hash_codes(model, xs_img, xs_txt, device, batch_size=1024):
    model.eval()
    model.to(device)
    codes = []
    for start in range(0, len(xs_img), batch_size):
        img = torch.as_tensor(xs_img[start:start + batch_size], dtype=torch.float32, device=device)
        txt = torch.as_tensor(xs_txt[start:start + batch_size], dtype=torch.float32, device=device)
        _, code, _ = model(img, txt)
        codes.append(torch.sign(code))
    return torch.cat(codes, dim=0)


@torch.no_grad()
def _raw_codes(model, xs_img, xs_txt, device, batch_size=1024):
    """Continuous code H (tanh output, nbit-d): the input of upstream's classify head."""
    model.eval()
    model.to(device)
    codes = []
    for start in range(0, len(xs_img), batch_size):
        img = torch.as_tensor(xs_img[start:start + batch_size], dtype=torch.float32, device=device)
        txt = torch.as_tensor(xs_txt[start:start + batch_size], dtype=torch.float32, device=device)
        _, code, _ = model(img, txt)
        codes.append(code)
    return torch.cat(codes, dim=0)


def _linear_probe_predict(train_x, train_y, test_x, num_classes, l2, max_iter):
    """Multinomial logistic regression on frozen codes; returns test predictions.

    Zero init + full-batch L-BFGS on a convex objective: deterministic and
    draws nothing from torch's RNG, so PMFH's shuffling/dropout stay as they
    would be without the probe.
    """
    train_x = train_x.detach().float()
    test_x = test_x.detach().float()
    train_y = torch.as_tensor(train_y, dtype=torch.long, device=train_x.device)
    W = torch.zeros(train_x.shape[1], num_classes, device=train_x.device, requires_grad=True)
    b = torch.zeros(num_classes, device=train_x.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(train_x @ W + b, train_y) + l2 * (W * W).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        return (test_x @ W + b).argmax(dim=1)


@torch.no_grad()
def _knn_predict(q_codes, db_codes, db_labels, k, num_classes):
    """Majority vote of the k nearest db codes (Hamming); ties -> nearest class."""
    nbit = q_codes.shape[1]
    hamm = 0.5 * (nbit - q_codes @ db_codes.t())
    order = torch.sort(hamm, dim=1, stable=True).indices[:, :k]
    nn_labels = db_labels[order]                                   # [Q, k]
    onehot = torch.nn.functional.one_hot(nn_labels, num_classes)   # [Q, k, C]
    counts = onehot.sum(dim=1)                                     # [Q, C]
    first_pos = onehot.float().argmax(dim=1)                       # first occurrence
    score = counts * (2 * k) - first_pos
    score = torch.where(counts > 0, score, torch.full_like(score, -10 ** 9))
    return score.argmax(dim=1)


def _shard_mean_acc(correct, shard_size):
    """IC val metric: unweighted mean of per-shard accuracies (see Hybrid v2)."""
    shard_size = max(1, int(shard_size))
    accs = [correct[s:s + shard_size].mean() for s in range(0, len(correct), shard_size)]
    return float(np.mean(accs))


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

    print("Loading SketchFusionB UCI HAR split (Acc / Gyro feature vectors)...")
    ds = _load_hybrid_v2_dataset_module()
    clients, global_test, meta = ds.load_uci_har_mm(
        args.dataset_dir,
        num_clients=args.num_clients,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=42,
    )
    acc_dim, gyro_dim = meta["mod_dims"]
    if (acc_dim, gyro_dim) != (args.image_dim, args.text_dim):
        print(f"Warning: data dims Acc={acc_dim}, Gyro={gyro_dim} differ from args. Using data dims.")
        args.image_dim, args.text_dim = acc_dim, gyro_dim
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
    train_x = np.concatenate([clients[c]["xs"][0] for c in active]).astype(np.float32)
    train_y = np.concatenate([clients[c]["xs"][1] for c in active]).astype(np.float32)
    train_L = _one_hot(np.concatenate([clients[c]["y"] for c in active]), args.classes)
    empty_x = np.empty((0, args.image_dim), dtype=np.float32)
    empty_y = np.empty((0, args.text_dim), dtype=np.float32)
    empty_L = np.empty((0, args.classes), dtype=np.int64)
    datasetall = [train_L, train_x, train_y, empty_L, empty_x, empty_y, empty_L, empty_x, empty_y]
    offsets = np.cumsum([0] + [n_per[c] for c in active])
    client_idcs_train = [list(range(offsets[j], offsets[j + 1])) for j in range(n_active)]
    client_idcs_empty = [[] for _ in range(n_active)]

    # Retrieval database for eval = all training samples (every client).
    db_img = np.concatenate([c["xs"][0] for c in clients]).astype(np.float32)
    db_txt = np.concatenate([c["xs"][1] for c in clients]).astype(np.float32)
    db_lab = np.concatenate([c["y"] for c in clients]).astype(np.int64)
    q_img, q_txt = global_test["xs"][0], global_test["xs"][1]
    q_lab = np.asarray(global_test["y"], dtype=np.int64)
    db_lab_t = torch.as_tensor(db_lab, device=device)

    args.img_hidden_dim = [args.image_dim, args.hidden_dim, args.common_dim]
    args.txt_hidden_dim = [args.text_dim, args.hidden_dim, args.common_dim]
    private_modal_list = [Layers(args=args) for _ in range(n_active)]
    global_modal = Layers(args=args)

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
    print(f"PMFH Layers: bit={args.bit} img={args.img_hidden_dim} txt={args.txt_hidden_dim}")
    print(f"Total params: {n_params}  (state_dict bytes per model: {model_bytes})")
    print(f"Active clients: {active} (n={n_active}), samples: {[n_per[c] for c in active]}")
    print("Engine: PMFH (upstream LocalUpdate + FedAvg of global model + prototype exchange)")
    print("Speed: " + ("upstream ConLoss loop + anomaly mode" if args.slow_upstream
                       else "FastConLoss (same values) + anomaly mode off"))
    print(f"Eval: kNN k={args.knn_k} Hamming on test ({len(q_lab)}) vs train db ({len(db_lab)}); shard={args.valid_batch_size}")
    print(f"Eval: linear probe Linear({args.bit}->{args.classes}) on frozen continuous codes, "
          f"fit on train ({len(db_lab)}) each round, L-BFGS zero-init, l2={args.probe_l2:g}, "
          f"max_iter={args.probe_iters}; shard={args.valid_batch_size}")

    logger = TableLogger()
    history = {k: [] for k in (
        "test_acc", "test_acc_probe", "mAP", "private_loss", "global_loss",
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
            local_model = LocalUpdate(
                args=args, id=idx, dataset=datasetall, idxs=client_idcs_train[idx],
                idcs_retrieval=client_idcs_empty, idcs_query=client_idcs_empty,
            )
            # Same loader as upstream, but never leave a final batch of 1 (BatchNorm1d).
            if not args.slow_upstream:
                local_model.criterion_CL = FastConLoss(temperature=0.07)
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

        # ---- eval: global model hash codes on the official test set ----
        q_codes = _hash_codes(global_modal, q_img, q_txt, device)
        db_codes = _hash_codes(global_modal, db_img, db_txt, device)
        pred = _knn_predict(q_codes, db_codes, db_lab_t, args.knn_k, args.classes).cpu().numpy()
        test_acc = _shard_mean_acc((pred == q_lab).astype(np.float64), args.valid_batch_size)
        test_map = float(calculate_map(
            qu_B=q_codes.cpu().numpy(), re_B=db_codes.cpu().numpy(),
            qu_L=_one_hot(q_lab, args.classes), re_L=_one_hot(db_lab, args.classes),
        ))
        # ---- eval: linear probe on frozen codes, read out like IC / MFedMC v2 / Hybrid v2 ----
        pred_probe = _linear_probe_predict(
            _raw_codes(global_modal, db_img, db_txt, device), db_lab,
            _raw_codes(global_modal, q_img, q_txt, device),
            args.classes, args.probe_l2, args.probe_iters,
        ).cpu().numpy()
        test_acc_probe = _shard_mean_acc((pred_probe == q_lab).astype(np.float64), args.valid_batch_size)
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
            "test_acc_probe": test_acc_probe,
            "mAP": test_map,
            # "priv_loss": total_private_loss / n_active,
            # "glob_loss": total_global_loss / n_active,
            "up_MB": upload_mb,
            "cum_up_MB": total_upload,
            "down_MB": download_mb,
            "cum_down_MB": total_download,
            "round_s": round_s,
            "time_s": elapsed_s,
        })
        history["test_acc"].append(test_acc)
        history["test_acc_probe"].append(test_acc_probe)
        history["mAP"].append(test_map)
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
        f"Final test accuracy (kNN): {history['test_acc'][-1]:.4f} "
        f"({100.0 * history['test_acc'][-1]:.2f}%) | "
        f"Final test accuracy (linear probe): "
        f"{history['test_acc_probe'][-1]:.4f} "
        f"({100.0 * history['test_acc_probe'][-1]:.2f}%) | "
        f"Final mAP: {history['mAP'][-1]:.4f}"
    )

    file_name = os.path.join(results_dir, f"PMFH_UCI_HAR_bit{args.bit}.npz")
    np.savez(
        file_name,
        test_acc=np.asarray(history["test_acc"]),
        test_acc_probe=np.asarray(history["test_acc_probe"]),
        mAP=np.asarray(history["mAP"]),
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
        bit=np.array(args.bit),
        knn_k=np.array(args.knn_k),
        probe_l2=np.array(args.probe_l2),
        probe_iters=np.array(args.probe_iters),
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
