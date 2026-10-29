"""FedMobile (Liu, Wang, Yuan; WWW 2025) on the shared UCI HAR / mHealth split, next to FLASH / FL-FD / Hybrid v2.

Ported from the released code (FedMobile/upstream/USC-HAD/MMFL_main, model_mhad, util; the 2-sensor HAR
setting closest to ours). Same algorithm, our data and model width:
  * data: HybridSketchMFedMC_v2's loaders, so the 10-client Dirichlet(0.1) split, labels and official test
    set are the ones the other baselines use; any number of modalities (UCI HAR 2, mHealth 4)
  * encoder: the released 2-D CNN expects raw (T x 3) windows; our caches are feature vectors, so each
    modality uses Hybrid v2's FeaExtractor + FeaRefiner (as FLASH --branch shared / FL-FD --encoder
    shared). As released, the encodings are concatenated and classified by one Linear layer.
  * generator (model_mhad.Generator): G(eps, one-hot y) -> a latent feature of the concat size; Linear +
    BatchNorm + ReLU + Linear, hidden = latent width, eps ~ U(0, 1) of size num_classes.

Per round (released main()):
  1. every client trains its own persistent model one local epoch with Adam (lr 1e-3, wd 1e-4, batch 16):
     loss = 1/2 CE + 1/2 (1/2 KL(model(x)/T || classifier(G(y))/T) + 1/2 CE(classifier(G(y')), y')),
     T = 0.5, y' uniform labels (the knowledge-distillation term; --distill 0 drops it)
  2. contribution-aware model aggregation (model_agg): client i is kept if its local contribution (its model's
     mean accuracy over the clients' data) plus its global contribution (accuracy of the running average of
     models 0..i minus last round's accuracy) beats last round's accuracy; if <= 3 are kept, all are used.
     Plain mean of the kept models' parameters, sent to every client.
  3. generator aggregation: k-means (k = 3) of the clients' generator outputs on 16 random labels, mean per
     cluster, exact Shapley value of each cluster (value = accuracy of the global classifier on generated
     features of the clients' labels), and every client gets the Shapley-weighted generator.
Missing-modality simulation (released default: miss_modal=[1], i.e. the first input, modality 0, zeroed for 80%
of each batch) is off by default (--miss_rate 0), since every client here holds every modality, as in the other
baselines; run_*_miss.sh turn it on (--miss_rate 0.8 --miss_modality 0).

Fixed vs. the release (both are evaluation bookkeeping in model_agg / eval_one, not the method):
  * local contribution: the release never resets its accumulator between models, so later clients'
    scores include earlier ones; here each model's score is its own mean accuracy.
  * global contribution: the release sums into client 0's model in place, so later "running averages"
    double count earlier models; here it is the true mean of models 0..i.
Kept as released: the generators are never optimised (the optimiser holds only the model; their gradients are
never applied), so they stay at their aggregated random init. --train_generator 1 adds them to the optimiser.
Data the release takes from elsewhere: contributions are measured on the clients' local validation splits;
here every client's data is training data (shared protocol), so its training set stands in. The generator
Shapley values only use the labels of a server-held set; here, the pooled client training labels. Neither uses
the test set.

Paper vs. release (this port follows the release): the paper (Eqs. 6-19, Alg. 1, App. C) distils on features
with no temperature and trains the generator jointly on the clients, weights every client by
n_k * local score * loss reduction (no threshold), uses k = 5 clusters, SGD (momentum 0.9), E = 5 local
epochs and T = 100 rounds. The released code does what is described above instead.

Communication: all clients upload model + generator every round and download the global model + generator
(nothing is downloaded in round 1: each client starts from its own init, as released). 4 bytes x every
parameter; BatchNorm running stats stay local (the release only exchanges parameters()). The release evaluates each client model
on every client's data at the server; the extra transfers that would need are not billed.
Reported in MB = 1e6 bytes, as the baselines' tables.
"""

import argparse
import copy
import io
import itertools
import os
import random
import sys
import time

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.append(os.path.join(ROOT, "HybridSketchMFedMC_v2"))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dataset import load_mhealth_mm, load_uci_har_mm  # HybridSketchMFedMC_v2
from federated import evaluate_model_ic_shards  # HybridSketchMFedMC_v2
from models import FeaExtractor, FeaRefiner  # HybridSketchMFedMC_v2 (as FLASH's --branch shared)
from utils import TableLogger, Timer  # HybridSketchMFedMC_v2


def args_parser():
    p = argparse.ArgumentParser(description="FedMobile (WWW'25) on the shared UCI HAR / mHealth split")
    p.add_argument("--dataset_dir", type=str, required=True, help="e.g. datasets/uci_har_mm or datasets/mhealth_mm")
    p.add_argument("--dataset", choices=["auto", "uci_har", "mhealth"], default="auto",
                   help="loader; auto = mhealth if 'mhealth' is in the folder name, else uci_har")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--dirichlet_alpha", type=float, default=0.1)
    # FedMobile recipe (released USC-HAD/MMFL_main parse_option / set_optimizer)
    p.add_argument("--rounds", type=int, default=150, help="released: 150 epochs, one round each")
    p.add_argument("--local_epochs", type=int, default=1, help="released: 1 local epoch per round")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr_decay_epochs", type=str, default="350,400,450")
    p.add_argument("--lr_decay_rate", type=float, default=0.02)
    p.add_argument("--distill", type=int, default=1,
                   help="1 = FedMobile (KD term + contribution-aware aggregation + Shapley generator); "
                        "0 = released model_agg_org (plain mean, no generator)")
    p.add_argument("--kd_temp", type=float, default=0.5)
    p.add_argument("--num_clusters", type=int, default=5)
    p.add_argument("--min_selected", type=int, default=3,
                   help="released: if <= 3 clients pass the contribution test, all are aggregated")
    p.add_argument("--gen_labels", type=int, default=16, help="labels drawn for the generator k-means")
    p.add_argument("--train_generator", type=int, default=0, help="released: 0 (generators never stepped)")
    p.add_argument("--miss_modality", type=int, default=0,
                   help="0-based modality zeroed when --miss_rate > 0 (released miss_modal=[1] = first input = 0)")
    p.add_argument("--miss_rate", type=float, default=0.0, help="released: 0.8; 0 = every modality present")
    # encoder (Hybrid v2's, as FLASH / FL-FD shared ports)
    p.add_argument("--feat_dim", type=int, default=64, help="encoder width per modality (FLASH shared runs: 64)")
    p.add_argument("--mm_dropout", type=float, default=0.3)
    p.add_argument("--valid_batch_size", type=int, default=8, help="test shard size (as the baselines)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--results_dir", type=str, default=os.path.join(HERE, "results"))
    args = p.parse_args()
    args.lr_decay_epochs = [int(e) for e in args.lr_decay_epochs.split(",") if e]
    return args


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


class FedMobileNet(nn.Module):
    """Per-modality Hybrid v2 encoder -> concat -> one Linear (released MyMMModel.classifier).

    ``forward(*xs)`` returns ``(logits, feature)`` like MyMMModel; ``forward(latent=z)`` classifies a
    generated feature (MyMMModel's ``latent=True`` path).
    """

    def __init__(self, mod_dims, num_classes, feat_dim=64, dropout=0.3):
        super().__init__()
        self.mod_dims = [int(d) for d in mod_dims]
        self.extractors = nn.ModuleList([FeaExtractor(d, feat_dim, dropout) for d in self.mod_dims])
        self.refiners = nn.ModuleList([FeaRefiner(feat_dim) for _ in self.mod_dims])
        for m in self.modules():  # Hybrid v2's encoder init; the classifier keeps PyTorch's, as released
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.latent_dim = feat_dim * len(self.mod_dims)
        self.classifier = nn.Linear(self.latent_dim, num_classes)

    def forward(self, *xs, latent=None):
        if latent is not None:
            return self.classifier(latent), latent
        feats = []
        for x, ext, ref in zip(xs, self.extractors, self.refiners):
            f = ext(x)
            feats.append(f * ref(f))
        feature = torch.cat(feats, dim=1)
        return self.classifier(feature), feature


class Generator(nn.Module):
    """Released model_mhad.Generator: z = [eps ~ U(0,1)^C, one-hot(y)] -> Linear -> BN -> ReLU -> Linear."""

    def __init__(self, num_classes, latent_dim):
        super().__init__()
        self.num_classes = num_classes
        self.fc_layers = nn.ModuleList([nn.Linear(2 * num_classes, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU()])
        self.representation_layer = nn.Linear(latent_dim, latent_dim)

    def forward(self, labels):
        eps = torch.rand((labels.shape[0], self.num_classes), device=labels.device)
        z = torch.cat((eps, F.one_hot(labels, self.num_classes).float()), dim=1)
        for layer in self.fc_layers:
            z = layer(z)
        return self.representation_layer(z)


def _state_bytes(module):
    """Bytes exchanged per model: parameters only (released aggregation never sends BatchNorm buffers)."""
    return 4 * sum(p.numel() for p in module.parameters())


def _copy_params(dst, src):
    """Released aggregation copies parameters only (no BatchNorm buffers)."""
    with torch.no_grad():
        for p, q in zip(dst.parameters(), src.parameters()):
            p.copy_(q)


def _mean_params(dst, srcs, weights=None):
    weights = weights if weights is not None else [1.0 / len(srcs)] * len(srcs)
    with torch.no_grad():
        for i, p in enumerate(dst.parameters()):
            p.copy_(sum(w * list(s.parameters())[i] for s, w in zip(srcs, weights)))


@torch.no_grad()
def _accuracy(model, data, bs=256):
    """Percent accuracy of ``model`` on one client's tensors (released validate_multi's top-1)."""
    n = len(data["y"])
    if n == 0:
        return None
    model.eval()
    correct = 0
    for s in range(0, n, bs):
        logits, _ = model(*[x[s:s + bs] for x in data["xs"]])
        correct += (logits.argmax(1) == data["y"][s:s + bs]).sum().item()
    return 100.0 * correct / n


def _mean_client_acc(model, clients):
    accs = [a for a in (_accuracy(model, c) for c in clients) if a is not None]
    return float(np.mean(accs))


def _kmeans(x, k, rng, iters=100, tol=1e-4):
    """Euclidean Lloyd's k-means (kmeans_pytorch's defaults: random initial points, stop on center shift)."""
    k = min(k, len(x))
    centers = x[rng.choice(len(x), k, replace=False)].clone()
    for _ in range(iters):
        ids = torch.cdist(x, centers).argmin(1)
        new = torch.stack([x[ids == j].mean(0) if (ids == j).any() else centers[j] for j in range(k)])
        shift = ((new - centers) ** 2).sum(1).sqrt().sum()
        centers = new
        if shift ** 2 < tol:
            break
    return torch.cdist(x, centers).argmin(1)


def train_client(model, gen, optimizer, data, args, num_classes):
    """Released train_multi: one local epoch, CE + (optional) generator KD; missing simulation if enabled."""
    model.train()
    n = len(data["y"])
    if n < 2:
        return float("nan")
    losses = []
    for _ in range(args.local_epochs):
        perm = torch.randperm(n, device=data["y"].device)
        for s in range(0, n, args.bs):
            idx = perm[s:s + args.bs]
            if len(idx) < 2:  # BatchNorm in the generator needs >= 2 samples (as PMFH's drop of size-1 batches)
                continue
            xs = [x[idx].clone() for x in data["xs"]]
            y = data["y"][idx]
            bsz = len(y)
            if args.miss_rate > 0 and 0 <= args.miss_modality < len(xs):
                zero = torch.tensor(sorted(random.sample(range(bsz), int(bsz * args.miss_rate))), dtype=torch.long)
                if len(zero):
                    xs[args.miss_modality][zero] = 0
            output, _ = model(*xs)
            loss = F.cross_entropy(output, y)
            if args.distill:
                out_gen, _ = model(latent=gen(y))
                loss_kl = F.kl_div(F.log_softmax(output / args.kd_temp, dim=1),
                                   F.softmax(out_gen / args.kd_temp, dim=1), reduction="batchmean")
                sampled_y = torch.randint(num_classes, (bsz,), device=y.device)
                out_s, _ = model(latent=gen(sampled_y))
                loss_t = F.nll_loss(F.log_softmax(out_s, dim=1), sampled_y)
                loss = 0.5 * (0.5 * loss_kl + 0.5 * loss_t) + 0.5 * loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def aggregate_models(models, clients, last_acc, args):
    """Released model_agg (contribution-aware selection, plain mean); returns (global model, kept ids)."""
    n = len(models)
    if not args.distill:
        kept = list(range(n))
    else:
        local_contri = [_mean_client_acc(m, clients) for m in models]
        global_contri = []
        running = copy.deepcopy(models[0])
        for i in range(n):
            _mean_params(running, models[:i + 1])
            global_contri.append(_mean_client_acc(running, clients) - last_acc)
        kept = [i for i in range(n) if local_contri[i] + global_contri[i] > last_acc]
        if len(kept) <= args.min_selected:
            kept = list(range(n))
    glob = copy.deepcopy(models[0])
    _mean_params(glob, [models[i] for i in kept])
    return glob, kept


@torch.no_grad()
def _gen_value(gen, glob, labels, bs=256):
    """Released model_evaluation_func: accuracy (%) of the classifier on generated features of ``labels``."""
    gen.eval()
    glob.eval()
    correct = 0
    for s in range(0, len(labels), bs):
        y = labels[s:s + bs]
        out, _ = glob(latent=gen(y))
        correct += (out.argmax(1) == y).sum().item()
    return 100.0 * correct / max(len(labels), 1)


def aggregate_generators(gens, glob, value_labels, args, num_classes, device, rng):
    """Released k-means + exact Shapley aggregation; every client's generator gets the result."""
    sys_label = torch.as_tensor(rng.choice(num_classes, size=args.gen_labels), device=device)
    with torch.no_grad():
        outs = torch.stack([g(sys_label).detach().flatten() for g in gens]).cpu()
    ids = _kmeans(outs, args.num_clusters, rng)
    cluster_gens = []
    for j in sorted(set(ids.tolist())):
        g = copy.deepcopy(gens[0])
        _mean_params(g, [gens[i] for i in range(len(gens)) if ids[i] == j])
        cluster_gens.append(g)

    history, scratch = {}, copy.deepcopy(gens[0])
    sv = np.zeros(len(cluster_gens))
    perms = list(itertools.permutations(range(len(cluster_gens))))
    for perm in perms:
        so_far = {}
        for c in perm:
            key = tuple(sorted(list(so_far) + [c]))
            if key not in history:
                _mean_params(scratch, [cluster_gens[i] for i in key])
                history[key] = _gen_value(scratch, glob, value_labels)
            so_far[c] = max(0.0, history[key] - sum(so_far.values()))
        for c, v in so_far.items():
            sv[c] += v
    sv /= len(perms)
    sv = sv / sv.sum() if sv.sum() > 0 else np.full(len(sv), 1.0 / len(sv))
    _mean_params(scratch, cluster_gens, list(sv))
    for g in gens:
        _copy_params(g, scratch)
    return ids.tolist(), sv


def _load(args):
    kind = args.dataset
    if kind == "auto":
        kind = "mhealth" if "mhealth" in os.path.basename(os.path.normpath(args.dataset_dir)) else "uci_har"
    loader = load_mhealth_mm if kind == "mhealth" else load_uci_har_mm
    return loader(args.dataset_dir, num_clients=args.num_clients, dirichlet_alpha=args.dirichlet_alpha, seed=args.seed)


def _run(args):
    timer = Timer()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print(f"Requested {args.device} but CUDA is unavailable — using {device}")

    print(f"Loading the shared split from {os.path.basename(os.path.normpath(args.dataset_dir))}...")
    clients_raw, test_raw, meta = _load(args)
    modalities, num_classes, mod_dims = meta["modalities"], meta["num_classes"], meta["mod_dims"]
    to_dev = lambda d: {"xs": [torch.as_tensor(x, device=device) for x in d["xs"]],
                        "y": torch.as_tensor(d["y"], dtype=torch.long, device=device)}
    clients = [to_dev(c) for c in clients_raw]
    value_labels = torch.cat([c["y"] for c in clients])  # labels only, for the generator Shapley value
    n_clients = len(clients)

    # one model / generator / Adam per client, each with its own init (released main())
    models = [FedMobileNet(mod_dims, num_classes, args.feat_dim, args.mm_dropout).to(device) for _ in range(n_clients)]
    gens = [Generator(num_classes, models[0].latent_dim).to(device) for _ in range(n_clients)]
    optimizers = []
    for m, g in zip(models, gens):
        params = list(m.parameters()) + (list(g.parameters()) if args.train_generator else [])
        optimizers.append(torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay))
    model_bytes = _state_bytes(models[0])
    gen_bytes = _state_bytes(gens[0]) if args.distill else 0
    per_client = model_bytes + gen_bytes
    n_params = sum(p.numel() for p in models[0].parameters())
    n_gen = sum(p.numel() for p in gens[0].parameters())

    print(f"Clients: {n_clients}, all participate every round")
    print(f"  samples/client: {meta['samples_per_client']}")
    print(f"Model: per-modality Hybrid v2 encoder {mod_dims} -> {args.feat_dim}, concat {models[0].latent_dim} -> "
          f"Linear -> {num_classes}; params {n_params}; bytes {model_bytes}")
    print(f"Generator: ({2 * num_classes}) -> {models[0].latent_dim} -> {models[0].latent_dim}; params {n_gen}; "
          f"bytes {_state_bytes(gens[0])} ({'sent every round' if args.distill else 'unused, --distill 0'})")
    print(f"Bytes per client per direction per round: {per_client}")
    print("Finished initializing in {:.2f} seconds".format(timer()))
    print("Hyperparameters (resolved):")
    for key, v in sorted(vars(args).items()):
        print(f"  {key}: {v}")

    history = {key: [] for key in ("test_acc", "test_loss", "upload_bytes", "download_bytes", "time_s")}
    kept_mask = np.zeros((args.rounds, n_clients), dtype=np.int8)
    shapley = []
    last_acc = 100.0 / num_classes  # released: ACC = 1/12 * 100 before the first round
    total_up = total_down = 0.0
    logger = TableLogger()
    train_start = time.perf_counter()
    prev_elapsed = 0.0
    glob = None

    for r in range(args.rounds):
        lr = args.lr * args.lr_decay_rate ** int(np.sum(r + 1 > np.asarray(args.lr_decay_epochs)))
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = lr
        down_bytes = 0 if r == 0 else n_clients * per_client  # round 1: each client starts from its own init (released)
        for c in range(n_clients):
            train_client(models[c], gens[c], optimizers[c], clients[c], args, num_classes)
        up_bytes = n_clients * per_client

        glob, kept = aggregate_models(models, clients, last_acc, args)
        kept_mask[r, kept] = 1
        for m in models:
            _copy_params(m, glob)
        if args.distill:
            _, sv = aggregate_generators(gens, glob, value_labels, args, num_classes, device, rng)
            shapley.append(np.pad(sv, (0, args.num_clusters - len(sv))))
        last_acc = _mean_client_acc(glob, clients)

        test_loss, test_acc = evaluate_model_ic_shards(glob, test_raw, device, shard_size=args.valid_batch_size)
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

    os.makedirs(args.results_dir, exist_ok=True)
    file_name = os.path.join(
        args.results_dir,
        f"FedMobile_{os.path.basename(os.path.normpath(args.dataset_dir))}_K{n_clients}_E{args.local_epochs}"
        f"_R{args.rounds}{_variant_tag(args, '_')}_seed{args.seed}.npz",
    )
    np.savez(
        file_name,
        test_loss=np.array(history["test_loss"]),
        test_acc=np.array(history["test_acc"]),
        upload_bytes=np.array(history["upload_bytes"]),
        download_bytes=np.array(history["download_bytes"]),
        time_s=np.array(history["time_s"]),  # cumulative training time per round (log's time_s)
        kept=kept_mask,
        shapley=np.array(shapley),
        modalities=np.array(modalities),
        dataset_dir=np.array(meta["dataset_dir"]),
        samples_per_client=np.array(meta["samples_per_client"]),
        total_download_mib=total_down,  # MB (1e6), key name as FLASH's npz
        total_upload_mib=total_up,
        n_params=n_params,
        model_bytes=model_bytes,
        gen_bytes=gen_bytes,
        clients_per_round=n_clients,
        local_epochs=np.array(args.local_epochs),
        feat_dim=np.array(args.feat_dim),
        distill=np.array(args.distill),
    )
    print(f"Results saved to {file_name}")
    return total_up


def _variant_tag(args, before="", after=""):
    """'' for FedMobile as released (feature width 64, KD on, no missing); else what differs."""
    tags = []
    if args.feat_dim != 64:
        tags.append(f"fd{args.feat_dim}")
    if not args.distill:
        tags.append("nodistill")
    if args.train_generator:
        tags.append("traingen")
    if args.miss_rate > 0:
        tags.append(f"miss{args.miss_modality}r{args.miss_rate:g}")
    return before + "_".join(tags) + after if tags else ""


def _txt_stem(args, cum_up_mb):
    dataset = os.path.basename(os.path.normpath(args.dataset_dir))
    up_str = f"{cum_up_mb:.2f}MB" if cum_up_mb is not None else "NAMB"
    return (
        f"FedMobile_{dataset}_mm_{up_str}_dirichlet{args.dirichlet_alpha:g}_it{args.rounds}_"
        f"K{args.num_clients}_E{args.local_epochs}_lr{args.lr:g}_"
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
