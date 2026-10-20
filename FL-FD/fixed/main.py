"""FL-FD (Qi et al., Information Fusion 2023) — fixed PyTorch port of FL-FD/main.py.

Input-level fusion on UP-Fall: [GAF(acc), GAF(gyro), camera frame-difference] -> 3 x 70 x 70,
the released 3-layer CNN, FedAvg with one client per subject.

Fixes vs. the released code:
  * real FedAvg: each selected client trains its own copy of the global model (the release gave
    every client the server's model object, so all uploaded deltas were zero)
  * per-client datasets are actually built (release indexed a (subject, activity, trial) dict by
    client id); test set is a proper tensor dataset
  * inputs are 70 x 70 as in the paper, matching the released Linear(588, 11)

Samples: one per (trial, wearable sensor), i.e. 5 per trial with --sensors all, as the released
data_processing.py builds them (and as the paper's per-class confusion counts imply).
"""

import argparse
import copy
import io
import os
import random
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.abspath(os.path.dirname(__file__))


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
    p = argparse.ArgumentParser(description="FL-FD (Inf. Fusion'23) federated fall detection, fixed port")
    p.add_argument("--cache", required=True, help="npz from prepare_upfall.py")
    p.add_argument("--task", choices=["activity", "fall"], default="activity",
                   help="activity = 11-class (paper 4.4.2); fall = fall/non-fall (paper 4.4.1)")
    p.add_argument("--sensors", type=str, default="all",
                   help="'all' or comma list of Ankle,RightPocket,Belt,Neck,Wrist (one sample per sensor)")
    p.add_argument("--rounds", type=int, default=None, help="default: 200 (activity) / 100 (fall), as the paper")
    p.add_argument("--clients_per_round", type=int, default=12)
    p.add_argument("--local_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=1e-4, help="released code: 0.0001")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--shuffle", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--results_dir", type=str, default=os.path.join(HERE, "results"))
    return p.parse_args()


class FLFDNet(nn.Module):
    """Released MyModel: 3 x (conv3x3 -> Softsign -> maxpool2 -> Dropout2d(0.2) -> BN) -> FC."""

    def __init__(self, num_classes, size=70):
        super().__init__()
        layers, c_in = [], 3
        for c_out in (3, 6, 12):
            layers += [nn.Conv2d(c_in, c_out, 3, 1), nn.Softsign(), nn.MaxPool2d(2, 2),
                       nn.Dropout2d(0.2), nn.BatchNorm2d(c_out)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        side = size
        for _ in range(3):
            side = (side - 2) // 2
        self.fc1 = nn.Linear(12 * side * side, num_classes)  # 588 at 70 x 70

    def forward(self, x):
        return self.fc1(torch.flatten(self.features(x), 1))


def load_clients(args):
    d = np.load(args.cache)
    names = d["sensors"].tolist()
    use = names if args.sensors == "all" else [s.strip() for s in args.sensors.split(",")]
    idx = [names.index(s) for s in use]

    gaf, cam = d["gaf"][:, idx], d["cam"]                    # (N, S, 2, H, W), (N, H, W)
    cam_rep = np.repeat(cam[:, None, None], len(idx), axis=1)  # (N, S, 1, H, W)
    x = np.concatenate([gaf, cam_rep], axis=2).reshape(-1, 3, *cam.shape[1:])
    rep = lambda a: np.repeat(a, len(idx))
    subject, activity, trial = rep(d["subject"]), rep(d["activity"]), rep(d["trial"])
    y = activity - 1 if args.task == "activity" else (activity > 5).astype(np.int64)  # fall=0, non-fall=1

    x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()
    train = trial != 3
    subjects = sorted(np.unique(subject[train]).tolist())  # subjects with training data in the cache
    clients = [TensorDataset(x[(subject == s) & train], y[(subject == s) & train]) for s in subjects]
    test = TensorDataset(x[~train], y[~train])
    meta = dict(subjects=subjects, sensors=use, size=int(d["size"]), n_trials=len(d["trial"]),
                samples_per_client=[len(c) for c in clients])
    return clients, test, meta


def local_train(global_model, dataset, args, device):
    model = copy.deepcopy(global_model)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=bool(args.shuffle))
    for _ in range(args.local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()
    return model.state_dict()


def fedavg(states):
    avg = {}
    for k in states[0]:
        stacked = torch.stack([s[k].float() for s in states])
        avg[k] = stacked.mean(0).to(states[0][k].dtype)
    return avg


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    loss, preds, ys = 0.0, [], []
    for xb, yb in DataLoader(dataset, batch_size=256):
        out = model(xb.to(device))
        loss += nn.functional.cross_entropy(out, yb.to(device), reduction="sum").item()
        preds.append(out.argmax(1).cpu())
        ys.append(yb)
    preds, ys = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    return (preds == ys).mean(), loss / len(ys), preds, ys


def _run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print(f"Requested {args.device} but CUDA is unavailable — using {device}")
    if args.rounds is None:
        args.rounds = 200 if args.task == "activity" else 100

    clients, test, meta = load_clients(args)
    if len(clients) < args.clients_per_round:
        print(f"WARNING: only {len(clients)} clients have data (camera download incomplete?) — "
              f"selecting {len(clients)} per round instead of {args.clients_per_round}")
        args.clients_per_round = len(clients)
    num_classes = 11 if args.task == "activity" else 2
    model = FLFDNet(num_classes, meta["size"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    round_bytes = 4 * n_params * args.clients_per_round

    print(f"UP-Fall cache: {meta['n_trials']} trials, sensors {meta['sensors']}")
    print(f"Clients (subjects): {meta['subjects']}")
    print(f"Train samples per client: {meta['samples_per_client']}; test samples: {len(test)}")
    print(f"Model params: {n_params}")
    print("Hyperparameters (resolved):")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")

    hist = {k: [] for k in ("test_acc", "test_loss", "upload_bytes")}
    best_acc, best_state = -1.0, None
    for r in range(args.rounds):
        selected = random.sample(range(len(clients)), args.clients_per_round)
        states = [local_train(model, clients[c], args, device) for c in selected]
        model.load_state_dict(fedavg(states))
        acc, loss, _, _ = evaluate(model, test, device)
        hist["test_acc"].append(acc)
        hist["test_loss"].append(loss)
        hist["upload_bytes"].append(round_bytes * (r + 1))
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())
        print(f"Round {r + 1:3d}/{args.rounds}  test_acc {100 * acc:6.2f}%  test_loss {loss:.4f}  "
              f"cum_up {hist['upload_bytes'][-1] / 2**20:.2f} MiB")

    final_acc, _, final_pred, ys = evaluate(model, test, device)
    model.load_state_dict(best_state)
    _, _, best_pred, _ = evaluate(model, test, device)
    labels = list(range(num_classes))
    names = [f"A{i + 1}" for i in labels] if args.task == "activity" else ["fall", "non-fall"]
    for tag, pred in (("final", final_pred), ("best-round", best_pred)):
        p, rc, f1, _ = precision_recall_fscore_support(ys, pred, labels=labels, average="weighted", zero_division=0)
        print(f"\n[{tag} global model] acc {100 * (pred == ys).mean():.3f}  precision {100 * p:.3f}  "
              f"recall {100 * rc:.3f}  F1 {100 * f1:.3f}  (weighted)")
    print("\nFinal model:\n" + classification_report(ys, final_pred, labels=labels, target_names=names,
                                                      digits=4, zero_division=0))

    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, _stem(args) + ".npz")
    np.savez(
        path, test_acc=np.array(hist["test_acc"]), test_loss=np.array(hist["test_loss"]),
        upload_bytes=np.array(hist["upload_bytes"]), n_params=n_params,
        confusion_final=confusion_matrix(ys, final_pred, labels=labels),
        confusion_best=confusion_matrix(ys, best_pred, labels=labels),
        subjects=np.array(meta["subjects"]), sensors=np.array(meta["sensors"]),
        samples_per_client=np.array(meta["samples_per_client"]), n_trials=meta["n_trials"],
    )
    print(f"Results saved to {path}")


def _stem(args):
    sensors = "all" if args.sensors == "all" else args.sensors.replace(",", "-")
    try:
        cam = f"C{int(np.load(args.cache)['camera'])}"
    except (OSError, KeyError):
        cam = "C"
    return (f"FLFD_UPFall_{args.task}_{cam}_{sensors}_R{args.rounds}_K{args.clients_per_round}_"
            f"E{args.local_epochs}_lr{args.lr:g}_seed{args.seed}")


def main():
    args = args_parser()
    log_buf = io.StringIO()
    stdout_orig = sys.stdout
    sys.stdout = _Tee(stdout_orig, log_buf)
    try:
        _run(args)
    finally:
        sys.stdout = stdout_orig
        updated_dir = os.path.join(HERE, "result_updated")
        os.makedirs(updated_dir, exist_ok=True)
        txt_path = os.path.join(updated_dir, _stem(args) + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(log_buf.getvalue())
        print(f"Console log saved to {txt_path}")


if __name__ == "__main__":
    main()
