"""FLASH federated loop (``FLASH/code/federated/federated_learning.py:360-456``).

Every round, every client:
  1. starts from its own latest local model (round 1: the global init) and
     overwrites only the branch(es) its policy draw selects with the global
     weights (``select_branch``; ``policy='all'`` = ``aggregatin_on_all``),
  2. trains with Adam + categorical CE (+ fusion regularizers) for up to
     ``local_epochs`` with EarlyStopping(val_loss, patience) on a local
     validation split,
  3. uploads its full model.
The server takes the plain mean of all uploaded models (BN moving stats
included) and evaluates it on the shared test set.

Two artifacts of the released Keras code are not reproduced: there the
"local" model is the same object as the global one (round 0 trains the
clients sequentially) and one Adam state is shared by all clients. Here each
client trains an independent copy with a fresh Adam.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from federated import evaluate_model_ic_shards  # HybridSketchMFedMC_v2
from utils import TableLogger  # HybridSketchMFedMC_v2

BYTES_PER_FLOAT32 = 4


def parse_policy(policy, n_groups):
    """'all' -> None (sync every branch); else M+1 probabilities."""
    if policy == "all":
        return None
    probs = [float(p) for p in policy.split(",")]
    if len(probs) != n_groups:
        raise ValueError(
            f"--policy needs {n_groups} probabilities "
            f"(one per modality branch + fusion), got {len(probs)}"
        )
    if sum(probs) > 1.0 + 1e-6:
        raise ValueError(f"--policy probabilities sum to {sum(probs)} > 1")
    return np.asarray(probs)


def select_branch(probs, rng):
    """FLASH ``select_branch``: one uniform draw picks at most one branch."""
    r = rng.rand()
    edges = np.cumsum(probs)
    hit = np.nonzero(r < edges)[0]
    return int(hit[0]) if hit.size else None


def _state_bytes(state, keys):
    return BYTES_PER_FLOAT32 * sum(
        state[k].numel() for k in keys if state[k].is_floating_point()
    )


def _split_train_val(client, val_frac, rng, device):
    n = len(client["y"])
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    if n_val < 1 or n - n_val < 1:
        n_val = 0

    def take(idx):
        return {
            "xs": [torch.as_tensor(x[idx], dtype=torch.float32, device=device) for x in client["xs"]],
            "y": torch.as_tensor(client["y"][idx], dtype=torch.long, device=device),
        }

    train = take(perm[n_val:])
    val = take(perm[:n_val]) if n_val else None
    return train, val


class KerasAdam(torch.optim.Optimizer):
    """tf.keras (TF 2.2) Adam, which puts epsilon outside the bias correction:

        lr_t = lr * sqrt(1 - b2^t) / (1 - b1^t)
        w   -= lr_t * m / (sqrt(v) + eps)

    ``torch.optim.Adam`` adds eps to the bias-corrected sqrt(v), which moves
    small-gradient weights by up to a full lr in the first steps where Keras
    barely moves them. Each local fit starts a fresh optimizer, so those first
    steps recur every round.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["t"] += 1
                t = st["t"]
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                lr_t = group["lr"] * (1 - b2 ** t) ** 0.5 / (1 - b1 ** t)
                p.addcdiv_(st["m"], st["v"].sqrt().add_(group["eps"]), value=-lr_t)


def _loss(model, xs, y):
    logits, _ = model(*xs)
    return F.cross_entropy(logits, y) + model.reg_loss(), logits


def _local_fit(model, train, val, args, gen):
    """Keras ``fit`` + EarlyStopping(val_loss, patience, restore_best_weights).

    As in TF 2.2, best weights are restored only when early stopping fires.
    Returns (epochs_run, last-epoch train loss, last-epoch train acc).
    """
    opt = KerasAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-7)
    n = len(train["y"])
    best_val, best_state, wait = float("inf"), None, 0
    epochs_run, tr_loss, tr_acc = 0, float("nan"), float("nan")
    for epoch in range(args.local_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(train["y"].device) if args.shuffle else torch.arange(n, device=train["y"].device)
        loss_sum, correct = 0.0, 0
        for start in range(0, n, args.bs):
            idx = perm[start:start + args.bs]
            xs = [x[idx] for x in train["xs"]]
            y = train["y"][idx]
            loss, logits = _loss(model, xs, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(idx)
            correct += int((logits.argmax(1) == y).sum().item())
        epochs_run = epoch + 1
        tr_loss, tr_acc = loss_sum / n, correct / n
        if val is None:
            continue
        model.eval()
        with torch.no_grad():
            val_loss = float(_loss(model, val["xs"], val["y"])[0].item())
        if val_loss < best_val:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= args.patience:
                model.load_state_dict(best_state)
                break
    return epochs_run, tr_loss, tr_acc


def _average(states, weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = {}
    for k in states[0]:
        if states[0][k].is_floating_point():
            out[k] = sum(float(wi) * s[k] for wi, s in zip(w, states))
        else:  # BN counters etc.: not part of the Keras weights
            out[k] = states[0][k].clone()
    return out


def run_flash(args, model, clients, global_test, modalities, device):
    n_clients = len(clients)
    groups = model.branch_groups()
    group_names = list(groups.keys())
    probs = parse_policy(args.policy, len(group_names))

    global_state = copy.deepcopy(model.state_dict())
    all_keys = list(global_state.keys())
    model_bytes = _state_bytes(global_state, all_keys)
    group_bytes = {g: _state_bytes(global_state, ks) for g, ks in groups.items()}
    n_params = sum(p.numel() for p in model.parameters())

    split_rng = np.random.RandomState(args.seed)
    policy_rng = np.random.RandomState(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    splits = []
    for c in clients:
        if len(c["y"]) == 0:
            splits.append((None, None))
        else:
            splits.append(_split_train_val(c, args.val_frac, split_rng, device))
    n_train = [0 if tr is None else len(tr["y"]) for tr, _ in splits]
    n_val = [0 if va is None else len(va["y"]) for _, va in splits]
    no_val = [i for i in range(n_clients) if n_train[i] > 0 and n_val[i] == 0]

    print(
        f"FLASH config: mods={list(modalities)} dims={model.mod_dims} "
        f"branch={'coord_mlp' if model.branch == 'gps' else 'shared f*FeaRefiner(f), f=FeaExtractor(x)'}"
        f" x{len(modalities)} branch_out={model.branch_out}"
        f" fusion_in={model.branch_out * len(modalities)}"
        + (f" branch_dims={list(model.branch_dims)}" if model.branch == "gps" else "")
        + f" fusion_dims={list(model.fusion_dims)}"
    )
    print(
        "Branch params (MB): "
        + ", ".join(
            f"{g}={group_bytes[g] / 1e6:.3f}" for g in group_names
        )
    )
    print(f"Total params: {n_params}  (model bytes per upload: {model_bytes})")
    print(f"Local train/val samples: {n_train} / {n_val}")
    if no_val:
        print(
            f"Clients {no_val} are too small for a validation split: they train "
            f"all {args.local_epochs} local epochs without early stopping."
        )
    policy_str = "P_All (every branch synced)" if probs is None else (
        "random branch " + dict(zip(group_names, probs.tolist())).__repr__()
    )
    print(
        f"Engine: FLASH policy={policy_str} | Adam lr={args.lr} bs={args.bs} "
        f"local_epochs<={args.local_epochs} early_stop(patience={args.patience}, "
        f"val_frac={args.val_frac}) | aggregation="
        f"{'sample-weighted' if args.weighted_avg else 'unweighted mean'} | all clients every round"
    )

    client_states = [None] * n_clients
    logger = TableLogger()
    history = {k: [] for k in (
        "train_loss", "train_acc", "test_loss", "test_acc",
        "upload_bytes", "download_bytes", "epochs_run", "branch_synced",
    )}
    total_up = total_down = 0.0
    train_start = time.perf_counter()
    prev_elapsed = 0.0

    for rnd in range(args.rounds):
        local_states, weights = [], []
        down_bytes = 0
        synced_row = np.zeros((n_clients, len(group_names)), dtype=np.int8)
        epochs_row = np.zeros(n_clients, dtype=np.int64)
        tr_losses, tr_accs = [], []

        for cid in range(n_clients):
            train, val = splits[cid]
            if train is None:
                continue
            if probs is None:
                synced = list(range(len(group_names)))
            else:
                pick = select_branch(probs, policy_rng)
                synced = [] if pick is None else [pick]

            if rnd == 0:
                # Global init is shared before training; not billed (as baselines).
                state = copy.deepcopy(global_state)
            else:
                state = client_states[cid]
                for gi in synced:
                    for k in groups[group_names[gi]]:
                        state[k] = global_state[k].clone()
                    down_bytes += group_bytes[group_names[gi]]
                    synced_row[cid, gi] = 1

            model.load_state_dict(state)
            ep, tl, ta = _local_fit(model, train, val, args, gen)
            epochs_row[cid] = ep
            tr_losses.append(tl)
            tr_accs.append(ta)
            client_states[cid] = copy.deepcopy(model.state_dict())
            local_states.append(client_states[cid])
            weights.append(n_train[cid] if args.weighted_avg else 1.0)

        global_state = _average(local_states, weights)
        model.load_state_dict(global_state)
        up_bytes = len(local_states) * model_bytes

        test_loss, test_acc = evaluate_model_ic_shards(
            model, global_test, device, shard_size=args.valid_batch_size
        )

        elapsed = time.perf_counter() - train_start
        round_s = elapsed - prev_elapsed
        prev_elapsed = elapsed
        up_mb, down_mb = up_bytes / 1e6, down_bytes / 1e6
        total_up += up_mb
        total_down += down_mb
        logger.append({
            "iter": rnd + 1,
            "test_acc": test_acc,
            "up_MB": up_mb,
            "cum_up_MB": total_up,
            "down_MB": down_mb,
            "cum_down_MB": total_down,
            "round_s": round_s,
            "time_s": elapsed,
        })

        history["train_loss"].append(float(np.mean(tr_losses)))
        history["train_acc"].append(float(np.mean(tr_accs)))
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["upload_bytes"].append(up_bytes)
        history["download_bytes"].append(down_bytes)
        history["epochs_run"].append(epochs_row)
        history["branch_synced"].append(synced_row)

    print("Total Download (MiB): {:0.2f}".format(total_down))
    print("Total Upload (MiB): {:0.2f}".format(total_up))
    print("Avg Download Per Client: {:0.2f}".format(total_down / max(n_clients, 1)))
    print("Avg Upload Per Client: {:0.2f}".format(total_up / max(n_clients, 1)))
    print("Cumulative train comm (download+upload): {:.4f} MiB".format(total_down + total_up))

    epochs = np.stack(history["epochs_run"], axis=0)
    active = epochs[:, [i for i in range(n_clients) if n_train[i] > 0]]
    print(
        f"\n=== Local epochs run (early stopping, max {args.local_epochs}) ===\n"
        f"Per round mean: {np.round(active.mean(axis=1), 1).tolist()}\n"
        f"Overall mean={active.mean():.1f} min={active.min()} max={active.max()}"
    )
    synced = np.stack(history["branch_synced"], axis=0)
    if args.rounds > 1:
        freq = synced[1:].mean(axis=(0, 1))
        print(
            "=== Branch sync frequency (rounds 2+, fraction of client-rounds) ===\n"
            + ", ".join(f"{g}={f:.3f}" for g, f in zip(group_names, freq))
        )

    for k in ("train_loss", "train_acc", "test_loss", "test_acc", "upload_bytes", "download_bytes"):
        history[k] = np.asarray(history[k])
    history["epochs_run"] = epochs
    history["branch_synced"] = synced
    history["branch_names"] = group_names
    history["total_download_mib"] = total_down
    history["total_upload_mib"] = total_up
    history["n_params"] = n_params
    return history
