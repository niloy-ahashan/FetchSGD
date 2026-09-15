"""FetchSGD federated loop + MFedMC client/modality selection (v2).

Unlike v1 (local SGD then client-weighted delta sketches), this engine
matches Independent Compression / original FetchSGD:

- example-weighted sketch aggregation (n_i / N)
- upload a gradient sketch by default (optional weight-delta sketch)
- no local SGD when local_epochs == 1
- IC-style weight decay: grad += (wd / n_participating) * weights
- server applies scheduled LR to gradient sketches (LR 1.0 for deltas)
"""

from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import build_param_index_maps
from selection import (
    compute_client_priority,
    compute_modality_priority,
    select_clients,
    shap_from_fusion,
)
from sketch import fetchsgd_server_step, sketch_vector
from utils import (
    PiecewiseLinear,
    TableLogger,
    Timer,
    get_grad_vec,
    get_param_vec,
    normalization,
    set_param_vec,
)

BYTES_PER_FLOAT32 = 4
DEQUE_MAXLEN_MULT = 10
_ce = nn.CrossEntropyLoss()


def _coord_diff_count(current, prev):
    """Number of coordinates that differ (IC ``ceil(abs).clamp(0,1).sum()``)."""
    return int(torch.ceil((current - prev).abs()).clamp(0, 1).sum().item())


def _init_download_state(args, n_clients, k_clients, grad_size, global_vec, device):
    """Same two download-tracking branches as CommEfficient FedModel."""
    if args.num_epochs <= 1 and args.local_batch_size == -1:
        return {
            "simple": True,
            "updated_since_init": torch.zeros(grad_size, dtype=torch.bool, device=device),
            "prev_ps_weights": global_vec.clone(),
        }
    participation = max(float(k_clients) / max(n_clients, 1), 1e-8)
    maxlen = max(1, int(DEQUE_MAXLEN_MULT / participation))
    return {
        "simple": False,
        "ps_weights_history": deque([], maxlen=maxlen),
        "client_num_stale_iters": torch.zeros(n_clients, dtype=torch.long),
    }


def _download_bytes_ic(state, selected, global_vec):
    """Bytes participating clients download *before* this round's PS update."""
    selected = np.asarray(selected, dtype=int)
    if selected.size == 0:
        return 0
    if state["simple"]:
        diff = global_vec - state["prev_ps_weights"]
        updated = torch.ceil(diff.abs()).clamp(0, 1).bool()
        state["updated_since_init"] |= updated
        state["prev_ps_weights"] = global_vec.clone()
        per_client = BYTES_PER_FLOAT32 * int(state["updated_since_init"].sum().item())
        return int(selected.size) * per_client

    history = state["ps_weights_history"]
    stale_iters = state["client_num_stale_iters"]
    history.append(global_vec.detach().cpu().clone())
    maxlen = history.maxlen
    current = history[-1]
    total = 0
    for cid in selected:
        stale = int(stale_iters[int(cid)].clamp(0, maxlen - 1).item())
        prev = history[-(stale + 1)]
        total += BYTES_PER_FLOAT32 * _coord_diff_count(current, prev)
    stale_iters[selected] = 0
    stale_iters += 1
    return int(total)


def _make_loader(client, batch_size, shuffle):
    n = len(client["y"])
    if n == 0:
        return None, 0
    tensors = [torch.as_tensor(x, dtype=torch.float32) for x in client["xs"]]
    tensors.append(torch.as_tensor(client["y"], dtype=torch.long))
    ds = TensorDataset(*tensors)
    bs = n if batch_size is None or batch_size <= 0 else min(int(batch_size), n)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle), n


def _split_batch(batch):
    *mods, y = batch
    return list(mods), y


def evaluate_model(model, data, device, batch_size=256):
    model.eval()
    n = len(data["y"])
    if n == 0:
        return float("nan"), float("nan")
    loader, _ = _make_loader(data, batch_size, shuffle=False)
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            mods, y = _split_batch(batch)
            mods = [m.to(device) for m in mods]
            y = y.to(device)
            logits, _ = model(*mods)
            loss = _ce(logits, y)
            total_loss += float(loss.item()) * y.size(0)
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.size(0))
    return total_loss / max(total, 1), total_correct / max(total, 1)


def evaluate_model_ic_shards(model, data, device, shard_size=8):
    """IC val metric: unweighted mean of per-shard mean loss/acc.

    CommEfficient splits test batches into ``valid_batch_size`` shards
    (default 8), then ``np.mean``s those shard metrics. A leftover shard
    of 3 counts as much as a shard of 8.
    """
    model.eval()
    n = len(data["y"])
    if n == 0:
        return float("nan"), float("nan")
    shard_size = max(1, int(shard_size))
    xs = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in data["xs"]]
    y = torch.as_tensor(data["y"], dtype=torch.long, device=device)
    losses = []
    accs = []
    with torch.no_grad():
        for start in range(0, n, shard_size):
            end = min(start + shard_size, n)
            mods = [x[start:end] for x in xs]
            yt = y[start:end]
            logits, _ = model(*mods)
            loss = _ce(logits, yt)
            acc = (logits.argmax(dim=1) == yt).float().mean()
            losses.append(float(loss.item()))
            accs.append(float(acc.item()))
    return float(np.mean(losses)), float(np.mean(accs))


def _apply_weight_decay(grad, weights, weight_decay, n_participating):
    """IC / FetchSGD ``get_grad``: add (wd / num_workers) * weights."""
    if weight_decay == 0.0 or n_participating <= 0:
        return grad
    return grad + (weight_decay / float(n_participating)) * weights


def _dataset_mean_grad(model, client, batch_size, device):
    """Mean CE gradient over the client's examples (no optimizer step)."""
    n = len(client["y"])
    if n == 0:
        return None, float("nan"), float("nan")
    model.train()
    model.zero_grad(set_to_none=True)
    loader, _ = _make_loader(client, batch_size, shuffle=False)
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        mods, y = _split_batch(batch)
        mods = [m.to(device) for m in mods]
        y = y.to(device)
        logits, _ = model(*mods)
        loss = _ce(logits, y)
        (loss * (y.size(0) / n)).backward()
        total_loss += float(loss.item()) * y.size(0)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.size(0))
    grad = get_grad_vec(model)
    loss_v = total_loss / max(total, 1)
    acc_v = total_correct / max(total, 1)
    return grad, loss_v, acc_v


def _mask_coords(vec, mod_indices, shared_indices, selected_mask, device):
    """Keep shared coords + selected modality branches; zero the rest."""
    keep = torch.zeros_like(vec, dtype=torch.bool)
    keep[shared_indices.to(device)] = True
    for i, chosen in enumerate(selected_mask):
        if chosen:
            idx = mod_indices[i].to(device)
            if idx.numel() > 0:
                keep[idx] = True
    masked = vec.clone()
    masked[~keep] = 0
    return masked


def _resolve_upload_object(args):
    upload_object = str(getattr(args, "upload_object", "gradient")).lower()
    if int(args.local_epochs) <= 1 and upload_object == "delta":
        print(
            "WARNING: --upload_object delta with --local_epochs 1 would be a "
            "zero delta (no local SGD). Falling back to gradient sketch."
        )
        return "gradient"
    return upload_object


def _client_upload_vec(
    model,
    client,
    global_vec,
    local_epochs,
    batch_size,
    lr,
    weight_decay,
    n_participating,
    upload_object,
    device,
):
    """Vector to sketch: mean gradient, or (global - local) after local SGD."""
    n = len(client["y"])
    if n == 0:
        return None, float("nan"), float("nan")

    set_param_vec(model, global_vec)

    if local_epochs <= 1:
        grad, loss_v, acc_v = _dataset_mean_grad(model, client, batch_size, device)
        if grad is None:
            return None, loss_v, acc_v
        grad = _apply_weight_decay(grad, global_vec, weight_decay, n_participating)
        model.zero_grad(set_to_none=True)
        return grad, loss_v, acc_v

    local = copy.deepcopy(model).to(device)
    local.train()
    loader, _ = _make_loader(client, batch_size, shuffle=True)
    grad_sum = None
    n_steps = 0
    last_loss = float("nan")
    last_acc = float("nan")

    for _ in range(local_epochs):
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch in loader:
            mods, y = _split_batch(batch)
            mods = [m.to(device) for m in mods]
            y = y.to(device)
            local.zero_grad(set_to_none=True)
            logits, _ = local(*mods)
            loss = _ce(logits, y)
            loss.backward()
            g = get_grad_vec(local)
            w = get_param_vec(local).to(device)
            g = _apply_weight_decay(g, w, weight_decay, n_participating)
            set_param_vec(local, w - float(lr) * g)
            if grad_sum is None:
                grad_sum = g
            else:
                grad_sum = grad_sum + g
            n_steps += 1
            total_loss += float(loss.item()) * y.size(0)
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.size(0))
        last_loss = total_loss / max(total, 1)
        last_acc = total_correct / max(total, 1)

    if upload_object == "delta":
        local_vec = get_param_vec(local).to(device)
        vec = global_vec - local_vec
    else:
        vec = grad_sum / max(n_steps, 1)

    del local
    set_param_vec(model, global_vec)
    model.zero_grad(set_to_none=True)
    return vec, last_loss, last_acc


def run_federated(args, model, clients, global_test, modalities, device):
    n_clients = len(clients)
    n_mod = len(modalities)
    rng = np.random.RandomState(args.seed)
    upload_object = _resolve_upload_object(args)
    args.effective_upload_object = upload_object

    global_vec = get_param_vec(model).to(device)
    grad_size = int(global_vec.numel())
    args.grad_size = grad_size
    mod_indices, shared_indices, _ = build_param_index_maps(model)
    branch_sizes = np.array(
        [int(ix.numel()) for ix in mod_indices], dtype=np.float64
    )

    Vvelocity = torch.zeros(args.num_rows, args.num_cols, device=device)
    Verror = torch.zeros(args.num_rows, args.num_cols, device=device)

    recency_history = np.full((n_clients, n_mod), -1, dtype=np.int64)
    client_last_selected = np.full((n_clients,), -1, dtype=np.int64)
    prev_losses = np.full((n_clients,), np.nan)

    lr_schedule = PiecewiseLinear(
        [0, args.pivot_epoch, args.num_epochs],
        [0, args.lr_scale, 0],
    )
    logger = TableLogger()
    timer = Timer()

    upload_per_client = args.num_rows * args.num_cols * BYTES_PER_FLOAT32
    total_download = 0.0
    total_upload = 0.0

    print(
        f"SketchFusionBNet config: fusion_mode={getattr(args, 'fusion_mode', 'sketch')} "
        f"mods={list(modalities)} "
        f"dims={[int(x.shape[1]) if x.ndim==2 else 0 for x in clients[0]['xs']]}"
    )
    print(f"Grad size: {grad_size}")
    print(f"Total params: {grad_size}")
    print(
        f"Engine: FetchSGD upload_object={upload_object} "
        f"local_epochs={args.local_epochs} "
        f"weight_decay={args.weight_decay} "
        f"aggregation=example-weighted"
    )
    print(
        f"Client select: {args.client_select} ratio={args.client_select_ratio} "
        f"(prefer_higher_loss={args.prefer_higher_loss}) | "
        f"Modality select: top={args.num_select_modalities or 'all'} "
        f"random={args.random_modality}"
    )
    print("Finished initializing in {:.2f} seconds".format(timer()))

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "upload_bytes": [],
        "download_bytes": [],
        "client_selected": [],
        "modality_selected": [],
        "lr": [],
    }

    k_clients = max(1, int(round(args.client_select_ratio * n_clients)))
    k_clients = min(k_clients, n_clients)
    download_state = _init_download_state(
        args, n_clients, k_clients, grad_size, global_vec, device
    )

    for epoch in range(args.num_epochs):
        lr = float(lr_schedule(epoch + 1))
        model.train()
        set_param_vec(model, global_vec)

        # ---- who to score for selection (same scheme as v1) ----
        if args.client_select == "random":
            active = rng.choice(n_clients, size=k_clients, replace=False)
            train_ids = np.sort(active)
        else:
            train_ids = np.arange(n_clients)

        losses = np.full((n_clients,), np.nan)
        accs = np.full((n_clients,), np.nan)
        shap_rows = []
        shap_client_ids = []

        # Forward-only scoring on the global model (no SGD for unselected clients).
        for cid in train_ids:
            cid = int(cid)
            if len(clients[cid]["y"]) == 0:
                shap_rows.append(np.zeros(n_mod, dtype=np.float64))
                shap_client_ids.append(cid)
                continue
            loss, acc = evaluate_model(model, clients[cid], device)
            losses[cid] = loss
            accs[cid] = acc
            if len(clients[cid]["y"]) >= 2:
                sv = shap_from_fusion(
                    model, clients[cid]["xs"], clients[cid]["y"], modalities, device
                )
            else:
                sv = np.zeros(n_mod, dtype=np.float64)
            shap_rows.append(sv)
            shap_client_ids.append(cid)

        prev_losses = np.where(np.isnan(losses), prev_losses, losses)

        # ---- client selection for aggregation / upload ----
        if args.client_select == "random":
            selected = train_ids
        else:
            eligible = [i for i in range(n_clients) if not np.isnan(prev_losses[i])]
            if not eligible:
                eligible = list(range(n_clients))
            pri = compute_client_priority(
                epoch,
                prev_losses[eligible],
                client_last_selected[eligible],
                args.client_weights,
                prefer_higher_loss=args.prefer_higher_loss,
            )
            picked_local = select_clients(
                pri, min(k_clients, len(eligible)), False, rng
            )
            selected = np.array([eligible[i] for i in picked_local], dtype=int)

        selected = np.asarray(selected, dtype=int)
        client_last_selected[selected] = epoch
        n_participating = int(len(selected))

        # ---- modality priority (same functions as v1) ----
        shap_map = {cid: shap_rows[j] for j, cid in enumerate(shap_client_ids)}
        active_for_mod = [int(c) for c in selected if int(c) in shap_map]
        if not active_for_mod:
            active_for_mod = [int(c) for c in selected]

        shap_mat = np.stack(
            [shap_map.get(c, np.zeros(n_mod, dtype=np.float64)) for c in active_for_mod],
            axis=0,
        )
        if args.random_modality:
            shap_mat = rng.rand(*shap_mat.shape)
        shap_mat = normalization(shap_mat)
        size_mat = np.tile(branch_sizes, (len(active_for_mod), 1))
        size_mat = normalization(size_mat)
        recency = (epoch - recency_history[active_for_mod, :]) / (epoch + 1)
        recency = normalization(recency)
        Priority = compute_modality_priority(
            shap_mat,
            size_mat,
            recency,
            args.modality_weights,
            args.num_select_modalities,
        )
        mod_mask_active = Priority != -1
        rows, cols = np.where(mod_mask_active)
        recency_history[np.array(active_for_mod)[rows], cols] = epoch

        cid_to_mask = {
            cid: mod_mask_active[i] for i, cid in enumerate(active_for_mod)
        }
        default_mask = np.ones(n_mod, dtype=bool)
        if args.num_select_modalities > 0:
            default_mask[:] = False
            default_mask[: min(args.num_select_modalities, n_mod)] = True

        # IC/FetchSGD: bill download from current PS weights vs last-seen
        # (before this round's unsketch / weight update). Epoch 1 is 0.
        download_bytes = _download_bytes_ic(download_state, selected, global_vec)

        # ---- FetchSGD: example-weighted sketches from selected clients ----
        sketch_sum = torch.zeros(args.num_rows, args.num_cols, device=device)
        n_examples = 0
        n_upload = 0
        client_sel_row = np.zeros(n_clients, dtype=np.int8)
        mod_sel_row = np.zeros((n_clients, n_mod), dtype=np.int8)

        for cid in selected:
            cid = int(cid)
            client_sel_row[cid] = 1
            n_i = int(len(clients[cid]["y"]))
            if n_i == 0:
                continue
            mask = cid_to_mask.get(cid, default_mask)
            mod_sel_row[cid] = mask.astype(np.int8)
            vec, loss_v, acc_v = _client_upload_vec(
                model,
                clients[cid],
                global_vec,
                args.local_epochs,
                args.local_batch_size,
                lr,
                args.weight_decay,
                n_participating,
                upload_object,
                device,
            )
            if vec is None:
                continue
            if not np.isnan(loss_v):
                losses[cid] = loss_v
            if not np.isnan(acc_v):
                accs[cid] = acc_v
            vec = _mask_coords(vec, mod_indices, shared_indices, mask, device)
            table = sketch_vector(
                vec, grad_size, args.num_rows, args.num_cols, device
            )
            sketch_sum = sketch_sum + table * float(n_i)
            n_examples += n_i
            n_upload += 1

        set_param_vec(model, global_vec)

        if n_upload > 0 and n_examples > 0:
            avg_sketch = sketch_sum / float(n_examples)
            server_lr = 1.0 if upload_object == "delta" else lr
            update, Vvelocity, Verror = fetchsgd_server_step(
                avg_sketch,
                Vvelocity,
                Verror,
                grad_size,
                args.num_rows,
                args.num_cols,
                min(args.k, grad_size),
                args.virtual_momentum,
                server_lr,
                device,
                error_type=args.error_type,
            )
            global_vec = global_vec - update
        set_param_vec(model, global_vec)

        upload_bytes = n_upload * upload_per_client

        train_ids_eval = selected if len(selected) else train_ids
        train_loss_vals = [losses[i] for i in train_ids_eval if not np.isnan(losses[i])]
        train_acc_vals = [accs[i] for i in train_ids_eval if not np.isnan(accs[i])]
        train_loss = float(np.mean(train_loss_vals)) if train_loss_vals else float("nan")
        train_acc = float(np.mean(train_acc_vals)) if train_acc_vals else float("nan")

        test_loss, test_acc = evaluate_model_ic_shards(
            model,
            global_test,
            device,
            shard_size=getattr(args, "valid_batch_size", 8),
        )

        timer()
        download_mb = download_bytes / (1000 * 1000)
        upload_mb = upload_bytes / (1000 * 1000)
        total_download += download_mb
        total_upload += upload_mb

        logger.append(
            {
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "down (MB)": download_mb,
                "up (MB)": upload_mb,
                "total_time": timer.total_time,
            }
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["upload_bytes"].append(upload_bytes)
        history["download_bytes"].append(download_bytes)
        history["client_selected"].append(client_sel_row)
        history["modality_selected"].append(mod_sel_row)
        history["lr"].append(lr)

    print("Total Download (MiB): {:0.2f}".format(total_download))
    print("Total Upload (MiB): {:0.2f}".format(total_upload))
    print(
        "Avg Download Per Client: {:0.2f}".format(total_download / max(n_clients, 1))
    )
    print(
        "Avg Upload Per Client: {:0.2f}".format(total_upload / max(n_clients, 1))
    )
    print(
        "Cumulative train comm (download+upload): {:.4f} MiB".format(
            total_download + total_upload
        )
    )

    history["client_selected"] = np.stack(history["client_selected"], axis=0)
    history["modality_selected"] = np.stack(history["modality_selected"], axis=0)
    for key in (
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "upload_bytes",
        "download_bytes",
        "lr",
    ):
        history[key] = np.asarray(history[key])
    history["total_download_mib"] = total_download
    history["total_upload_mib"] = total_upload
    history["upload_object"] = upload_object
    return history
