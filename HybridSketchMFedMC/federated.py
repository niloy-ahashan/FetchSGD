"""Federated loop: SketchFusionB local training + FetchSGD sketch upload + MFedMC selection."""

from __future__ import annotations

import copy

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
    get_param_vec,
    normalization,
    set_param_vec,
)

BYTES_PER_FLOAT32 = 4
_ce = nn.CrossEntropyLoss()


def _client_tensors(client, device):
    xs = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in client["xs"]]
    y = torch.as_tensor(client["y"], dtype=torch.long, device=device)
    return xs, y


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


def train_local(model, client, local_epochs, batch_size, lr, weight_decay, device):
    n = len(client["y"])
    if n == 0:
        return copy.deepcopy(model), float("nan"), float("nan")

    local = copy.deepcopy(model).to(device)
    local.train()
    opt = torch.optim.SGD(local.parameters(), lr=lr, weight_decay=weight_decay)
    loader, _ = _make_loader(client, batch_size, shuffle=True)

    last_loss = float("nan")
    last_acc = float("nan")
    for _ in range(local_epochs):
        total_loss = 0.0
        total_correct = 0
        total = 0
        n_batches = 0
        for batch in loader:
            mods, y = _split_batch(batch)
            mods = [m.to(device) for m in mods]
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = local(*mods)
            loss = _ce(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.size(0))
            n_batches += 1
        last_loss = total_loss / max(n_batches, 1)
        last_acc = total_correct / max(total, 1)
    return local, last_loss, last_acc


def _mask_delta(delta, mod_indices, shared_indices, selected_mask, device):
    """Keep shared coords + selected modality branches; zero the rest."""
    keep = torch.zeros_like(delta, dtype=torch.bool)
    keep[shared_indices.to(device)] = True
    for i, chosen in enumerate(selected_mask):
        if chosen:
            idx = mod_indices[i].to(device)
            if idx.numel() > 0:
                keep[idx] = True
    masked = delta.clone()
    masked[~keep] = 0
    return masked


def run_federated(args, model, clients, global_test, modalities, device):
    n_clients = len(clients)
    n_mod = len(modalities)
    rng = np.random.RandomState(args.seed)

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
    prev_update_nnz = grad_size
    updated_since_init = torch.zeros(grad_size, dtype=torch.bool, device=device)

    print(
        f"SketchFusionBNet config: fusion_mode={getattr(args, 'fusion_mode', 'sketch')} "
        f"mods={list(modalities)} "
        f"dims={[int(x.shape[1]) if x.ndim==2 else 0 for x in clients[0]['xs']]}"
    )
    print(f"Grad size: {grad_size}")
    print(f"Total params: {grad_size}")
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

    for epoch in range(args.num_epochs):
        lr = float(lr_schedule(epoch + 1))
        model.train()
        set_param_vec(model, global_vec)

        # ---- who to train locally ----
        if args.client_select == "random":
            active = rng.choice(n_clients, size=k_clients, replace=False)
            train_ids = np.sort(active)
        else:
            train_ids = np.arange(n_clients)

        local_models = {}
        losses = np.full((n_clients,), np.nan)
        accs = np.full((n_clients,), np.nan)
        shap_rows = []
        shap_client_ids = []

        for cid in train_ids:
            local, loss, acc = train_local(
                model,
                clients[cid],
                args.local_epochs,
                args.local_batch_size,
                lr,
                args.weight_decay,
                device,
            )
            local_models[int(cid)] = local
            losses[int(cid)] = loss
            accs[int(cid)] = acc
            if len(clients[cid]["y"]) >= 2:
                sv = shap_from_fusion(
                    local, clients[cid]["xs"], clients[cid]["y"], modalities, device
                )
            else:
                sv = np.zeros(n_mod, dtype=np.float64)
            shap_rows.append(sv)
            shap_client_ids.append(int(cid))

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

        # ---- modality priority on the clients we trained ----
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

        # ---- FetchSGD: sketch masked (global - local) deltas ----
        sketch_sum = torch.zeros(args.num_rows, args.num_cols, device=device)
        n_upload = 0
        client_sel_row = np.zeros(n_clients, dtype=np.int8)
        mod_sel_row = np.zeros((n_clients, n_mod), dtype=np.int8)

        for cid in selected:
            cid = int(cid)
            client_sel_row[cid] = 1
            local = local_models.get(cid)
            if local is None or len(clients[cid]["y"]) == 0:
                continue
            mask = cid_to_mask.get(cid, default_mask)
            mod_sel_row[cid] = mask.astype(np.int8)
            local_vec = get_param_vec(local).to(device)
            delta = global_vec - local_vec
            delta = _mask_delta(delta, mod_indices, shared_indices, mask, device)
            table = sketch_vector(
                delta, grad_size, args.num_rows, args.num_cols, device
            )
            sketch_sum = sketch_sum + table
            n_upload += 1
            del local
            local_models.pop(cid, None)

        local_models.clear()

        if n_upload > 0:
            avg_sketch = sketch_sum / n_upload
            update, Vvelocity, Verror = fetchsgd_server_step(
                avg_sketch,
                Vvelocity,
                Verror,
                grad_size,
                args.num_rows,
                args.num_cols,
                min(args.k, grad_size),
                args.virtual_momentum,
                1.0,
                device,
                error_type=args.error_type,
            )
            global_vec = global_vec - update
            nnz = int((update.abs() > 0).sum().item())
            updated_since_init |= update.abs() > 0
            prev_update_nnz = nnz
        else:
            nnz = 0

        set_param_vec(model, global_vec)

        upload_bytes = n_upload * upload_per_client
        download_bytes = int(client_sel_row.sum()) * prev_update_nnz * BYTES_PER_FLOAT32
        if epoch == 0:
            download_bytes = int(client_sel_row.sum()) * grad_size * BYTES_PER_FLOAT32

        train_ids_eval = selected if len(selected) else train_ids
        train_loss_vals = [losses[i] for i in train_ids_eval if not np.isnan(losses[i])]
        train_acc_vals = [accs[i] for i in train_ids_eval if not np.isnan(accs[i])]
        train_loss = float(np.mean(train_loss_vals)) if train_loss_vals else float("nan")
        train_acc = float(np.mean(train_acc_vals)) if train_acc_vals else float("nan")

        test_loss, test_acc = evaluate_model(model, global_test, device)

        timer()
        download_mb = download_bytes / (1000 * 1000)
        upload_mb = upload_bytes / (1000 * 1000)
        total_download += download_mb
        total_upload += upload_mb

        try:
            rounded_down = round(download_mb)
        except Exception:
            rounded_down = np.nan
        try:
            rounded_up = round(upload_mb)
        except Exception:
            rounded_up = np.nan

        logger.append(
            {
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "down (MiB)": rounded_down,
                "up (MiB)": rounded_up,
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
    return history
