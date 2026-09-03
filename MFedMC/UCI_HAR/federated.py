import copy
import concurrent.futures as cf
import time

import numpy as np
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier

from utils import (
    average_weights,
    count_parameters,
    get_aligned_shap_values,
    normalization,
    normalize_1d,
)

BYTES_PER_FLOAT32 = 4


def _cuda_safe_pool_workers(requested: int) -> int:
    if torch.cuda.is_available():
        return 1
    return max(1, requested)


def round_upload_bytes(global_models, counts_per_modality) -> int:
    total = 0
    for mi, n in enumerate(counts_per_modality):
        total += int(n) * count_parameters(global_models[mi]) * BYTES_PER_FLOAT32
    return total


def _to_tensor_xy(client_data, device):
    data, target = client_data
    if len(target) == 0:
        return None, None
    x = torch.tensor(np.array(data)).float().to(device)
    y = torch.tensor(target).long().to(device)
    return x, y


def train_client(client_data, global_model, local_epochs=5, lr=0.01, batch_size=32, device="cuda"):
    data, target = _to_tensor_xy(client_data, device)
    if data is None:
        return copy.deepcopy(global_model), np.nan

    client_model = copy.deepcopy(global_model).to(device)
    optimizer = optim.SGD(client_model.parameters(), lr=lr)
    n = data.size(0)
    bs = n if batch_size <= 0 else min(batch_size, n)
    loader = torch.utils.data.DataLoader(
        list(zip(data, target)), batch_size=bs, shuffle=True
    )
    criterion = nn.NLLLoss()

    avg_loss = np.nan
    for _ in range(local_epochs):
        total_loss = 0.0
        n_batches = 0
        for batch_data, batch_target in loader:
            optimizer.zero_grad()
            output = client_model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
    return client_model, avg_loss


def _stack_shap(shap_value):
    if isinstance(shap_value, list):
        shap_value = np.stack(shap_value, axis=-1)
    shap_value = np.asarray(shap_value)
    if shap_value.ndim == 2:
        return np.sum(np.abs(shap_value), axis=0)
    if shap_value.ndim == 3:
        return np.sum(np.abs(shap_value), axis=(0, 2))
    return np.abs(shap_value).reshape(-1)


def calculate_shapley_values_via_fusion(client_data, client_models, modalities, device="cuda"):
    n = len(client_data[modalities[0]][1])
    if n < 2:
        return np.zeros(len(modalities), dtype=np.float64)
    _, target = client_data[modalities[0]]
    fusion_input = []
    for mod_idx, modality in enumerate(modalities):
        with torch.no_grad():
            data, _ = _to_tensor_xy(client_data[modality], device)
            output = client_models[mod_idx].to(device)(data)
            pred = output.argmax(dim=1, keepdim=True)
            fusion_input.append(pred.cpu().numpy())
    fusion_input = np.hstack(fusion_input)
    try:
        fusion_module = RandomForestClassifier(n_estimators=10, random_state=0).fit(
            fusion_input, target
        )
        ns = min(50, len(fusion_input))
        fusion_input_sampled = shap.sample(fusion_input, ns)
        explainer = shap.TreeExplainer(fusion_module, fusion_input_sampled)
        shap_value = explainer.shap_values(fusion_input_sampled)
        shap_value = _stack_shap(shap_value)
        shap_value = get_aligned_shap_values(shap_value, list(modalities), list(modalities))
        return shap_value
    except Exception:
        return np.zeros(len(modalities), dtype=np.float64)


def train_local_fusion_module(client_data, trained_global_models, modalities, device="cuda"):
    n = len(client_data[modalities[0]][1])
    if n < 2:
        return None, np.nan
    _, target = client_data[modalities[0]]
    fusion_input = []
    for modality in modalities:
        with torch.no_grad():
            mod_idx = modalities.index(modality)
            data, _ = _to_tensor_xy(client_data[modality], device)
            if data is None:
                return None, np.nan
            output = trained_global_models[mod_idx].to(device)(data)
            pred = output.argmax(dim=1, keepdim=True)
            fusion_input.append(pred.cpu().numpy())
    fusion_input = np.hstack(fusion_input)
    try:
        fusion_module = RandomForestClassifier(n_estimators=10, random_state=0).fit(
            fusion_input, target
        )
        accuracy = fusion_module.score(fusion_input, target) * 100
        return fusion_module, accuracy
    except Exception:
        return None, np.nan


def test_client(client_data, global_models, fusion_module, modalities, device="cuda"):
    n = len(client_data[modalities[0]][1])
    nan_row = [float("nan")] * (len(modalities) + 1)
    if n < 1 or fusion_module is None:
        return nan_row
    fusion_input = []
    modality_accuracies = []
    target_np = None
    for modality in modalities:
        mod_idx = modalities.index(modality)
        with torch.no_grad():
            data, target = _to_tensor_xy(client_data[modality], device)
            if data is None:
                return nan_row
            target_np = target.cpu().numpy()
            output = global_models[mod_idx].to(device)(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            modality_accuracies.append(100.0 * correct / target.size(0))
            fusion_input.append(pred.cpu().numpy())
    fusion_input = np.hstack(fusion_input)
    try:
        fusion_accuracy = fusion_module.score(fusion_input, target_np) * 100
    except Exception:
        fusion_accuracy = float("nan")
    return modality_accuracies + [fusion_accuracy]


def compute_modality_priority(shap_value, model_size, recency, modality_weights, top_shap):
    # Same rule as ActionSense/federated.py: keep the `top_shap` highest-priority
    # modalities (set the rest to -1 so they are not uploaded).
    Priority = (
        modality_weights[0] * shap_value
        + modality_weights[1] * (1 - model_size)
        + modality_weights[2] * recency
    )
    top_shap = int(top_shap)
    if top_shap <= 0:
        return Priority
    top_indices = np.argsort(Priority, axis=1)[:, :-top_shap]
    Priority[np.arange(Priority.shape[0])[:, None], top_indices] = -1
    return Priority


def compute_client_priority(
    ite, client_losses_matrix, client_last_selected_round, client_weights, prefer_higher_loss=True
):
    client_loss_mean = np.nanmean(client_losses_matrix, axis=1)
    if np.all(np.isnan(client_loss_mean)):
        client_loss_mean = np.zeros_like(client_loss_mean)
    else:
        maxv = np.nanmax(client_loss_mean)
        client_loss_mean[np.isnan(client_loss_mean)] = maxv
    norm_loss = normalize_1d(client_loss_mean)
    staleness = (ite - client_last_selected_round) / (ite + 1)
    norm_staleness = normalize_1d(staleness)
    loss_term = norm_loss if prefer_higher_loss else (1 - norm_loss)
    return client_weights[0] * loss_term + client_weights[1] * norm_staleness


def run_local_training_round(
    client_list, global_models, client_data_train, modalities, local_epochs, lr, batch_size
):
    global_models_copy = [copy.deepcopy(model) for model in global_models]
    client_models = [[] for _ in modalities]
    client_idx_per_mod = [[] for _ in modalities]
    client_loss = [[] for _ in modalities]
    client_par = [[] for _ in modalities]
    shap_values = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for cidx, client in enumerate(client_list):
        local_models = []
        n_local = len(client_data_train[client][modalities[0]][1])
        if n_local == 0:
            for mod_idx in range(len(modalities)):
                client_par[mod_idx].append(count_parameters(global_models_copy[mod_idx]))
                client_loss[mod_idx].append(np.nan)
            shap_values.append(np.zeros(len(modalities), dtype=np.float64))
            continue

        futs = []
        with cf.ThreadPoolExecutor(max_workers=_cuda_safe_pool_workers(len(modalities))) as ex:
            for mod_idx, modality in enumerate(modalities):
                futs.append(
                    ex.submit(
                        train_client,
                        client_data_train[client][modality],
                        global_models_copy[mod_idx],
                        local_epochs,
                        lr,
                        batch_size,
                        device,
                    )
                )
            for mod_idx, fut in enumerate(futs):
                client_model, avg_loss = fut.result()
                local_models.append(client_model)
                client_models[mod_idx].append(client_model.state_dict())
                client_idx_per_mod[mod_idx].append(cidx)
                client_par[mod_idx].append(count_parameters(client_model))
                client_loss[mod_idx].append(avg_loss)

        shap_value = calculate_shapley_values_via_fusion(
            client_data_train[client], local_models, modalities, device=device
        )
        shap_values.append(shap_value)

    shap_values = normalization(np.array(shap_values))
    model_size = normalization(np.array(client_par).T)
    return client_models, client_idx_per_mod, client_loss, shap_values, model_size


def select_and_aggregate(
    args,
    ite,
    modalities,
    shap_vals,
    model_size,
    recency_history,
    client_loss,
    client_last_selected_round,
    client_models,
    client_idx_per_mod,
    global_models,
    active_indices,
):
    total_clients = recency_history.shape[0]
    n_mod = len(modalities)
    modality_recency = (ite - recency_history) / (ite + 1)
    modality_recency = modality_recency[active_indices, :]
    if args.random_modality:
        shap_vals = np.random.rand(*shap_vals.shape)
    Priority = compute_modality_priority(
        shap_vals, model_size, modality_recency, args.modality_weights, args.top_shap
    )
    rows, cols = np.where(Priority != -1)
    recency_history[np.array(active_indices)[rows], cols] = ite
    client_losses_matrix = np.array(client_loss).T
    clsr_active = client_last_selected_round[active_indices]

    client_priority = compute_client_priority(
        ite,
        client_losses_matrix,
        clsr_active,
        args.client_weights,
        prefer_higher_loss=args.prefer_higher_loss,
    )
    k = max(1, round(args.client_select_ratio * total_clients))
    k = min(k, len(active_indices))

    if args.random_clients:
        selected_active = np.random.choice(
            np.arange(len(active_indices)), size=k, replace=False
        ).tolist()
    else:
        selected_active = np.argsort(-client_priority)[:k]
    selected_clients = [active_indices[i] for i in selected_active]
    client_last_selected_round[selected_clients] = ite

    client_sel = np.zeros(total_clients, dtype=np.int8)
    client_sel[selected_clients] = 1
    mod_sel = np.zeros((total_clients, n_mod), dtype=np.int8)

    counts = []
    for midx, _mod in enumerate(modalities):
        allowed = set(np.where(Priority.T[midx] != -1)[0].tolist())
        picked_clients = [int(i) for i in selected_active if i in allowed]
        for i in picked_clients:
            mod_sel[active_indices[i], midx] = 1
        picked_clients_global = [active_indices[i] for i in picked_clients]
        mapping = {c_idx: w_idx for w_idx, c_idx in enumerate(client_idx_per_mod[midx])}
        chosen_weight_indices = [mapping[c] for c in picked_clients_global if c in mapping]
        gw, cnt = average_weights(client_models[midx], chosen_weight_indices, global_models[midx])
        global_models[midx].load_state_dict(gw)
        counts.append(cnt)
    return np.array(counts), client_sel, mod_sel


def evaluate_round(client_list, client_data_train, client_data_test, global_models, modalities, device):
    fusion_modules = {}
    with cf.ThreadPoolExecutor(max_workers=_cuda_safe_pool_workers(len(client_list))) as ex:
        futs = {
            client: ex.submit(
                train_local_fusion_module,
                client_data_train[client],
                global_models,
                modalities,
                device,
            )
            for client in client_list
        }
        for client in client_list:
            em, _ = futs[client].result()
            fusion_modules[client] = em

    with cf.ThreadPoolExecutor(max_workers=_cuda_safe_pool_workers(len(client_list))) as ex:
        futs2 = {
            client: ex.submit(
                test_client,
                client_data_test[client],
                global_models,
                fusion_modules[client],
                modalities,
                device,
            )
            for client in client_list
        }
        accs = [futs2[client].result() for client in client_list]
    return np.array(accs, dtype=np.float64)


def federated_learning(args, client_data_train, client_data_test, global_models, modalities, device):
    num_clients = len(client_data_train)
    recency_history = np.full((num_clients, len(modalities)), -1)
    client_last_selected_round = np.full((num_clients,), -1)
    accuracy_matrix, modality_counts = [], []
    upload_bytes_per_round: list[int] = []
    elapsed_seconds_per_round: list[float] = []
    client_selected_rounds = []
    modality_selected_rounds = []

    mod_acc_hdr = "  ".join(f"{m:>8}" for m in modalities)
    header = (
        f"{'iter':>4}  {'fusion':>8}  {mod_acc_hdr}  "
        f"{'up_MB':>10}  {'cum_up_MB':>10}  {'time_s':>10}"
    )
    train_start = time.perf_counter()
    print(header)
    print("-" * len(header))

    for ite in range(args.iterations):
        client_list = list(client_data_train.keys())[:num_clients]
        active_indices = list(range(len(client_list)))

        client_models, client_idx_per_mod, client_loss, shap_values, model_size = (
            run_local_training_round(
                client_list,
                global_models,
                client_data_train,
                modalities,
                args.local_epochs,
                args.lr,
                args.batch_size,
            )
        )
        counts, client_sel, mod_sel = select_and_aggregate(
            args,
            ite,
            modalities,
            shap_values,
            model_size,
            recency_history,
            client_loss,
            client_last_selected_round,
            client_models,
            client_idx_per_mod,
            global_models,
            active_indices,
        )
        modality_counts.append(counts)
        client_selected_rounds.append(client_sel)
        modality_selected_rounds.append(mod_sel)
        ub_round = round_upload_bytes(global_models, counts)
        upload_bytes_per_round.append(ub_round)
        accs = evaluate_round(
            client_list, client_data_train, client_data_test, global_models, modalities, device
        )
        accuracy_matrix.append(accs)
        mean_fusion = float(np.nanmean(accs[:, -1]))
        mean_mod = np.nanmean(accs[:, :-1], axis=0)
        cum_up = int(np.sum(upload_bytes_per_round))
        elapsed_s = time.perf_counter() - train_start
        elapsed_seconds_per_round.append(elapsed_s)
        mod_acc = "  ".join(f"{a:8.2f}" for a in mean_mod)
        print(
            f"{ite + 1:4d}  {mean_fusion:8.2f}  {mod_acc}  "
            f"{ub_round / 1e6:10.6f}  {cum_up / 1e6:10.6f}  {elapsed_s:10.1f}"
        )

    acc = np.array(accuracy_matrix)
    mod_counts = np.array(modality_counts)
    client_selected = np.stack(client_selected_rounds, axis=0)
    modality_selected = np.stack(modality_selected_rounds, axis=0)
    upload_bytes_round = np.array(upload_bytes_per_round, dtype=np.int64)
    elapsed_seconds_round = np.array(elapsed_seconds_per_round, dtype=np.float64)
    return (
        acc,
        mod_counts,
        upload_bytes_round,
        client_selected,
        modality_selected,
        elapsed_seconds_round,
    )
