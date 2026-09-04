import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import copy
import concurrent.futures as cf
from sklearn.ensemble import RandomForestClassifier
from utils import *

def _cuda_safe_pool_workers(requested):
    """Single GPU: concurrent CUDA from threads is unsafe and can segfault; keep one worker."""
    if torch.cuda.is_available():
        return 1
    return max(1, requested)

def train_client(client_data, global_model, local_epochs=5, device="cuda"):
    criterion = nn.NLLLoss()
    client_model = type(global_model)().to(device)
    client_model.load_state_dict(copy.deepcopy(global_model).state_dict())
    data, target = client_data
    if len(target) == 0:
        return client_model, np.nan
    optimizer = optim.SGD(client_model.parameters(), lr=0.1)
    data = torch.tensor(np.array(data)).float().to(device)
    target = torch.tensor(target).long().to(device)
    data_loader = torch.utils.data.DataLoader(list(zip(data, target)), batch_size=32, shuffle=True)

    for epoch in range(local_epochs):
        total_loss = 0
        for batch_data, batch_target in data_loader:
            optimizer.zero_grad()
            output = client_model(batch_data)
            loss = criterion(output, batch_target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(data_loader)
    return client_model, avg_loss

def test_client(client, client_data, global_models, fusion_module, modalities, device="cuda"):
    fusion_input = []
    modality_accuracies = []
    valid_modalities = modalities.copy()
    n = len(client_data[valid_modalities[0]][1])
    if n < 1 or fusion_module is None:
        return [float('nan')] * (len(modalities) + 1)
    for modality in valid_modalities:
        mod_idx = modalities.index(modality)
        with torch.no_grad():
            data, target = client_data[modality]
            data = torch.tensor(np.array(data)).float().to(device)
            target = torch.tensor(target).long().to(device)
            global_model = global_models[mod_idx].to(device)
            output = global_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            modality_accuracies.append(100. * correct / target.size(0))
            fusion_input.append(pred.cpu().numpy())
    fusion_input = np.hstack(fusion_input)
    fusion_accuracy = fusion_module.score(fusion_input, target.cpu().numpy()) * 100
    full_accuracies = []
    for modality in modalities:
        if modality in valid_modalities:
            full_accuracies.append(modality_accuracies.pop(0))
        else:
            full_accuracies.append(float('nan'))
    return full_accuracies + [fusion_accuracy]

def calculate_shapley_values_via_fusion(client, client_data, client_models, modalities, device="cuda"):
    valid_modalities = modalities.copy()
    _, target = client_data[valid_modalities[0]]
    if len(target) < 2:
        return np.zeros(len(modalities), dtype=np.float64)
    fusion_input = []
    for mod_idx, modality in enumerate(valid_modalities):
        with torch.no_grad():
            data, _ = client_data[modality]
            data = torch.tensor(np.array(data)).float().to(device)
            model = client_models[mod_idx].to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            fusion_input.append(pred.cpu().numpy())

    fusion_input = np.hstack(fusion_input)
    fusion_module = RandomForestClassifier(n_estimators=10).fit(fusion_input, target)

    fusion_input_sampled = shap.sample(fusion_input, min(50, len(fusion_input)))
    explainer = shap.TreeExplainer(fusion_module, fusion_input_sampled)
    shap_value = explainer.shap_values(fusion_input_sampled)
    if isinstance(shap_value, list):
        shap_value = np.stack(shap_value, axis=-1)
    shap_value = np.asarray(shap_value)
    if shap_value.ndim == 2:
        shap_value = np.sum(np.abs(shap_value), axis=0)
    else:
        shap_value = np.sum(np.abs(shap_value), axis=(0, 2))
    shap_value = get_aligned_shap_values(shap_value, valid_modalities, modalities)   
    return shap_value

def train_local_fusion_module(client, client_data, trained_global_models, modalities, device="cuda"):
    valid_modalities = modalities.copy()
    _, target = client_data[valid_modalities[0]]
    if len(target) < 2:
        return None, np.nan
    fusion_input = []
    for modality in valid_modalities:
        with torch.no_grad():
            mod_idx = modalities.index(modality)
            data, _ = client_data[modality]
            data = torch.tensor(np.array(data)).float().to(device)
            model = trained_global_models[mod_idx].to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            fusion_input.append(pred.cpu().numpy())

    fusion_input = np.hstack(fusion_input)
    fusion_module = RandomForestClassifier(n_estimators=10).fit(fusion_input, target)
    accuracy = fusion_module.score(fusion_input, target) * 100
    return fusion_module, accuracy

def compute_modality_priority(shap_value, model_size, recency, modality_weights, top_shap):
    Priority = modality_weights[0]*shap_value + modality_weights[1]*(1-model_size) + modality_weights[2]*recency
    top_indices = np.argsort(Priority, axis=1)[:, :-top_shap]
    Priority[np.arange(Priority.shape[0])[:, None], top_indices] = -1
    return Priority

def compute_client_priority(ite, client_losses_matrix, client_last_selected_round, client_weights, prefer_higher_loss=True):
    client_loss_mean = np.nanmean(client_losses_matrix, axis=1)
    if np.all(np.isnan(client_loss_mean)):
        client_loss_mean = np.zeros_like(client_loss_mean)
    else:
        maxv = np.nanmax(client_loss_mean)
        client_loss_mean[np.isnan(client_loss_mean)] = maxv
    norm_loss = normalize_1d(client_loss_mean)
    staleness = (ite - client_last_selected_round) / (ite + 1)
    norm_staleness = normalize_1d(staleness)
    if prefer_higher_loss:
        loss_term = norm_loss
    else:
        loss_term = 1 - norm_loss
    client_priority = client_weights[0]*loss_term + client_weights[1]*norm_staleness
    return client_priority

def run_local_training_round(client_list, global_models, client_data_train, modalities, local_epochs):
    global_models_copy = [copy.deepcopy(model) for model in global_models]
    client_models = [[] for _ in modalities]
    client_idx_per_mod = [[] for _ in modalities]
    client_loss = [[] for _ in modalities]
    client_par = [[] for _ in modalities]
    shap_values = []
    
    for cidx, client in enumerate(client_list):
        local_models = []
        futs = []
        with cf.ThreadPoolExecutor(max_workers=_cuda_safe_pool_workers(len(modalities))) as ex:
            for mod_idx, modality in enumerate(modalities):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                futs.append(ex.submit(
                    lambda mi, md, dv: train_client(client_data_train[client][md], global_models_copy[mi], local_epochs, device=dv),
                    mod_idx, modality, device
                ))
            idx = 0
            for mod_idx, modality in enumerate(modalities):
                client_model, avg_loss = futs[idx].result()
                idx += 1
                local_models.append(client_model)
                client_models[mod_idx].append(client_model.state_dict())
                client_idx_per_mod[mod_idx].append(cidx)
                client_par[mod_idx].append(count_parameters(client_model))
                client_loss[mod_idx].append(avg_loss)
                
        device = "cuda" if torch.cuda.is_available() else "cpu"
        shap_value = calculate_shapley_values_via_fusion(client, client_data_train[client], local_models, modalities, device=device)
        shap_values.append(shap_value)
        
    shap_values = normalization(np.array(shap_values))
    model_size = normalization(np.array(client_par).T)
    return client_models, client_idx_per_mod, client_loss, shap_values, model_size

def select_and_aggregate(args, ite, modalities, shap_vals, model_size, recency_history, client_loss, client_last_selected_round, client_models, client_idx_per_mod, global_models, active_indices):
    modality_recency = (ite - recency_history) / (ite + 1)
    modality_recency = modality_recency[active_indices, :]
    if args.random_modality:
        shap_vals = np.random.rand(*shap_vals.shape)
    Priority = compute_modality_priority(shap_vals, model_size, modality_recency, args.modality_weights, args.top_shap)
    rows, cols = np.where(Priority != -1)
    recency_history[np.array(active_indices)[rows], cols] = ite
    client_losses_matrix = np.array(client_loss).T
    clsr_active = client_last_selected_round[active_indices]
    
    client_priority = compute_client_priority(ite, client_losses_matrix, clsr_active, args.client_weights)
    total_clients = recency_history.shape[0] 
    k = max(1, round(args.client_select_ratio * total_clients))
    k = min(k, len(active_indices))

    if args.random_clients:
        selected_active = np.random.choice(np.arange(len(active_indices)), size=k, replace=False).tolist()
    else:
        selected_active = np.argsort(-client_priority)[:k]
    selected_clients = [active_indices[i] for i in selected_active]
    client_last_selected_round[selected_clients] = ite

    counts = []
    per_mod_selected = []
    for midx, mod in enumerate(modalities):
        allowed = set(np.where(Priority.T[midx] != -1)[0].tolist())
        picked_clients = [int(i) for i in selected_active if i in allowed]
        per_mod_selected.append(picked_clients)
        picked_clients_global = [active_indices[i] for i in picked_clients]
        mapping = {c_idx: w_idx for w_idx, c_idx in enumerate(client_idx_per_mod[midx])}
        chosen_weight_indices = [mapping[c] for c in picked_clients_global if c in mapping]
        gw, cnt = average_weights(client_models[midx], chosen_weight_indices, global_models[midx])
        global_models[midx].load_state_dict(gw)
        counts.append(cnt)
    return np.array(counts)

def _pack_to_tensors(packed, device):
    data, target = packed
    x = torch.tensor(np.array(data)).float().to(device)
    y = torch.tensor(target).long().to(device)
    return x, y


def evaluate_official_test(global_models, global_test, modalities, device):
    """One global predictor on the official test set (same protocol as Hybrid / Independent).

    Acc and Gyro encoders are scored once. Fusion is late fusion: sum of log-softmax
    outputs, then argmax. ``test_acc`` is a fraction in [0, 1].
    """
    logps = []
    y = None
    mod_acc = []
    with torch.no_grad():
        for mi, modality in enumerate(modalities):
            x, y = _pack_to_tensors(global_test[modality], device)
            logp = global_models[mi].to(device)(x)
            pred = logp.argmax(dim=1)
            mod_acc.append(float((pred == y).float().mean().item()))
            logps.append(logp)
    fused = torch.stack(logps, dim=0).sum(dim=0)
    test_acc = float((fused.argmax(dim=1) == y).float().mean().item())
    return test_acc, mod_acc


def evaluate_round(ite, client_list, client_data_train, client_data_test, global_models, modalities, device):
    fusion_modules = {}
    with cf.ThreadPoolExecutor(max_workers=_cuda_safe_pool_workers(len(client_list))) as ex:
        futs = {client: ex.submit(train_local_fusion_module, client, client_data_train[client], global_models, modalities, device=device) for client in client_list}
        for client in client_list:
            em, _ = futs[client].result()
            fusion_modules[client] = em
            
    with cf.ThreadPoolExecutor(max_workers=_cuda_safe_pool_workers(len(client_list))) as ex:
        futs2 = {client: ex.submit(test_client, client, client_data_test[client], global_models, fusion_modules[client], modalities, device=device) for client in client_list}
        accs = [futs2[client].result() for client in client_list]
        
    ca = np.array(accs, dtype=np.float64)
    mean_fusion = float(np.nanmean(ca[:, -1]))
    return accs, mean_fusion

def federated_learning(args, client_data_train, client_data_test, global_models, global_test, device):
    modalities = ['Acc', 'Gyro']
    num_clients = len(client_data_train)
    recency_history = np.full((num_clients, len(modalities)), -1)
    client_last_selected_round = np.full((num_clients,), -1)
    accuracy_matrix, modality_counts = [], []
    test_acc_rounds = []
    acc_test_rounds = []
    gyro_test_rounds = []
    
    for ite in range(args.iterations):
        client_list = list(client_data_train.keys())[:num_clients]
        active_indices = [client_list.index(c) for c in client_list]
            
        client_models, client_idx_per_mod, client_loss, shap_values, model_size = run_local_training_round(
            client_list, global_models, client_data_train, modalities, args.local_epochs
        )
        counts = select_and_aggregate(
            args, ite, modalities, shap_values, model_size, recency_history, 
            client_loss, client_last_selected_round, client_models, client_idx_per_mod, 
            global_models, active_indices
        )
        modality_counts.append(counts)
        accs, mean_fusion = evaluate_round(
            ite, client_list, client_data_train, client_data_test, global_models, modalities, device
        )
        accuracy_matrix.append(accs)
        test_acc, mod_acc = evaluate_official_test(global_models, global_test, modalities, device)
        test_acc_rounds.append(test_acc)
        acc_test_rounds.append(mod_acc[0])
        gyro_test_rounds.append(mod_acc[1])
        print(
            f"Iteration {ite} - test_acc: {test_acc:.4f} ({100.0 * test_acc:.2f}%) | "
            f"Acc {100.0 * mod_acc[0]:.2f}% | Gyro {100.0 * mod_acc[1]:.2f}% | "
            f"fusion_mean {mean_fusion:.2f}%"
        )
        
    return (
        np.array(accuracy_matrix),
        np.array(modality_counts),
        np.array(test_acc_rounds, dtype=np.float64),
        np.array(acc_test_rounds, dtype=np.float64),
        np.array(gyro_test_rounds, dtype=np.float64),
    )
