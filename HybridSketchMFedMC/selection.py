"""MFedMC-style modality priority (SHAP + size + recency) and client selection."""

from __future__ import annotations

import numpy as np
import torch

from utils import get_aligned_shap_values, normalization, normalize_1d

try:
    import shap
    from sklearn.ensemble import RandomForestClassifier

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


def _stack_shap(shap_value):
    if isinstance(shap_value, list):
        shap_value = np.stack(shap_value, axis=-1)
    shap_value = np.asarray(shap_value)
    if shap_value.ndim == 2:
        return np.sum(np.abs(shap_value), axis=0)
    if shap_value.ndim == 3:
        return np.sum(np.abs(shap_value), axis=(0, 2))
    return np.abs(shap_value).reshape(-1)


@torch.no_grad()
def per_modality_preds(model, xs, device):
    preds = []
    for i, x in enumerate(xs):
        t = torch.as_tensor(x, dtype=torch.float32, device=device)
        logits = model.forward_single_modality(i, t)
        preds.append(logits.argmax(dim=1).cpu().numpy().reshape(-1, 1))
    return np.hstack(preds)


def shap_from_fusion(model, xs, y, modalities, device):
    n = len(y)
    n_mod = len(modalities)
    if n < 2:
        return np.zeros(n_mod, dtype=np.float64)

    fusion_input = per_modality_preds(model, xs, device)
    target = np.asarray(y)

    if _HAS_SHAP:
        try:
            fusion_module = RandomForestClassifier(n_estimators=10, random_state=0).fit(
                fusion_input, target
            )
            ns = min(50, len(fusion_input))
            sampled = shap.sample(fusion_input, ns)
            explainer = shap.TreeExplainer(fusion_module, sampled)
            shap_value = _stack_shap(explainer.shap_values(sampled))
            return get_aligned_shap_values(shap_value, list(modalities), list(modalities))
        except Exception:
            pass

    # Fallback: per-modality accuracy as importance (no shap / sklearn).
    scores = np.zeros(n_mod, dtype=np.float64)
    for i in range(n_mod):
        scores[i] = float((fusion_input[:, i] == target).mean())
    return scores


@torch.no_grad()
def per_modality_accuracy(model, xs, y, device):
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    accs = []
    for i, x in enumerate(xs):
        t = torch.as_tensor(x, dtype=torch.float32, device=device)
        pred = model.forward_single_modality(i, t).argmax(dim=1)
        accs.append(float((pred == y_t).float().mean().item()))
    return accs


def compute_modality_priority(shap_value, model_size, recency, modality_weights, top_shap):
    Priority = (
        modality_weights[0] * shap_value
        + modality_weights[1] * (1 - model_size)
        + modality_weights[2] * recency
    )
    top_shap = int(top_shap)
    if top_shap <= 0 or top_shap >= Priority.shape[1]:
        return Priority
    top_indices = np.argsort(Priority, axis=1)[:, :-top_shap]
    Priority[np.arange(Priority.shape[0])[:, None], top_indices] = -1
    return Priority


def compute_client_priority(
    ite,
    client_losses,
    client_last_selected_round,
    client_weights,
    prefer_higher_loss=False,
):
    loss = np.asarray(client_losses, dtype=float)
    if np.all(np.isnan(loss)):
        loss = np.zeros_like(loss)
    else:
        maxv = np.nanmax(loss)
        loss[np.isnan(loss)] = maxv
    norm_loss = normalize_1d(loss)
    staleness = (ite - client_last_selected_round) / (ite + 1)
    norm_staleness = normalize_1d(staleness)
    loss_term = norm_loss if prefer_higher_loss else (1.0 - norm_loss)
    w0 = client_weights[0] if len(client_weights) > 0 else 1.0
    w1 = client_weights[1] if len(client_weights) > 1 else 0.0
    return w0 * loss_term + w1 * norm_staleness


def select_clients(priority, k, random_clients, rng):
    n = len(priority)
    k = max(1, min(int(k), n))
    if random_clients:
        return rng.choice(n, size=k, replace=False)
    return np.argsort(-priority)[:k]


def select_modalities_for_clients(priority_matrix):
    """priority_matrix: (n_active, n_mod); -1 means dropped."""
    return priority_matrix != -1
