"""Load the FedMultiModal4 mHealth split into MFedMC client dicts.

Expects the same folder as ``run_mhealth_sketch_fusion_B_4mod.sh``:
  datasets/mhealth_mm/data.npz
  optional cached client0.npz … client{N-1}.npz + test.npz  (FedMultiModal4)

Modalities (see prepare_mhealth_mm.py):
  Acc  — 177-D accelerometer features, chest+ankle+arm (m0_* in data.npz)
  Gyro — 118-D gyroscope features, ankle+arm (m1_* in data.npz)
  Mag  — 118-D magnetometer features, ankle+arm (m2_* in data.npz)
  ECG  — 43-D chest 2-lead ECG features (m3_* in data.npz)

Only Dirichlet-10-client partitioning is supported for now (reusing the
FedMultiModal4 cache directly, mirroring ``MFedMC/UCI_HAR/dataset.py``'s
``load_from_sketchfusion_dir``); subject-based partitioning (one client per
mHealth subject) is a possible follow-up, analogous to
``load_subject_clients`` there.
"""

from __future__ import annotations

import json
import os

import numpy as np

MODALITIES = ["Acc", "Gyro", "Mag", "ECG"]


def _as_int_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        y = y.argmax(axis=1)
    return y.astype(np.int64).ravel()


def _pack_client(mods: list[np.ndarray], labels: np.ndarray) -> dict:
    labels = _as_int_labels(labels)
    n = len(labels)
    y = [int(labels[i]) for i in range(n)]
    packed = {}
    for name, arr in zip(MODALITIES, mods):
        arr = np.asarray(arr, dtype=np.float32)
        packed[name] = [[arr[i] for i in range(n)], list(y)]
    return packed


def dirichlet_indices(labels: np.ndarray, n_clients: int, alpha: float, seed: int = 42):
    """Match CommEfficient FedMultiModal4.prepare_datasets (RandomState + Dirichlet)."""
    labels = _as_int_labels(labels)
    rng = np.random.RandomState(seed)
    num_classes = int(labels.max()) + 1
    label_dist = rng.dirichlet([alpha] * n_clients, num_classes)
    client_indices = [[] for _ in range(n_clients)]
    for i in range(len(labels)):
        dc = int(labels[i])
        cid = int(rng.choice(n_clients, p=label_dist[dc]))
        client_indices[cid].append(i)
    return client_indices


def _cache_matches(dataset_dir: str, num_clients: int) -> bool:
    stats_path = os.path.join(dataset_dir, "stats.json")
    test_path = os.path.join(dataset_dir, "test.npz")
    if not os.path.isfile(test_path):
        return False
    for i in range(num_clients):
        if not os.path.isfile(os.path.join(dataset_dir, f"client{i}.npz")):
            return False
    if os.path.isfile(stats_path):
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        cached_n = len(stats.get("images_per_client", []))
        if cached_n != num_clients:
            return False
    return True


def load_mhealth_dir(
    dataset_dir: str,
    num_clients: int = 10,
    dirichlet_alpha: float = 0.1,
    seed: int = 42,
):
    """Return (client_data, global_test, meta).

    client_data maps client id → {Acc, Gyro, Mag, ECG: (samples, labels)}.
    global_test is the official mHealth test split (same as SketchFusionB4).
    """
    dataset_dir = os.path.abspath(dataset_dir)
    data_npz = os.path.join(dataset_dir, "data.npz")
    if not os.path.isfile(data_npz):
        raise FileNotFoundError(
            f"Missing {data_npz}. Build it with "
            "CommEfficient/CommEfficient/prepare_mhealth_mm.py "
            "(same as run_mhealth_sketch_fusion_B_4mod.sh)."
        )

    raw = np.load(data_npz)
    mods_tr = [raw[f"m{k}_train"] for k in range(len(MODALITIES))]
    mods_te = [raw[f"m{k}_test"] for k in range(len(MODALITIES))]
    y_tr = _as_int_labels(raw["labels_train"])
    y_te = _as_int_labels(raw["labels_test"])
    global_test = _pack_client(mods_te, y_te)

    reused_cache = False
    if _cache_matches(dataset_dir, num_clients):
        print(
            f"Reusing FedMultiModal4 client cache in {dataset_dir} "
            f"({num_clients} clients)."
        )
        client_data = {}
        n_per = []
        for i in range(num_clients):
            d = np.load(os.path.join(dataset_dir, f"client{i}.npz"))
            mods_i = [d[f"m{k}_feats"] for k in range(len(MODALITIES))]
            packed = _pack_client(mods_i, d["labels"])
            client_data[f"C{i:02d}"] = packed
            n_per.append(len(packed[MODALITIES[0]][1]))
        reused_cache = True
    else:
        print(
            f"Partitioning data.npz with Dirichlet alpha={dirichlet_alpha}, "
            f"seed={seed}, n_clients={num_clients} "
            "(same algorithm as FedMultiModal4; not writing client*.npz)."
        )
        idxs = dirichlet_indices(y_tr, num_clients, dirichlet_alpha, seed=seed)
        client_data = {}
        n_per = []
        for i, idx in enumerate(idxs):
            idx = np.asarray(idx, dtype=np.int64)
            if len(idx) == 0:
                empty_mods = [
                    np.empty((0, mods_tr[k].shape[1]), dtype=np.float32)
                    for k in range(len(MODALITIES))
                ]
                packed = _pack_client(empty_mods, np.empty((0,), dtype=np.int64))
            else:
                packed = _pack_client([m[idx] for m in mods_tr], y_tr[idx])
            client_data[f"C{i:02d}"] = packed
            n_per.append(len(packed[MODALITIES[0]][1]))

    mod_dims = [int(m.shape[1]) for m in mods_tr]
    print(
        f"  mod_dims={dict(zip(MODALITIES, mod_dims))}, "
        f"classes={int(y_tr.max()) + 1}, train={len(y_tr)}, test={len(y_te)}"
    )
    print(f"  samples/client: {n_per} (min={min(n_per)}, max={max(n_per)})")

    meta = {
        "dataset_dir": dataset_dir,
        "reused_sketchfusion_cache": reused_cache,
        "num_clients": num_clients,
        "dirichlet_alpha": dirichlet_alpha,
        "mod_dims": mod_dims,
        "samples_per_client": n_per,
        "num_test": int(len(y_te)),
        "modalities": MODALITIES,
    }
    return client_data, global_test, meta


def dirichlet_partition_data(client_data, alpha, seed=42):
    """Optional extra label skew, same role as ActionSense ``class_non_iid_rate``."""
    from collections import defaultdict

    np.random.seed(seed)
    clients = list(client_data.keys())
    data_by_label = defaultdict(list)
    for client_id, streams in client_data.items():
        ref_key = list(streams.keys())[0]
        _, ref_labels = streams[ref_key]
        for i, label in enumerate(ref_labels):
            sample_data = {
                mapped_key: (datasets[i], labels[i])
                for mapped_key, (datasets, labels) in streams.items()
                if i < len(datasets)
            }
            data_by_label[label].append((client_id, sample_data))

    new_client_data = {
        client_id: {key: ([], []) for key in client_data[client_id].keys()}
        for client_id in clients
    }
    num_clients = len(clients)
    for label, label_data in data_by_label.items():
        if not label_data:
            continue
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        client_sample_counts = np.round(proportions * len(label_data)).astype(int)
        diff = len(label_data) - np.sum(client_sample_counts)
        if diff > 0:
            indices = np.random.choice(num_clients, int(diff), replace=False)
            for index in indices:
                client_sample_counts[index] += 1
        elif diff < 0:
            indices = np.random.choice(num_clients, int(-diff), replace=False)
            for index in indices:
                if client_sample_counts[index] > 0:
                    client_sample_counts[index] -= 1
        np.random.shuffle(label_data)
        start_idx = 0
        for i, client_id in enumerate(clients):
            count = client_sample_counts[i]
            if count > 0:
                end_idx = min(start_idx + count, len(label_data))
                for j in range(start_idx, end_idx):
                    for mapped_key, (dataset, data_label) in label_data[j][1].items():
                        if mapped_key in new_client_data[client_id]:
                            new_client_data[client_id][mapped_key][0].append(dataset)
                            new_client_data[client_id][mapped_key][1].append(data_label)
                start_idx = end_idx
    return new_client_data


def stratified_split_client_data(client_data, train_ratio=0.8, seed=42):
    """Per-client stratified 80/20, matching ActionSense ``main.py``."""
    rng = np.random.RandomState(seed)
    client_data_train = {}
    client_data_test = {}
    for client, modalities_data in client_data.items():
        client_data_train[client] = {}
        client_data_test[client] = {}
        ref_modality = list(modalities_data.keys())[0]
        _, y = modalities_data[ref_modality]
        y = np.asarray(y)
        if len(y) == 0 or train_ratio >= 1.0:
            for device_stream, data in modalities_data.items():
                client_data_train[client][device_stream] = (list(data[0]), list(data[1]))
                client_data_test[client][device_stream] = ([], [])
            continue
        train_indices = []
        test_indices = []
        for cls in np.unique(y):
            all_indices = np.where(y == cls)[0]
            rng.shuffle(all_indices)
            boundary = int(len(all_indices) * train_ratio)
            train_indices.extend(all_indices[:boundary].tolist())
            test_indices.extend(all_indices[boundary:].tolist())
        for device_stream, data in modalities_data.items():
            x = data[0]
            y_all = data[1]
            client_data_train[client][device_stream] = (
                [x[i] for i in train_indices],
                [y_all[i] for i in train_indices],
            )
            client_data_test[client][device_stream] = (
                [x[i] for i in test_indices],
                [y_all[i] for i in test_indices],
            )
    return client_data_train, client_data_test


def attach_global_test(client_ids, global_test):
    """Give every client the official mHealth test set (SketchFusionB4 eval protocol)."""
    out = {}
    for cid in client_ids:
        out[cid] = {
            m: (list(global_test[m][0]), list(global_test[m][1]))
            for m in global_test
        }
    return out
