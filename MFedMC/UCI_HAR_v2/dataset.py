"""Load the SketchFusionB UCI HAR multimodal split into MFedMC client dicts.

Expects the same folder as ``run_uci_har_sketch_fusion_B.sh``:
  datasets/uci_har_mm/data.npz
  optional cached client0.npz … client{N-1}.npz + test.npz  (FedMultiModal)

Modalities:
  Acc  — 348-D accelerometer-related engineered features (img_* in data.npz)
  Gyro — 213-D gyroscope-related engineered features (txt_* in data.npz)
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

MODALITIES = ["Acc", "Gyro"]

_DEFAULT_UCI_ROOTS = (
    "human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset",
    "MFedMC/UCI_HAR/data/UCI HAR Dataset",
)


def _as_int_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        y = y.argmax(axis=1)
    return y.astype(np.int64).ravel()


def _pack_client(acc: np.ndarray, gyro: np.ndarray, labels: np.ndarray) -> dict:
    labels = _as_int_labels(labels)
    n = len(labels)
    acc = np.asarray(acc, dtype=np.float32)
    gyro = np.asarray(gyro, dtype=np.float32)
    y = [int(labels[i]) for i in range(n)]
    return {
        "Acc": [[acc[i] for i in range(n)], list(y)],
        "Gyro": [[gyro[i] for i in range(n)], list(y)],
    }


def dirichlet_indices(labels: np.ndarray, n_clients: int, alpha: float, seed: int = 42):
    """Match CommEfficient FedMultiModal.prepare_datasets (RandomState + Dirichlet)."""
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


def load_from_sketchfusion_dir(
    dataset_dir: str,
    num_clients: int = 10,
    dirichlet_alpha: float = 0.1,
    seed: int = 42,
):
    """Return (client_data, global_test, meta).

    client_data maps client id → {Acc, Gyro: (samples, labels)}.
    global_test is the official UCI HAR test split (same as SketchFusionB).
    """
    dataset_dir = os.path.abspath(dataset_dir)
    data_npz = os.path.join(dataset_dir, "data.npz")
    if not os.path.isfile(data_npz):
        raise FileNotFoundError(
            f"Missing {data_npz}. Build it with "
            "CommEfficient/CommEfficient/prepare_uci_har_mm.py "
            "(same as run_uci_har_sketch_fusion_B.sh)."
        )

    raw = np.load(data_npz)
    img_tr, txt_tr, y_tr = raw["img_train"], raw["txt_train"], _as_int_labels(raw["labels_train"])
    img_te, txt_te, y_te = raw["img_test"], raw["txt_test"], _as_int_labels(raw["labels_test"])
    global_test = _pack_client(img_te, txt_te, y_te)

    reused_cache = False
    if _cache_matches(dataset_dir, num_clients):
        print(
            f"Reusing SketchFusionB client cache in {dataset_dir} "
            f"({num_clients} clients)."
        )
        client_data = {}
        n_per = []
        for i in range(num_clients):
            d = np.load(os.path.join(dataset_dir, f"client{i}.npz"))
            packed = _pack_client(d["img_feats"], d["txt_feats"], d["labels"])
            client_data[f"C{i:02d}"] = packed
            n_per.append(len(packed["Acc"][1]))
        reused_cache = True
    else:
        print(
            f"Partitioning data.npz with Dirichlet alpha={dirichlet_alpha}, "
            f"seed={seed}, n_clients={num_clients} "
            "(same algorithm as FedMultiModal; not writing client*.npz)."
        )
        idxs = dirichlet_indices(y_tr, num_clients, dirichlet_alpha, seed=seed)
        client_data = {}
        n_per = []
        for i, idx in enumerate(idxs):
            idx = np.asarray(idx, dtype=np.int64)
            if len(idx) == 0:
                packed = _pack_client(
                    np.empty((0, img_tr.shape[1]), dtype=np.float32),
                    np.empty((0, txt_tr.shape[1]), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                )
            else:
                packed = _pack_client(img_tr[idx], txt_tr[idx], y_tr[idx])
            client_data[f"C{i:02d}"] = packed
            n_per.append(len(packed["Acc"][1]))

    print(
        f"  Acc dim={img_tr.shape[1]}, Gyro dim={txt_tr.shape[1]}, "
        f"classes={int(y_tr.max()) + 1}, train={len(y_tr)}, test={len(y_te)}"
    )
    print(f"  samples/client: {n_per} (min={min(n_per)}, max={max(n_per)})")

    meta = {
        "dataset_dir": dataset_dir,
        "reused_sketchfusion_cache": reused_cache,
        "num_clients": num_clients,
        "dirichlet_alpha": dirichlet_alpha,
        "acc_dim": int(img_tr.shape[1]),
        "gyro_dim": int(txt_tr.shape[1]),
        "samples_per_client": n_per,
        "num_test": int(len(y_te)),
        "modalities": MODALITIES,
    }
    return client_data, global_test, meta


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def find_uci_root(uci_root: str | None = None) -> str:
    candidates = []
    if uci_root:
        candidates.append(uci_root)
    env = os.environ.get("UCI_ROOT")
    if env:
        candidates.append(env)
    repo = _repo_root()
    candidates.extend(os.path.join(repo, rel) for rel in _DEFAULT_UCI_ROOTS)
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.isfile(os.path.join(path, "train", "subject_train.txt")):
            return path
    raise FileNotFoundError(
        "Need UCI HAR train/subject_train.txt for subject-as-client partitioning. "
        "Pass --uci_root to the inner 'UCI HAR Dataset' folder."
    )


def load_subject_clients(dataset_dir: str, uci_root: str | None = None):
    """ActionSense-style clients: one client per UCI HAR subject, all 6 activities.

    Uses the same Acc/Gyro vectors as SketchFusionB ``data.npz``, grouped by
    ``subject_*.txt`` (train+test concatenated, then split 80/20 locally).
    """
    dataset_dir = os.path.abspath(dataset_dir)
    data_npz = os.path.join(dataset_dir, "data.npz")
    if not os.path.isfile(data_npz):
        raise FileNotFoundError(
            f"Missing {data_npz}. Build it with "
            "CommEfficient/CommEfficient/prepare_uci_har_mm.py."
        )
    root = find_uci_root(uci_root)
    raw = np.load(data_npz)
    img = np.concatenate([raw["img_train"], raw["img_test"]], axis=0).astype(np.float32)
    gyro = np.concatenate([raw["txt_train"], raw["txt_test"]], axis=0).astype(np.float32)
    y = np.concatenate(
        [_as_int_labels(raw["labels_train"]), _as_int_labels(raw["labels_test"])]
    )
    s_tr = np.loadtxt(os.path.join(root, "train", "subject_train.txt")).astype(np.int64).ravel()
    s_te = np.loadtxt(os.path.join(root, "test", "subject_test.txt")).astype(np.int64).ravel()
    if len(s_tr) != len(raw["labels_train"]) or len(s_te) != len(raw["labels_test"]):
        raise ValueError(
            f"subject files length mismatch: train {len(s_tr)} vs {len(raw['labels_train'])}, "
            f"test {len(s_te)} vs {len(raw['labels_test'])}"
        )
    subj = np.concatenate([s_tr, s_te])

    client_data = {}
    n_per = []
    n_cls = []
    for sid in sorted(np.unique(subj).tolist()):
        mask = subj == sid
        packed = _pack_client(img[mask], gyro[mask], y[mask])
        cid = f"S{int(sid):02d}"
        client_data[cid] = packed
        n_per.append(int(mask.sum()))
        n_cls.append(len(set(packed["Acc"][1])))

    print(f"ActionSense-style subject clients from {root}")
    print(
        f"  Acc dim={img.shape[1]}, Gyro dim={gyro.shape[1]}, "
        f"classes={int(y.max()) + 1}, N={len(y)}, subjects={len(client_data)}"
    )
    print(f"  samples/client: min={min(n_per)}, max={max(n_per)}, mean={np.mean(n_per):.0f}")
    print(f"  classes/client: min={min(n_cls)}, max={max(n_cls)}")

    meta = {
        "dataset_dir": dataset_dir,
        "uci_root": root,
        "reused_sketchfusion_cache": False,
        "partition": "subject",
        "num_clients": len(client_data),
        "acc_dim": int(img.shape[1]),
        "gyro_dim": int(gyro.shape[1]),
        "samples_per_client": n_per,
        "num_test": int(len(raw["labels_test"])),
        "modalities": MODALITIES,
    }
    global_test = _pack_client(raw["img_test"], raw["txt_test"], raw["labels_test"])
    return client_data, global_test, meta


def dirichlet_partition_data(client_data, alpha, seed=42):
    """Optional extra label skew, same role as ActionSense ``class_non_iid_rate``."""
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
    """Give every client the official UCI HAR test set (SketchFusionB eval protocol)."""
    out = {}
    for cid in client_ids:
        out[cid] = {
            m: (list(global_test[m][0]), list(global_test[m][1]))
            for m in global_test
        }
    return out
