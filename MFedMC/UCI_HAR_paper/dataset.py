"""Load the raw UCI HAR Inertial Signals into MFedMC client dicts — the paper's
own input representation, not SketchFusionB's engineered 348/213-D feature cache.

Modalities (paper Table 1: "Accelerometer 128x3, Gyroscope 128x3"):
  Acc  — total_acc_{x,y,z}_{train,test}.txt: the raw, unfiltered accelerometer
         reading (gravity + body motion), in g's. Per UCI HAR's own README.txt
         this is "the acceleration signal from the smartphone accelerometer" —
         the literal sensor output.
  Gyro — body_gyro_{x,y,z}_{train,test}.txt: the gyroscope's angular velocity,
         rad/s. No body/total split exists for the gyroscope.
(body_acc_* is deliberately NOT used here: README.txt describes it as a
*derived* signal, "obtained by subtracting the gravity from the total
acceleration" via a low-pass filter — less literally "raw" than total_acc_*.)

Natural partition: one client per real UCI HAR subject (30 total: 21 in
subject_train.txt, 9 in subject_test.txt, confirmed no overlap), matching the
paper's "Natural Distribution" (Table 1: "30 Subjects"). No Dirichlet client
synthesis, no shared global test set — evaluation is per-client local holdout.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

MODALITIES = ["Acc", "Gyro"]
WINDOW_LEN = 128
NUM_AXES = 3

_ACC_PREFIX = "total_acc"
_GYRO_PREFIX = "body_gyro"

_DEFAULT_UCI_ROOTS = (
    "human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset",
    "MFedMC/UCI_HAR/data/UCI HAR Dataset",
)


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
        "Need UCI HAR train/subject_train.txt (and Inertial Signals/) for the "
        "raw-signal paper reproduction. Pass --uci_root to the inner "
        "'UCI HAR Dataset' folder."
    )


def _load_axis(root: str, split: str, prefix: str, axis: str) -> np.ndarray:
    path = os.path.join(root, split, "Inertial Signals", f"{prefix}_{axis}_{split}.txt")
    return np.loadtxt(path, dtype=np.float32)  # (N, 128)


def _load_modality(root: str, split: str, prefix: str) -> np.ndarray:
    axes = [_load_axis(root, split, prefix, ax) for ax in ("x", "y", "z")]
    return np.stack(axes, axis=-1)  # (N, 128, 3)


def _load_labels(root: str, split: str) -> np.ndarray:
    y = np.loadtxt(os.path.join(root, split, f"y_{split}.txt"), dtype=np.int64)
    return y.ravel() - 1  # UCI HAR labels are 1..6 -> 0..5


def _load_subjects(root: str, split: str) -> np.ndarray:
    return np.loadtxt(os.path.join(root, split, f"subject_{split}.txt"), dtype=np.int64).ravel()


def load_raw_signals(uci_root: str | None = None) -> dict:
    """Loads all 18 Inertial Signals files + labels + subjects, train and test.
    Small text files (largest ~4MB) — load whole, no chunking needed.
    """
    root = find_uci_root(uci_root)
    out = {"root": root}
    for split in ("train", "test"):
        out[f"acc_{split}"] = _load_modality(root, split, _ACC_PREFIX)
        out[f"gyro_{split}"] = _load_modality(root, split, _GYRO_PREFIX)
        out[f"y_{split}"] = _load_labels(root, split)
        out[f"subj_{split}"] = _load_subjects(root, split)
    return out


def _pack_client(acc: np.ndarray, gyro: np.ndarray, labels: np.ndarray) -> dict:
    """Per-sample Python lists (not one stacked tensor), matching UCI_HAR_v2's
    _pack_client convention, so stratified_split_client_data's `x[i] for i in
    indices` stays shape-agnostic regardless of whether x[i] is a flat vector
    or, here, a (128, 3) sequence array.
    """
    n = len(labels)
    y = [int(labels[i]) for i in range(n)]
    return {
        "Acc": ([acc[i] for i in range(n)], list(y)),
        "Gyro": ([gyro[i] for i in range(n)], list(y)),
    }


def load_subject_clients(uci_root: str | None = None):
    """One client per real UCI HAR subject (train+test concatenated, grouped by
    subject id). Returns (client_data, meta) -- no shared global test set; this
    package only ever evaluates on each client's own local holdout.
    """
    raw = load_raw_signals(uci_root)
    acc = np.concatenate([raw["acc_train"], raw["acc_test"]], axis=0)
    gyro = np.concatenate([raw["gyro_train"], raw["gyro_test"]], axis=0)
    y = np.concatenate([raw["y_train"], raw["y_test"]])
    subj = np.concatenate([raw["subj_train"], raw["subj_test"]])

    client_data = {}
    n_per = []
    n_cls = []
    for sid in sorted(np.unique(subj).tolist()):
        mask = subj == sid
        packed = _pack_client(acc[mask], gyro[mask], y[mask])
        cid = f"S{int(sid):02d}"
        client_data[cid] = packed
        n_per.append(int(mask.sum()))
        n_cls.append(len(set(packed["Acc"][1])))

    print(f"Raw UCI HAR Inertial Signals from {raw['root']}")
    print(
        f"  Acc shape=(128,3) [{_ACC_PREFIX}]  Gyro shape=(128,3) [{_GYRO_PREFIX}]  "
        f"classes=6  N={len(y)}  subjects={len(client_data)}"
    )
    print(f"  samples/client: min={min(n_per)}, max={max(n_per)}, mean={np.mean(n_per):.0f}")
    print(f"  classes/client: min={min(n_cls)}, max={max(n_cls)}")

    meta = {
        "uci_root": raw["root"],
        "num_clients": len(client_data),
        "acc_shape": (WINDOW_LEN, NUM_AXES),
        "gyro_shape": (WINDOW_LEN, NUM_AXES),
        "samples_per_client": n_per,
        "modalities": MODALITIES,
    }
    return client_data, meta


def dirichlet_partition_data(client_data, alpha, seed=42):
    """Optional extra label-skew ablation (off by default via --class_non_iid_rate
    1.0), same role as UCI_HAR_v2's function of the same name -- reimplemented
    independently, identical algorithm.
    """
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
    """Per-client stratified split, matching the paper's local-holdout protocol.
    Shape-agnostic (only ever indexes `x[i] for i in indices`), so this works
    unmodified on the (128, 3) sequence samples packed above.
    """
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
