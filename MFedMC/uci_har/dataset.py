import json
import os
from collections import defaultdict
import numpy as np

# Same 10-client Dirichlet split as run_uci_har_sketch_fusion_B.sh / FedMultiModal.
NUM_CLIENTS = 10
DIRICHLET_ALPHA = 0.1
DIRICHLET_SEED = 42


def _as_int_labels(y):
    y = np.asarray(y)
    if y.ndim == 2:
        y = y.argmax(axis=1)
    return y.astype(np.int64).ravel()


def _resolve_data_npz(filepath):
    filepath = os.path.abspath(filepath)
    if os.path.isdir(filepath):
        data_npz = os.path.join(filepath, 'data.npz')
        dataset_dir = filepath
    else:
        data_npz = filepath
        dataset_dir = os.path.dirname(filepath)
    if not os.path.isfile(data_npz):
        raise FileNotFoundError(
            f"Missing {data_npz}. Build it with "
            "CommEfficient/CommEfficient/prepare_uci_har_mm.py "
            "(same as run_uci_har_sketch_fusion_B.sh)."
        )
    return data_npz, dataset_dir


def _pack_client(acc, gyro, labels):
    labels = _as_int_labels(labels)
    n = len(labels)
    acc = np.asarray(acc, dtype=np.float32)
    gyro = np.asarray(gyro, dtype=np.float32)
    y = [int(labels[i]) for i in range(n)]
    return {
        'Acc': [[acc[i] for i in range(n)], list(y)],
        'Gyro': [[gyro[i] for i in range(n)], list(y)],
    }


def _dirichlet_indices(labels, n_clients, alpha, seed=42):
    """Same algorithm as CommEfficient FedMultiModal.prepare_datasets."""
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


def _cache_matches(dataset_dir, num_clients):
    stats_path = os.path.join(dataset_dir, 'stats.json')
    test_path = os.path.join(dataset_dir, 'test.npz')
    if not os.path.isfile(test_path):
        return False
    for i in range(num_clients):
        if not os.path.isfile(os.path.join(dataset_dir, f'client{i}.npz')):
            return False
    if os.path.isfile(stats_path):
        with open(stats_path, encoding='utf-8') as f:
            stats = json.load(f)
        cached_n = len(stats.get('images_per_client', []))
        if cached_n != num_clients:
            return False
    return True


def load_and_restructure_uci_har_data(filepath):
    """Load SketchFusionB UCI HAR Acc/Gyro features into ActionSense-style client dicts.

    Clients are the same 10 Dirichlet partitions of the **official train** split
    used by ``run_uci_har_sketch_fusion_B.sh`` (reuses ``client*.npz`` when present).
    Returns (client_data, global_test) where global_test is the official UCI HAR
    test set (``img_test`` / ``txt_test`` / ``labels_test``).
    """
    data_npz, dataset_dir = _resolve_data_npz(filepath)
    raw = np.load(data_npz)
    img_tr = raw['img_train']
    txt_tr = raw['txt_train']
    y_tr = _as_int_labels(raw['labels_train'])
    global_test = _pack_client(raw['img_test'], raw['txt_test'], raw['labels_test'])

    if _cache_matches(dataset_dir, NUM_CLIENTS):
        print(f"Reusing SketchFusionB client cache in {dataset_dir} ({NUM_CLIENTS} clients).")
        client_data = {}
        n_per = []
        for i in range(NUM_CLIENTS):
            d = np.load(os.path.join(dataset_dir, f'client{i}.npz'))
            packed = _pack_client(d['img_feats'], d['txt_feats'], d['labels'])
            client_data[f'C{i:02d}'] = packed
            n_per.append(len(packed['Acc'][1]))
    else:
        print(
            f"Partitioning data.npz with Dirichlet alpha={DIRICHLET_ALPHA}, "
            f"seed={DIRICHLET_SEED}, n_clients={NUM_CLIENTS} "
            "(same algorithm as FedMultiModal / SketchFusionB)."
        )
        idxs = _dirichlet_indices(y_tr, NUM_CLIENTS, DIRICHLET_ALPHA, seed=DIRICHLET_SEED)
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
            client_data[f'C{i:02d}'] = packed
            n_per.append(len(packed['Acc'][1]))

    print(
        f"  Acc dim={img_tr.shape[1]}, Gyro dim={txt_tr.shape[1]}, "
        f"classes={int(y_tr.max()) + 1}, train={len(y_tr)}, "
        f"official test={len(global_test['Acc'][1])}"
    )
    print(f"  samples/client: {n_per} (min={min(n_per)}, max={max(n_per)})")
    return client_data, global_test


def attach_global_test(client_ids, global_test):
    """Give every client the official UCI HAR test set (SketchFusionB eval protocol)."""
    out = {}
    for cid in client_ids:
        out[cid] = {
            m: (list(global_test[m][0]), list(global_test[m][1]))
            for m in global_test
        }
    return out


def dirichlet_partition_data(client_data, alpha, seed=42):
    np.random.seed(seed)
    clients = list(client_data.keys())
    data_by_label = defaultdict(list)

    for client_id, streams in client_data.items():
        ref_key = list(streams.keys())[0]
        ref_datasets, ref_labels = streams[ref_key]
        for i, label in enumerate(ref_labels):
            sample_data = {mapped_key: (datasets[i], labels[i]) for mapped_key, (datasets, labels) in streams.items() if i < len(datasets)}
            data_by_label[label].append((client_id, sample_data))

    new_client_data = {client_id: {key: ([], []) for key in client_data[client_id].keys()} for client_id in clients}

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


def stratified_split_client_data(client_data, train_ratio=0.7):
    client_data_train = {}
    client_data_test = {}
    for client, modalities_data in client_data.items():
        client_data_train[client] = {}
        client_data_test[client] = {}
        ref_modality = list(modalities_data.keys())[0]
        _, y = modalities_data[ref_modality]
        y = np.asarray(y)
        train_indices = []
        test_indices = []
        if len(y) == 0:
            for device_stream, data in modalities_data.items():
                client_data_train[client][device_stream] = (list(data[0]), list(data[1]))
                client_data_test[client][device_stream] = ([], [])
            continue
        for cls in np.unique(y):
            all_indices = np.where(y == cls)[0]
            np.random.shuffle(all_indices)
            boundary = int(len(all_indices) * train_ratio)
            train_indices.extend(all_indices[:boundary].tolist())
            test_indices.extend(all_indices[boundary:].tolist())
        for device_stream, data in modalities_data.items():
            x = data[0]
            y_all = data[1]
            x_train = [x[i] for i in train_indices]
            y_train = [y_all[i] for i in train_indices]
            x_test = [x[i] for i in test_indices]
            y_test = [y_all[i] for i in test_indices]
            client_data_train[client][device_stream] = (x_train, y_train)
            client_data_test[client][device_stream] = (x_test, y_test)
    return client_data_train, client_data_test
