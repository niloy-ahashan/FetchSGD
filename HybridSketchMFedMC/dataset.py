"""Load the SketchFusionB UCI HAR split (Acc 348-D + Gyro 213-D), or the
SketchFusionB4 mHealth split (Acc 177-D + Gyro 118-D + Mag 118-D + ECG 43-D).

Reuses ``datasets/uci_har_mm`` / ``datasets/mhealth_mm`` including cached
``client*.npz`` when present so the 10-client Dirichlet partition matches
SketchFusionB(4) / MFedMC.
"""

from __future__ import annotations

import json
import os

import numpy as np

UCI_HAR_MODALITIES = ["Acc", "Gyro"]
MHEALTH_MODALITIES = ["Acc", "Gyro", "Mag", "ECG"]


def _as_int_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        y = y.argmax(axis=1)
    return y.astype(np.int64).ravel()


def dirichlet_indices(labels, n_clients, alpha, seed=42):
    """Same algorithm as CommEfficient FedMultiModal (RandomState(42) + Dirichlet)."""
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


def _pack(mods, labels):
    labels = _as_int_labels(labels)
    return {
        "xs": [np.asarray(m, dtype=np.float32) for m in mods],
        "y": labels,
    }


def load_uci_har_mm(
    dataset_dir,
    num_clients=10,
    dirichlet_alpha=0.1,
    seed=42,
):
    dataset_dir = os.path.abspath(dataset_dir)
    data_npz = os.path.join(dataset_dir, "data.npz")
    if not os.path.isfile(data_npz):
        raise FileNotFoundError(
            f"Missing {data_npz}. Build it with "
            "CommEfficient/CommEfficient/prepare_uci_har_mm.py "
            "(same as run_uci_har_sketch_fusion_B.sh)."
        )

    raw = np.load(data_npz)
    img_tr = raw["img_train"]
    txt_tr = raw["txt_train"]
    y_tr = _as_int_labels(raw["labels_train"])
    img_te = raw["img_test"]
    txt_te = raw["txt_test"]
    y_te = _as_int_labels(raw["labels_test"])
    global_test = _pack([img_te, txt_te], y_te)

    reused_cache = False
    if _cache_matches(dataset_dir, num_clients):
        print(
            f"Reusing SketchFusionB client cache in {dataset_dir} "
            f"({num_clients} clients)."
        )
        clients = []
        n_per = []
        for i in range(num_clients):
            d = np.load(os.path.join(dataset_dir, f"client{i}.npz"))
            packed = _pack([d["img_feats"], d["txt_feats"]], d["labels"])
            clients.append(packed)
            n_per.append(int(len(packed["y"])))
        reused_cache = True
    else:
        print(
            f"Partitioning data.npz with Dirichlet alpha={dirichlet_alpha}, "
            f"seed={seed}, n_clients={num_clients}."
        )
        idxs = dirichlet_indices(y_tr, num_clients, dirichlet_alpha, seed=seed)
        clients = []
        n_per = []
        for idx in idxs:
            idx = np.asarray(idx, dtype=np.int64)
            if len(idx) == 0:
                packed = _pack(
                    [
                        np.empty((0, img_tr.shape[1]), dtype=np.float32),
                        np.empty((0, txt_tr.shape[1]), dtype=np.float32),
                    ],
                    np.empty((0,), dtype=np.int64),
                )
            else:
                packed = _pack([img_tr[idx], txt_tr[idx]], y_tr[idx])
            clients.append(packed)
            n_per.append(int(len(packed["y"])))

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
        "mod_dims": [int(img_tr.shape[1]), int(txt_tr.shape[1])],
        "modalities": list(UCI_HAR_MODALITIES),
        "samples_per_client": n_per,
        "num_classes": int(y_tr.max()) + 1,
        "num_test": int(len(y_te)),
    }
    return clients, global_test, meta


def load_mhealth_mm(
    dataset_dir,
    num_clients=10,
    dirichlet_alpha=0.1,
    seed=42,
):
    """SketchFusionB4-style mHealth split: reuses ``datasets/mhealth_mm``
    (see ``prepare_mhealth_mm.py``) including cached ``client*.npz`` when
    present so the 10-client Dirichlet partition matches SketchFusionB4 /
    MFedMC/mHealth.
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
    n_mod = len(MHEALTH_MODALITIES)
    mods_tr = [raw[f"m{k}_train"] for k in range(n_mod)]
    mods_te = [raw[f"m{k}_test"] for k in range(n_mod)]
    y_tr = _as_int_labels(raw["labels_train"])
    y_te = _as_int_labels(raw["labels_test"])
    global_test = _pack(mods_te, y_te)

    reused_cache = False
    if _cache_matches(dataset_dir, num_clients):
        print(
            f"Reusing SketchFusionB4 client cache in {dataset_dir} "
            f"({num_clients} clients)."
        )
        clients = []
        n_per = []
        for i in range(num_clients):
            d = np.load(os.path.join(dataset_dir, f"client{i}.npz"))
            packed = _pack([d[f"m{k}_feats"] for k in range(n_mod)], d["labels"])
            clients.append(packed)
            n_per.append(int(len(packed["y"])))
        reused_cache = True
    else:
        print(
            f"Partitioning data.npz with Dirichlet alpha={dirichlet_alpha}, "
            f"seed={seed}, n_clients={num_clients}."
        )
        idxs = dirichlet_indices(y_tr, num_clients, dirichlet_alpha, seed=seed)
        clients = []
        n_per = []
        for idx in idxs:
            idx = np.asarray(idx, dtype=np.int64)
            if len(idx) == 0:
                packed = _pack(
                    [np.empty((0, m.shape[1]), dtype=np.float32) for m in mods_tr],
                    np.empty((0,), dtype=np.int64),
                )
            else:
                packed = _pack([m[idx] for m in mods_tr], y_tr[idx])
            clients.append(packed)
            n_per.append(int(len(packed["y"])))

    mod_dims = [int(m.shape[1]) for m in mods_tr]
    print(
        f"  mod_dims={dict(zip(MHEALTH_MODALITIES, mod_dims))}, "
        f"classes={int(y_tr.max()) + 1}, train={len(y_tr)}, test={len(y_te)}"
    )
    print(f"  samples/client: {n_per} (min={min(n_per)}, max={max(n_per)})")

    meta = {
        "dataset_dir": dataset_dir,
        "reused_sketchfusion_cache": reused_cache,
        "num_clients": num_clients,
        "dirichlet_alpha": dirichlet_alpha,
        "mod_dims": mod_dims,
        "modalities": list(MHEALTH_MODALITIES),
        "samples_per_client": n_per,
        "num_classes": int(y_tr.max()) + 1,
        "num_test": int(len(y_te)),
    }
    return clients, global_test, meta
