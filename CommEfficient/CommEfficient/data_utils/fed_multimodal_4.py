"""
Federated dataset with **four** modality vectors per sample (ActionSense 4-way).

Expects ``data.npz`` produced by ``prepare_actionsense_4mod_mm.py``:
  ``m0_train`` … ``m3_train``, ``labels_train``, ``m0_test`` … ``m3_test``, ``labels_test``

Caches per-client ``client{i}.npz`` with ``m0_feats`` … ``m3_feats`` + ``labels``,
plus ``stats.json`` (same role as :class:`FedMultiModal`).
"""

from __future__ import annotations

import json
import os
import glob as glob_module

import numpy as np
import torch

from data_utils.fed_dataset import FedDataset

__all__ = ["FedMultiModal4"]

_NUM_MOD = 4


class FedMultiModal4(FedDataset):

    def __init__(self, *args, dirichlet_alpha=0.1, **kwargs):
        self.dirichlet_alpha = dirichlet_alpha

        dataset_dir = args[0] if args else kwargs.get("dataset_dir", "")
        num_clients = kwargs.get("num_clients", None)
        stats_path = os.path.join(dataset_dir, "stats.json")

        if num_clients is not None and os.path.exists(stats_path):
            with open(stats_path) as f:
                cached_n = len(json.load(f)["images_per_client"])
            if cached_n != num_clients:
                print(
                    f"Cache has {cached_n} clients but {num_clients} "
                    f"requested — clearing cache and re-preparing"
                )
                for fp in glob_module.glob(
                        os.path.join(dataset_dir, "client*.npz")):
                    os.remove(fp)
                for fp in [os.path.join(dataset_dir, "test.npz"),
                           stats_path]:
                    if os.path.exists(fp):
                        os.remove(fp)

        super().__init__(*args, **kwargs)

        if self.type == "train":
            self.client_m = [[] for _ in range(_NUM_MOD)]
            self.client_labels = []
            for cid in range(len(self.images_per_client)):
                data = np.load(self.client_fn(cid))
                for k in range(_NUM_MOD):
                    self.client_m[k].append(data[f"m{k}_feats"])
                self.client_labels.append(data["labels"])
        else:
            data = np.load(self.test_fn())
            self.test_m = [data[f"m{k}_feats"] for k in range(_NUM_MOD)]
            self.test_labels = data["labels"]

    def _load_meta(self, train):
        super()._load_meta(train)
        with open(self.stats_fn(), "r") as f:
            stats = json.load(f)
        self.multi_label = stats.get("multi_label", False)
        self.mm_num_classes = stats.get("num_classes", None)

    def prepare_datasets(self, download=False):
        os.makedirs(self.dataset_dir, exist_ok=True)
        packs = self._load_raw_features()
        (
            m0_tr, m1_tr, m2_tr, m3_tr, lbl_tr,
            m0_te, m1_te, m2_te, m3_te, lbl_te,
        ) = packs

        multi_label = (
            lbl_tr.ndim == 2 and lbl_tr.shape[1] > 1
            and lbl_tr.sum(axis=1).max() > 1
        )

        if multi_label:
            num_classes = lbl_tr.shape[1]
            dominant_class = lbl_tr.argmax(axis=1)
        else:
            if lbl_tr.ndim == 2:
                lbl_tr = lbl_tr.argmax(axis=1)
            if lbl_te.ndim == 2:
                lbl_te = lbl_te.argmax(axis=1)
            num_classes = int(lbl_tr.max()) + 1
            dominant_class = lbl_tr

        n_clients = self._num_clients if self._num_clients else num_classes
        lbl_dtype = np.float32 if multi_label else np.int64

        alpha = getattr(self, "dirichlet_alpha", 0.1)
        rng = np.random.RandomState(42)
        label_dist = rng.dirichlet([alpha] * n_clients, num_classes)

        client_indices = [[] for _ in range(n_clients)]
        for i in range(len(lbl_tr)):
            dc = int(dominant_class[i])
            cid = rng.choice(n_clients, p=label_dist[dc])
            client_indices[cid].append(i)

        m_tr = (m0_tr, m1_tr, m2_tr, m3_tr)
        images_per_client = []
        for cid in range(n_clients):
            idx = np.array(client_indices[cid])
            if len(idx) > 0:
                payload = {
                    f"m{k}_feats": m_tr[k][idx].astype(np.float32)
                    for k in range(_NUM_MOD)
                }
                payload["labels"] = lbl_tr[idx].astype(lbl_dtype)
                np.savez(self.client_fn(cid), **payload)
            else:
                payload = {
                    f"m{k}_feats": np.empty((0, m_tr[k].shape[1]),
                                            dtype=np.float32)
                    for k in range(_NUM_MOD)
                }
                payload["labels"] = np.empty((0,) + lbl_tr.shape[1:],
                                             dtype=lbl_dtype)
                np.savez(self.client_fn(cid), **payload)
            images_per_client.append(len(idx))

        print(
            f"Dirichlet partitioning (alpha={alpha}): "
            f"{n_clients} clients, samples/client: "
            f"min={min(images_per_client)}, "
            f"max={max(images_per_client)}, "
            f"mean={np.mean(images_per_client):.0f}"
        )

        lbl_te_save = lbl_te.astype(lbl_dtype)
        te_payload = {
            f"m{k}_feats": m_te.astype(np.float32)
            for k, m_te in enumerate((m0_te, m1_te, m2_te, m3_te))
        }
        te_payload["labels"] = lbl_te_save
        np.savez(self.test_fn(), **te_payload)

        with open(self.stats_fn(), "w") as f:
            json.dump({
                "images_per_client": images_per_client,
                "num_val_images": int(len(lbl_te)),
                "multi_label": bool(multi_label),
                "num_classes": int(num_classes),
            }, f)

    def _load_raw_features(self):
        npz_path = os.path.join(self.dataset_dir, "data.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Expected 4-modality data.npz at {npz_path} "
                f"(run prepare_actionsense_4mod_mm.py)."
            )
        d = np.load(npz_path)
        for k in range(_NUM_MOD):
            for split in ("train", "test"):
                key = f"m{k}_{split}"
                if key not in d:
                    raise KeyError(f"Missing {key!r} in {npz_path}")
        need = ("labels_train", "labels_test")
        for key in need:
            if key not in d:
                raise KeyError(f"Missing {key!r} in {npz_path}")
        return (
            d["m0_train"], d["m1_train"], d["m2_train"], d["m3_train"],
            d["labels_train"],
            d["m0_test"], d["m1_test"], d["m2_test"], d["m3_test"],
            d["labels_test"],
        )

    def _get_train_item(self, client_id, idx_within_client):
        tensors = tuple(
            torch.as_tensor(self.client_m[k][client_id][idx_within_client],
                            dtype=torch.float32)
            for k in range(_NUM_MOD)
        )
        lbl = self.client_labels[client_id][idx_within_client]
        if self.multi_label:
            lbl_t = torch.as_tensor(lbl, dtype=torch.float32)
        else:
            lbl_t = int(lbl)
        return tensors + (lbl_t,)

    def _get_val_item(self, idx):
        tensors = tuple(
            torch.as_tensor(self.test_m[k][idx], dtype=torch.float32)
            for k in range(_NUM_MOD)
        )
        lbl = self.test_labels[idx]
        if self.multi_label:
            lbl_t = torch.as_tensor(lbl, dtype=torch.float32)
        else:
            lbl_t = int(lbl)
        return tensors + (lbl_t,)

    def __getitem__(self, idx):
        if self.type == "train":
            orig_idx = idx
            if self.do_iid:
                idx = self.iid_shuffle[idx]

            cumsum = np.cumsum(self.images_per_client)
            cid_store = int(np.searchsorted(cumsum, idx, side="right"))
            offsets = np.hstack([[0], cumsum[:-1]])
            idx_within_client = idx - offsets[cid_store]

            items = self._get_train_item(cid_store, idx_within_client)
            *mods, target = items

            cumsum_dpc = np.cumsum(self.data_per_client)
            client_id = int(np.searchsorted(cumsum_dpc, orig_idx,
                                            side="right"))
        else:
            items = self._get_val_item(idx)
            *mods, target = items
            client_id = -1

        return (client_id, *mods, target)

    def client_fn(self, client_id):
        return os.path.join(self.dataset_dir, f"client{client_id}.npz")

    def test_fn(self):
        return os.path.join(self.dataset_dir, "test.npz")
