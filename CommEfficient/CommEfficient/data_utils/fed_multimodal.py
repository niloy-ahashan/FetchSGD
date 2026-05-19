"""
Federated multimodal dataset for image-text pairs.

Supported input formats
-----------------------
1. **HDF5** file  (``mir.h5``, ``wiki.h5``, ``nus.h5``)
   Keys: ``I_tr / T_tr / L_tr / I_te / T_te / L_te``
   (arrays are stored transposed — ``(feat_dim, N)`` — and will be
   automatically transposed to ``(N, feat_dim)`` on load).

2. **Single .mat** file  (``data.mat``)
   Auto-detected key names.

3. **Single .npz** file  (``data.npz``)
   Keys: ``img_train, txt_train, labels_train,
           img_test, txt_test, labels_test``

4. **Separate .npy** files in ``dataset_dir``.

Multi-label support
-------------------
When labels are 2-D one-hot matrices (e.g. MIR-Flickr 24-class),
they are kept as float vectors and samples are partitioned across
clients using Dirichlet-based non-IID splitting (same as PMFH).
A ``multi_label`` flag is stored in ``stats.json`` so the training
script can pick the right loss function automatically.

After the first run the data is split per-client and cached as
``client0.npz … clientN.npz`` + ``test.npz`` + ``stats.json``.
"""

import json
import os
import glob as glob_module

import numpy as np
import torch

from data_utils.fed_dataset import FedDataset

__all__ = ["FedMultiModal"]


class FedMultiModal(FedDataset):

    def __init__(self, *args, dirichlet_alpha=0.1, **kwargs):
        self.dirichlet_alpha = dirichlet_alpha

        dataset_dir = args[0] if args else kwargs.get('dataset_dir', '')
        num_clients = kwargs.get('num_clients', None)
        stats_path = os.path.join(dataset_dir, "stats.json")

        if num_clients is not None and os.path.exists(stats_path):
            with open(stats_path) as f:
                cached_n = len(json.load(f)["images_per_client"])
            if cached_n != num_clients:
                print(f"Cache has {cached_n} clients but {num_clients} "
                      f"requested — clearing cache and re-preparing")
                for fp in glob_module.glob(
                        os.path.join(dataset_dir, "client*.npz")):
                    os.remove(fp)
                for fp in [os.path.join(dataset_dir, "test.npz"),
                           stats_path]:
                    if os.path.exists(fp):
                        os.remove(fp)

        super().__init__(*args, **kwargs)

        if self.type == "train":
            self.client_img = []
            self.client_txt = []
            self.client_labels = []
            for cid in range(len(self.images_per_client)):
                data = np.load(self.client_fn(cid))
                self.client_img.append(data["img_feats"])
                self.client_txt.append(data["txt_feats"])
                self.client_labels.append(data["labels"])
        else:
            data = np.load(self.test_fn())
            self.test_img = data["img_feats"]
            self.test_txt = data["txt_feats"]
            self.test_labels = data["labels"]

    def _load_meta(self, train):
        super()._load_meta(train)
        with open(self.stats_fn(), "r") as f:
            stats = json.load(f)
        self.multi_label = stats.get("multi_label", False)
        self.mm_num_classes = stats.get("num_classes", None)

    # ------------------------------------------------------------------
    # Preparation (runs once to cache per-client splits)
    # ------------------------------------------------------------------
    def prepare_datasets(self, download=False):
        os.makedirs(self.dataset_dir, exist_ok=True)
        img_tr, txt_tr, lbl_tr, img_te, txt_te, lbl_te = \
            self._load_raw_features()

        multi_label = (lbl_tr.ndim == 2 and lbl_tr.shape[1] > 1
                       and lbl_tr.sum(axis=1).max() > 1)

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

        # Dirichlet-based non-IID partitioning (same strategy as PMFH)
        alpha = getattr(self, 'dirichlet_alpha', 0.1)
        rng = np.random.RandomState(42)
        label_dist = rng.dirichlet([alpha] * n_clients, num_classes)

        client_indices = [[] for _ in range(n_clients)]
        for i in range(len(lbl_tr)):
            dc = int(dominant_class[i])
            cid = rng.choice(n_clients, p=label_dist[dc])
            client_indices[cid].append(i)

        images_per_client = []
        for cid in range(n_clients):
            idx = np.array(client_indices[cid])
            if len(idx) > 0:
                np.savez(
                    self.client_fn(cid),
                    img_feats=img_tr[idx].astype(np.float32),
                    txt_feats=txt_tr[idx].astype(np.float32),
                    labels=lbl_tr[idx].astype(lbl_dtype),
                )
            else:
                np.savez(
                    self.client_fn(cid),
                    img_feats=np.empty((0, img_tr.shape[1]),
                                       dtype=np.float32),
                    txt_feats=np.empty((0, txt_tr.shape[1]),
                                       dtype=np.float32),
                    labels=np.empty((0,) + lbl_tr.shape[1:],
                                    dtype=lbl_dtype),
                )
            images_per_client.append(len(idx))

        print(f"Dirichlet partitioning (alpha={alpha}): "
              f"{n_clients} clients, samples/client: "
              f"min={min(images_per_client)}, "
              f"max={max(images_per_client)}, "
              f"mean={np.mean(images_per_client):.0f}")

        lbl_te_save = lbl_te.astype(lbl_dtype)
        np.savez(
            self.test_fn(),
            img_feats=img_te.astype(np.float32),
            txt_feats=txt_te.astype(np.float32),
            labels=lbl_te_save,
        )

        with open(self.stats_fn(), "w") as f:
            json.dump({
                "images_per_client": images_per_client,
                "num_val_images": int(len(lbl_te)),
                "multi_label": bool(multi_label),
                "num_classes": int(num_classes),
            }, f)

    # ------------------------------------------------------------------
    # Raw feature loading (auto-detect format)
    # ------------------------------------------------------------------
    def _load_raw_features(self):
        # --- HDF5 (.h5) or MATLAB v7.3 (.mat, also HDF5 internally) ---
        h5_names = (
            "mir_cnn_twt.mat", "nus_cnn_twt.mat", "coco_cnn_twt_2014.mat",
            "data.mat",
            "mir.h5", "wiki.h5", "nus.h5", "data.h5",
        )
        for name in h5_names:
            path = os.path.join(self.dataset_dir, name)
            if os.path.exists(path):
                try:
                    return self._load_h5(path)
                except Exception:
                    pass

        # --- legacy .mat (v5/v7, readable by scipy) ---
        mat_path = os.path.join(self.dataset_dir, "data.mat")
        if os.path.exists(mat_path):
            import scipy.io as sio
            data = sio.loadmat(mat_path)
            km = self._detect_mat_keys(data)
            return (
                np.asarray(data[km["img_train"]]),
                np.asarray(data[km["txt_train"]]),
                np.asarray(data[km["label_train"]]).squeeze(),
                np.asarray(data[km["img_test"]]),
                np.asarray(data[km["txt_test"]]),
                np.asarray(data[km["label_test"]]).squeeze(),
            )

        # --- .npz ---
        npz_path = os.path.join(self.dataset_dir, "data.npz")
        if os.path.exists(npz_path):
            d = np.load(npz_path)
            return (d["img_train"], d["txt_train"], d["labels_train"],
                    d["img_test"], d["txt_test"], d["labels_test"])

        # --- separate .npy ---
        return (
            np.load(os.path.join(self.dataset_dir, "img_train.npy")),
            np.load(os.path.join(self.dataset_dir, "txt_train.npy")),
            np.load(os.path.join(self.dataset_dir, "labels_train.npy")),
            np.load(os.path.join(self.dataset_dir, "img_test.npy")),
            np.load(os.path.join(self.dataset_dir, "txt_test.npy")),
            np.load(os.path.join(self.dataset_dir, "labels_test.npy")),
        )

    @staticmethod
    def _load_h5(path):
        """Load a PMFH-style HDF5 file (arrays stored transposed)."""
        import h5py
        with h5py.File(path, "r") as f:
            img_tr = np.array(f["I_tr"]).T.astype(np.float32)
            txt_tr = np.array(f["T_tr"]).T.astype(np.float32)
            lbl_tr = np.array(f["L_tr"]).T.astype(np.float32)
            img_te = np.array(f["I_te"]).T.astype(np.float32)
            txt_te = np.array(f["T_te"]).T.astype(np.float32)
            lbl_te = np.array(f["L_te"]).T.astype(np.float32)
        return img_tr, txt_tr, lbl_tr, img_te, txt_te, lbl_te

    @staticmethod
    def _detect_mat_keys(data):
        keys = set(data.keys())
        patterns = [
            {"img_train": "I_tr", "txt_train": "T_tr",
             "label_train": "L_tr",
             "img_test": "I_te", "txt_test": "T_te",
             "label_test": "L_te"},
            {"img_train": "image_train", "txt_train": "text_train",
             "label_train": "label_train",
             "img_test": "image_test", "txt_test": "text_test",
             "label_test": "label_test"},
            {"img_train": "img_train", "txt_train": "txt_train",
             "label_train": "labels_train",
             "img_test": "img_test", "txt_test": "txt_test",
             "label_test": "labels_test"},
        ]
        for p in patterns:
            if all(v in keys for v in p.values()):
                return p
        raise ValueError(
            f"Unrecognised .mat key layout. Available keys: "
            f"{keys - {'__header__', '__version__', '__globals__'}}"
        )

    # ------------------------------------------------------------------
    # Item access (overrides base to return 4-tuples)
    # ------------------------------------------------------------------
    def _get_train_item(self, client_id, idx_within_client):
        img = self.client_img[client_id][idx_within_client]
        txt = self.client_txt[client_id][idx_within_client]
        lbl = self.client_labels[client_id][idx_within_client]

        img_t = torch.as_tensor(img, dtype=torch.float32)
        txt_t = torch.as_tensor(txt, dtype=torch.float32)
        if self.multi_label:
            lbl_t = torch.as_tensor(lbl, dtype=torch.float32)
        else:
            lbl_t = int(lbl)
        return img_t, txt_t, lbl_t

    def _get_val_item(self, idx):
        img = self.test_img[idx]
        txt = self.test_txt[idx]
        lbl = self.test_labels[idx]

        img_t = torch.as_tensor(img, dtype=torch.float32)
        txt_t = torch.as_tensor(txt, dtype=torch.float32)
        if self.multi_label:
            lbl_t = torch.as_tensor(lbl, dtype=torch.float32)
        else:
            lbl_t = int(lbl)
        return img_t, txt_t, lbl_t

    def __getitem__(self, idx):
        if self.type == "train":
            orig_idx = idx
            if self.do_iid:
                idx = self.iid_shuffle[idx]

            cumsum = np.cumsum(self.images_per_client)
            client_id = int(np.searchsorted(cumsum, idx, side="right"))
            offsets = np.hstack([[0], cumsum[:-1]])
            idx_within_client = idx - offsets[client_id]

            img, txt, target = self._get_train_item(
                client_id, idx_within_client
            )

            cumsum_dpc = np.cumsum(self.data_per_client)
            client_id = int(np.searchsorted(cumsum_dpc, orig_idx,
                                            side="right"))
        else:
            img, txt, target = self._get_val_item(idx)
            client_id = -1

        return client_id, img, txt, target

    # ------------------------------------------------------------------
    # File-path helpers
    # ------------------------------------------------------------------
    def client_fn(self, client_id):
        return os.path.join(self.dataset_dir, f"client{client_id}.npz")

    def test_fn(self):
        return os.path.join(self.dataset_dir, "test.npz")
