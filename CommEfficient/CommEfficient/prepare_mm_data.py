#!/usr/bin/env python3
"""
Prepare multimodal data in the format expected by FedMultiModal.

Modes
-----
1. ``--source synthetic``  (default)
   Generates class-conditional image and text feature vectors derived
   from CIFAR-10 so you can test the pipeline without new downloads.

   * Image features  – 128-d vectors extracted via a small random
     projection of the 3072-d flattened CIFAR-10 pixel values, then
     L2-normalised.
   * Text features   – 100-d class-conditional Gaussian vectors
     (each class gets a fixed prototype; per-sample noise is added).

2. ``--source mat --mat_path <path>``
   Converts an existing .mat file (Wikipedia / MIR-Flickr / NUS-WIDE)
   into the ``data.npz`` format expected by FedMultiModal.

Output
------
All files are written to ``--out_dir`` (default ``~/datasets/multimodal/``):
  data.npz  –  keys: img_train, txt_train, labels_train,
                       img_test,  txt_test,  labels_test
"""

import argparse
import os

import numpy as np


def make_synthetic(out_dir, img_dim=128, txt_dim=100, seed=42):
    """Build a multimodal dataset from CIFAR-10."""
    import torchvision

    rng = np.random.RandomState(seed)

    cifar_train = torchvision.datasets.CIFAR10(
        root=os.path.join(out_dir, "_cifar10_cache"),
        train=True, download=True,
    )
    cifar_test = torchvision.datasets.CIFAR10(
        root=os.path.join(out_dir, "_cifar10_cache"),
        train=False, download=True,
    )

    def extract_img(dataset):
        pixels = np.array(dataset.data, dtype=np.float32).reshape(len(dataset), -1)
        pixels = (pixels - pixels.mean(axis=0)) / (pixels.std(axis=0) + 1e-8)
        proj = rng.randn(pixels.shape[1], img_dim).astype(np.float32) * 0.05
        feats = pixels @ proj
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        return (feats / norms).astype(np.float32)

    num_classes = 10
    class_protos = rng.randn(num_classes, txt_dim).astype(np.float32)

    def make_txt(labels):
        out = np.stack([
            class_protos[int(l)] + rng.randn(txt_dim).astype(np.float32) * 0.3
            for l in labels
        ])
        return out.astype(np.float32)

    img_train = extract_img(cifar_train)
    img_test = extract_img(cifar_test)
    labels_train = np.array(cifar_train.targets, dtype=np.int64)
    labels_test = np.array(cifar_test.targets, dtype=np.int64)
    txt_train = make_txt(labels_train)
    txt_test = make_txt(labels_test)

    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "data.npz"),
        img_train=img_train, txt_train=txt_train, labels_train=labels_train,
        img_test=img_test, txt_test=txt_test, labels_test=labels_test,
    )
    print(f"Saved synthetic multimodal data to {out_dir}/data.npz")
    print(f"  img_train: {img_train.shape}  txt_train: {txt_train.shape}")
    print(f"  img_test:  {img_test.shape}   txt_test:  {txt_test.shape}")
    print(f"  classes:   {num_classes}")


def convert_mat(mat_path, out_dir):
    """Convert a standard multimodal .mat file to data.npz."""
    import scipy.io as sio

    data = sio.loadmat(mat_path)
    available = set(data.keys()) - {"__header__", "__version__", "__globals__"}
    print(f"Available keys in {mat_path}: {available}")

    patterns = [
        ("I_tr", "T_tr", "L_tr", "I_te", "T_te", "L_te"),
        ("image_train", "text_train", "label_train",
         "image_test", "text_test", "label_test"),
    ]
    chosen = None
    for p in patterns:
        if all(k in available for k in p):
            chosen = p
            break
    if chosen is None:
        raise ValueError(f"Cannot auto-detect key names.  Keys: {available}")

    img_tr = np.asarray(data[chosen[0]], dtype=np.float32)
    txt_tr = np.asarray(data[chosen[1]], dtype=np.float32)
    lbl_tr = np.asarray(data[chosen[2]]).squeeze()
    img_te = np.asarray(data[chosen[3]], dtype=np.float32)
    txt_te = np.asarray(data[chosen[4]], dtype=np.float32)
    lbl_te = np.asarray(data[chosen[5]]).squeeze()

    if lbl_tr.ndim == 2:
        lbl_tr = lbl_tr.argmax(axis=1)
    if lbl_te.ndim == 2:
        lbl_te = lbl_te.argmax(axis=1)

    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "data.npz"),
        img_train=img_tr, txt_train=txt_tr,
        labels_train=lbl_tr.astype(np.int64),
        img_test=img_te, txt_test=txt_te,
        labels_test=lbl_te.astype(np.int64),
    )
    print(f"Converted {mat_path} → {out_dir}/data.npz")
    print(f"  img_train: {img_tr.shape}  txt_train: {txt_tr.shape}")
    print(f"  img_test:  {img_te.shape}   txt_test:  {txt_te.shape}")
    print(f"  classes:   {int(lbl_tr.max()) + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare multimodal data for FedMultiModal"
    )
    parser.add_argument("--source", choices=["synthetic", "mat"],
                        default="synthetic")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.expanduser("~/datasets/multimodal/"))
    parser.add_argument("--mat_path", type=str, default=None,
                        help="Path to .mat file (required when --source mat)")
    parser.add_argument("--img_dim", type=int, default=128,
                        help="Image feature dim for synthetic mode")
    parser.add_argument("--txt_dim", type=int, default=100,
                        help="Text feature dim for synthetic mode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.source == "synthetic":
        make_synthetic(args.out_dir, args.img_dim, args.txt_dim, args.seed)
    elif args.source == "mat":
        if args.mat_path is None:
            parser.error("--mat_path is required when --source mat")
        convert_mat(args.mat_path, args.out_dir)
