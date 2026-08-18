"""
Independent Compression training entry.

Summation fusion of the two modalities, then original FetchSGD Count
Sketch of the full gradient (Rothchild et al.).  Does **not** enable
``--mm_sketch_fusion`` / ``--mm_sketch_fusion_tri`` / ``--mm_sketch_separated``.

Isolated from ``mm_train.py`` so existing MultiModalNet / SketchFusion
runs are unchanged.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import models
from models.independent_compression_net import IndependentCompression

# So ``parse_args --model IndependentCompression`` is a valid choice
# without editing models/__init__.py.
models.IndependentCompression = IndependentCompression

from CommEfficient.utils import get_grad
from fed_aggregator import FedModel, FedOptimizer
from mm_train import (
    compute_loss_multi_label,
    compute_loss_single_label,
    get_data_loaders,
    train,
)
from utils import (
    PiecewiseLinear,
    TableLogger,
    Timer,
    make_logdir,
    parse_args,
    steps_per_epoch,
)

import torch.multiprocessing as multiprocessing


def _warn_if_sketch_fusion_flags(args):
    flags = (
        "mm_sketch_fusion",
        "mm_sketch_fusion_tri",
        "mm_sketch_separated",
    )
    on = [f for f in flags if getattr(args, f, False)]
    if on:
        print(
            "WARNING: Independent Compression uses original FetchSGD "
            "(one Count Sketch of the full gradient).  Ignoring: "
            + ", ".join("--" + f for f in on)
        )
        for f in on:
            setattr(args, f, False)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    print("MY PID:", os.getpid())

    args = parse_args()
    args.model = "IndependentCompression"
    _warn_if_sketch_fusion_flags(args)
    print(args)

    timer = Timer()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, test_loader = get_data_loaders(args)

    multi_label = train_loader.dataset.multi_label
    if multi_label:
        print("Detected MULTI-LABEL dataset — using BCEWithLogitsLoss")
        compute_loss_fn = compute_loss_multi_label
    else:
        print("Detected single-label dataset — using CrossEntropyLoss")
        compute_loss_fn = compute_loss_single_label

    if args.num_classes is None and train_loader.dataset.mm_num_classes:
        args.num_classes = train_loader.dataset.mm_num_classes
    num_classes = args.num_classes

    model_config = {
        "img_dim": args.img_dim,
        "txt_dim": args.txt_dim,
        "feat_dim": args.feat_dim,
        "num_classes": num_classes,
        "dropout": args.mm_dropout,
    }
    print(f"IndependentCompression config: {model_config}")
    print(
        "Fusion: summation (f'_img + f'_txt).  "
        "Gradient compression: original FetchSGD Count Sketch."
    )
    model = IndependentCompression(**model_config)

    param_groups = model.parameters()
    opt = optim.SGD(param_groups, lr=1)

    model = FedModel(model, compute_loss_fn, args, compute_loss_fn)
    opt = FedOptimizer(opt, args)

    if args.mode != "fedavg":
        lr_schedule = PiecewiseLinear(
            [0, args.pivot_epoch, args.num_epochs],
            [0, args.lr_scale, 0],
        )
        spe = steps_per_epoch(
            args.local_batch_size, train_loader.dataset, args.num_workers,
        )
        mle = max(float(getattr(args, "mm_local_epochs", 1.0)), 1e-8)
        spe_lr = spe * mle
        lr_scheduler = LambdaLR(
            opt, lr_lambda=lambda step: lr_schedule(step / spe_lr),
        )
    else:
        lr_scheduler = None

    log_dir = make_logdir(args)
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    print("Finished initializing in {:.2f} seconds".format(timer()))

    _spe0 = steps_per_epoch(
        args.local_batch_size, train_loader.dataset, args.num_workers,
    )
    _mle = max(float(getattr(args, "mm_local_epochs", 1.0)), 1e-8)
    print(
        "Federated train: steps_per_epoch(spe)={:.0f}, mm_local_epochs={:.3g}, "
        "LR divisor spe*mle={:.3g} (≈ max batches per full epoch before "
        "last partial)".format(_spe0, _mle, _spe0 * _mle)
    )

    grad = get_grad(model, args)
    print("Grad size:", grad.numel())
    print("Total params:", sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))

    train(
        model, opt, lr_scheduler, train_loader, test_loader, args,
        writer, loggers=(TableLogger(),), timer=timer,
    )
    model.finalize()
