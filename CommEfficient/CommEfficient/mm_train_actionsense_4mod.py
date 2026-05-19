"""
4-modality ActionSense training: :class:`FedMultiModal4` + :class:`SketchFusionB4`.

Requires ``dataset_dir/prepare_stats.json`` from ``prepare_actionsense_4mod_mm.py``
(``mod_dims``, ``num_classes``) and ``data.npz`` with ``m0_*`` … ``m3_*``.

Use ``--skip_map`` (retrieval MAP is only defined for the 2-modality pipeline).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import models
import mm_train as mm_base
from CommEfficient.utils import get_grad
from data_utils import FedMultiModal4, FedSampler
from fed_aggregator import FedModel, FedOptimizer
from utils import (
    Timer,
    TableLogger,
    parse_args,
    PiecewiseLinear,
    steps_per_epoch,
    make_logdir,
)

import torch.multiprocessing as multiprocessing

_ce_criterion = torch.nn.CrossEntropyLoss(reduction="mean")
_bce_criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")


class _TopOneAccuracy(torch.nn.Module):
    def forward(self, logits, target):
        return (logits.max(dim=1)[1] == target).float().mean()


_top1_acc = _TopOneAccuracy()


def _load_prepare_stats(args):
    path = os.path.join(args.dataset_dir, "prepare_stats.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"4-modality training expects {path} — run "
            f"prepare_actionsense_4mod_mm.py (or the S00 wrapper) first."
        )
    with open(path, encoding="utf-8") as f:
        st = json.load(f)
    if "mod_dims" not in st or len(st["mod_dims"]) != 4:
        raise ValueError(
            f"{path} must list mod_dims with length 4; got "
            f"{st.get('mod_dims')!r}"
        )
    args.mod_dims = [int(x) for x in st["mod_dims"]]
    if args.num_classes is None and "num_classes" in st:
        args.num_classes = int(st["num_classes"])


def compute_loss_single_label(model, batch, args):
    m0, m1, m2, m3, targets = batch
    mp = mm_base._get_missing_prob(model, args)
    pred, _H = model(m0, m1, m2, m3, missing_prob=mp)
    loss = _ce_criterion(pred, targets)
    miss_w = getattr(args, "missing_loss_weight", 0.0)
    if miss_w > 0:
        loss = loss + miss_w * model._missing_loss
    accuracy = _top1_acc(pred, targets)
    return loss, accuracy


def compute_loss_multi_label(model, batch, args):
    m0, m1, m2, m3, targets = batch
    mp = mm_base._get_missing_prob(model, args)
    pred, H = model(m0, m1, m2, m3, missing_prob=mp)
    cls_loss = _bce_criterion(pred, targets)

    sim_w = getattr(args, "sim_loss_weight", 0.0)
    if sim_w > 0:
        aff = targets @ targets.t()
        aff = torch.sigmoid(aff)
        aff = 2.0 * aff - 1.0
        H_norm = F.normalize(H, dim=1)
        sim = H_norm @ H_norm.t()
        sim_loss = F.mse_loss(sim, aff)
        loss = cls_loss + sim_w * sim_loss
    else:
        loss = cls_loss

    miss_w = getattr(args, "missing_loss_weight", 0.0)
    if miss_w > 0:
        loss = loss + miss_w * model._missing_loss

    predicted = (torch.sigmoid(pred) > 0.5).float()
    accuracy = (predicted == targets).float().mean()
    return loss, accuracy


def get_data_loaders(args):
    alpha = getattr(args, "dirichlet_alpha", 0.1)
    train_dataset = FedMultiModal4(
        args.dataset_dir, args.dataset_name, transform=None,
        do_iid=args.do_iid, num_clients=args.num_clients,
        train=True, download=True,
        dirichlet_alpha=alpha,
    )
    test_dataset = FedMultiModal4(
        args.dataset_dir, args.dataset_name, transform=None,
        train=False, download=False,
        dirichlet_alpha=alpha,
    )
    train_sampler = FedSampler(
        train_dataset, args.num_workers, args.local_batch_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.train_dataloader_workers,
        pin_memory=True,
    )
    test_batch_size = args.valid_batch_size * args.num_workers
    n_te = len(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.val_dataloader_workers,
        pin_memory=True,
    )
    n_drop = n_te % test_batch_size
    if n_drop:
        print(
            f"Validation drop_last=True: omitting last {n_drop} test samples "
            f"(evaluating {n_te - n_drop}/{n_te}; batch_size={test_batch_size})"
        )
    print(len(train_loader), len(test_loader))
    return train_loader, test_loader


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    print("MY PID:", os.getpid())

    args = parse_args()
    _load_prepare_stats(args)
    print(args)

    if args.model != "SketchFusionB4":
        raise ValueError(
            "mm_train_actionsense_4mod.py only supports --model SketchFusionB4 "
            f"(got {args.model!r})."
        )
    if not getattr(args, "skip_map", False):
        raise ValueError(
            "mm_train_actionsense_4mod requires --skip_map "
            "(retrieval MAP uses the 2-modality extract_fused API only)."
        )

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

    print(
        "SketchFusionB4 config:",
        {
            "mod_dims": tuple(args.mod_dims),
            "feat_dim": args.feat_dim,
            "num_classes": num_classes,
            "dropout": args.mm_dropout,
            "sketch_r": getattr(args, "sketch_r", 4),
            "sketch_c": getattr(args, "sketch_c", 128),
        },
    )
    model = models.SketchFusionB4(
        mod_dims=tuple(args.mod_dims),
        feat_dim=args.feat_dim,
        num_classes=num_classes,
        dropout=args.mm_dropout,
        sketch_r=getattr(args, "sketch_r", 4),
        sketch_c=getattr(args, "sketch_c", 128),
    )

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
        "LR divisor spe*mle={:.3g}".format(_spe0, _mle, _spe0 * _mle)
    )

    grad = get_grad(model, args)
    print("Grad size:", grad.numel())
    print("Total params:", sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))

    mm_base.train(
        model, opt, lr_scheduler, train_loader, test_loader, args,
        writer, loggers=(TableLogger(),), timer=timer,
    )
    model.finalize()
