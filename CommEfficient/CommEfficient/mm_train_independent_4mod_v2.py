"""
4-modality Independent Compression training, logged like the UCI HAR runner.

Same model, data and FetchSGD training as ``mm_train_independent_4mod.py``
(:class:`IndependentCompression4`, summation fusion, one Count Sketch of the
full gradient), but with the console handling of ``mm_train_independent.py``:
compact per-round table (``iter / test_acc / up_MB / cum_up_MB / ...``),
resolved hyperparameters printed, and the console teed to
``result_updated/IC_<dataset>_mm_<cum_up>MB_..._seed<N>.txt`` so
``summarize_seeds.py --dataset <dataset>`` can average seeds.

``mm_train_independent_4mod.py`` is left unchanged so prior mHealth runs
(``run_mhealth_independent_compression_4mod.sh``) stay reproducible.
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import mm_train as mm_base
from CommEfficient.utils import get_grad
from fed_aggregator import FedModel, FedOptimizer
from mm_train_independent import _Tee, _result_updated_dir, _txt_stem
from mm_train_independent_4mod import (
    IndependentCompression4,
    _load_prepare_stats,
    _warn_if_unused_flags,
    compute_loss_multi_label,
    compute_loss_single_label,
    get_data_loaders,
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


def _run(args):
    _warn_if_unused_flags(args)
    _load_prepare_stats(args)
    print("MY PID:", os.getpid())

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
        "mod_dims": tuple(args.mod_dims),
        "feat_dim": args.feat_dim,
        "num_classes": num_classes,
        "dropout": args.mm_dropout,
    }
    print(f"IndependentCompression4 config: {model_config}")
    print(
        "Fusion: summation (sum of 4 refined modality features).  "
        "Gradient compression: original FetchSGD Count Sketch."
    )
    model = IndependentCompression4(**model_config)

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
        lr_scheduler = LambdaLR(
            opt, lr_lambda=lambda step: lr_schedule(step / spe),
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
    _mle = max(float(getattr(args, "mm_local_epochs", 1.0)), 1.0)
    print(
        "Federated train: steps_per_epoch(spe)={:.0f} (CommEfficient "
        "rounds/epoch; LR divisor=spe). mm_local_epochs={:.3g} is local "
        "SGD per round, not extra uploads.".format(_spe0, _mle)
    )

    grad = get_grad(model, args)
    print("Grad size:", grad.numel())
    print("Total params:", sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))
    print("Hyperparameters (resolved):")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")

    summary = mm_base.train(
        model, opt, lr_scheduler, train_loader, test_loader, args,
        writer, loggers=(TableLogger(),), timer=timer,
        compact_console=True,
    )
    model.finalize()
    if summary and "cum_up_MB" in summary:
        return summary["cum_up_MB"]
    return None


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    args = parse_args()
    args.model = "IndependentCompression4"
    # Retrieval MAP only exists for the 2-modality extract_fused API.
    args.skip_map = True

    log_buf = io.StringIO()
    stdout_orig = sys.stdout
    sys.stdout = _Tee(stdout_orig, log_buf)
    cum_up_mb = None
    try:
        cum_up_mb = _run(args)
    finally:
        sys.stdout = stdout_orig
        updated_dir = _result_updated_dir()
        os.makedirs(updated_dir, exist_ok=True)
        txt_path = os.path.join(updated_dir, _txt_stem(args, cum_up_mb) + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(log_buf.getvalue())
        print(f"Console log saved to {txt_path}")
