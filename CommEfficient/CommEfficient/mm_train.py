"""
Multimodal FetchSGD training script.

Uses the PFMH-style feature fusion model (FeaExtractor → FeaRefiner →
Integrate → Classifier) with the FetchSGD sketch-based communication
pipeline from CommEfficient.

Adapted from cv_train.py — the only structural changes are:
  • Model:   MultiModalNet (two-branch fusion) instead of FixupResNet9
  • Dataset: FedMultiModal (returns img_feat, txt_feat, target per sample)
  • Loss:    Unpacks three tensors instead of two; supports both
             single-label (CrossEntropy) and multi-label (BCEWithLogits)
Everything else (FedModel, FedOptimizer, sketch, virtual momentum,
error feedback) is left untouched.
"""

from CommEfficient.utils import get_grad
import torch
import numpy as np
import math
import os
import time
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import h5py

import models
from fed_aggregator import FedModel, FedOptimizer
from utils import make_logdir, Timer, TableLogger, parse_args
from utils import PiecewiseLinear, steps_per_epoch, set_param_vec
from data_utils import FedSampler, FedMultiModal

import torch.multiprocessing as multiprocessing


# ------------------------------------------------------------------
# Loss / metric helpers
# ------------------------------------------------------------------

# ---- single-label (CrossEntropy) ----
_ce_criterion = torch.nn.CrossEntropyLoss(reduction="mean")

class _TopOneAccuracy(torch.nn.Module):
    def forward(self, logits, target):
        return (logits.max(dim=1)[1] == target).float().mean()

_top1_acc = _TopOneAccuracy()


def _get_missing_prob(model, args):
    """Return missing_prob if training, 0 otherwise."""
    if model.training:
        return getattr(args, 'missing_prob', 0.0)
    return 0.0


def compute_loss_single_label(model, batch, args):
    img_feats, txt_feats, targets = batch
    mp = _get_missing_prob(model, args)
    pred, _H = model(img_feats, txt_feats, missing_prob=mp)
    loss = _ce_criterion(pred, targets)

    miss_w = getattr(args, 'missing_loss_weight', 0.0)
    if miss_w > 0:
        loss = loss + miss_w * model._missing_loss

    accuracy = _top1_acc(pred, targets)
    return loss, accuracy


# ---- multi-label (BCE with logits + optional similarity + missing loss) ----
_bce_criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")


def compute_loss_multi_label(model, batch, args):
    img_feats, txt_feats, targets = batch
    mp = _get_missing_prob(model, args)
    pred, H = model(img_feats, txt_feats, missing_prob=mp)

    cls_loss = _bce_criterion(pred, targets)

    sim_w = getattr(args, 'sim_loss_weight', 0.0)
    if sim_w > 0:
        # PFMH-style affinity: sigmoid-scaled shared-label count → [-1, 1]
        aff = targets @ targets.t()
        aff = torch.sigmoid(aff)
        aff = 2.0 * aff - 1.0

        H_norm = F.normalize(H, dim=1)
        sim = H_norm @ H_norm.t()
        sim_loss = F.mse_loss(sim, aff)

        loss = cls_loss + sim_w * sim_loss
    else:
        loss = cls_loss

    miss_w = getattr(args, 'missing_loss_weight', 0.0)
    if miss_w > 0:
        loss = loss + miss_w * model._missing_loss

    predicted = (torch.sigmoid(pred) > 0.5).float()
    accuracy = (predicted == targets).float().mean()
    return loss, accuracy


# ------------------------------------------------------------------
# MAP evaluation  (cosine similarity on fused features H)
# ------------------------------------------------------------------

_retrieval_cache = {}


def _load_retrieval_data(args):
    """Load query (I_te) and database (I_db) features from the data file."""
    cache_key = args.dataset_dir
    if cache_key in _retrieval_cache:
        return _retrieval_cache[cache_key]

    data_dir = args.dataset_dir
    h5_names = (
        "mir_cnn_twt.mat", "nus_cnn_twt.mat", "coco_cnn_twt_2014.mat",
        "data.mat", "mir.h5", "wiki.h5", "nus.h5", "data.h5",
    )
    data_path = None
    for name in h5_names:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            data_path = p
            break
    if data_path is None:
        raise FileNotFoundError(f"No HDF5 / .mat data file in {data_dir}")

    with h5py.File(data_path, "r") as f:
        qu_img = np.array(f["I_te"]).T.astype(np.float32)
        qu_txt = np.array(f["T_te"]).T.astype(np.float32)
        qu_L   = np.array(f["L_te"]).T.astype(np.float32)
        db_img = np.array(f["I_db"]).T.astype(np.float32)
        db_txt = np.array(f["T_db"]).T.astype(np.float32)
        db_L   = np.array(f["L_db"]).T.astype(np.float32)

    result = (qu_img, qu_txt, qu_L, db_img, db_txt, db_L)
    _retrieval_cache[cache_key] = result
    print(f"Loaded retrieval data: query={qu_img.shape[0]}, "
          f"database={db_img.shape[0]}")
    return result


@torch.no_grad()
def _extract_fused_batched(model, img, txt, device, batch_size=256):
    """Run model.extract_fused in batches and return concatenated H."""
    all_H = []
    for i in range(0, len(img), batch_size):
        img_b = torch.as_tensor(img[i:i+batch_size]).to(device)
        txt_b = torch.as_tensor(txt[i:i+batch_size]).to(device)
        H = model.extract_fused(img_b, txt_b)
        all_H.append(H.cpu())
    return torch.cat(all_H, dim=0)


def _calculate_map(qu_H, db_H, qu_L, db_L, topk=None):
    """
    Mean Average Precision.

    Relevance: two samples share at least one label  (L_q · L_d > 0).
    Ranking:   descending cosine similarity of fused features.
    """
    sim = qu_H @ db_H.T                     # (Q, D)
    num_query = qu_L.shape[0]
    total_ap = 0.0

    for i in range(num_query):
        gnd = (qu_L[i] @ db_L.T > 0).astype(np.float32)
        tsum = gnd.sum()
        if tsum == 0:
            continue

        ind = np.argsort(-sim[i])            # descending similarity
        gnd = gnd[ind]

        if topk is not None:
            gnd = gnd[:topk]
            tsum = min(tsum, float(topk))

        positions = np.where(gnd == 1)[0] + 1.0   # 1-indexed ranks
        count = np.arange(1, len(positions) + 1, dtype=np.float64)
        ap = np.mean(count / positions)
        total_ap += ap

    return total_ap / num_query


def evaluate_map(fed_model, args, topk=None):
    """
    Compute MAP after each epoch.

    Syncs model weights from the parameter server, extracts fused
    features H for query and database sets, ranks by cosine similarity.
    """
    import fed_aggregator
    set_param_vec(fed_model.model,
                  fed_aggregator.g_ps_weights.cpu())
    model = fed_model.model
    was_training = model.training
    model.eval()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    qu_img, qu_txt, qu_L, db_img, db_txt, db_L = \
        _load_retrieval_data(args)

    qu_H = _extract_fused_batched(model, qu_img, qu_txt, device)
    db_H = _extract_fused_batched(model, db_img, db_txt, device)

    qu_H = F.normalize(qu_H, dim=1).numpy()
    db_H = F.normalize(db_H, dim=1).numpy()

    map_val = _calculate_map(qu_H, db_H, qu_L, db_L, topk=topk)

    model.cpu()
    model.train(was_training)
    return map_val


# ------------------------------------------------------------------
# Training loops (mirrors cv_train.py)
# ------------------------------------------------------------------
def train(model, opt, lr_scheduler, train_loader, test_loader,
          args, writer, loggers=(), timer=None):
    timer = timer or Timer()

    total_download = 0
    total_upload = 0

    comm_tracker = None
    max_comm = getattr(args, "max_comm_megabytes", None)
    if max_comm is not None and max_comm > 0:
        comm_tracker = {
            "used_mb": 0.0,
            "budget_mb": float(max_comm),
            "exhausted": False,
            "train_batches": 0,
        }
        print(
            f"Communication budget: {max_comm} MiB (train download+upload, "
            f"cumulative). Stopping when reached."
        )

    target_acc_hit = False

    for epoch in range(math.ceil(args.num_epochs)):
        epoch_fraction = (args.num_epochs - epoch
                          if epoch == math.ceil(args.num_epochs) - 1
                          else 1)

        train_loss, train_acc, download, upload = run_batches(
            model, opt, lr_scheduler, train_loader,
            True, epoch_fraction, args, comm_tracker=comm_tracker,
        )
        if train_loss is np.nan:
            print("TERMINATING TRAINING DUE TO NAN LOSS")
            return

        train_time = timer()
        download_mb = download.sum().item() / (1000 * 1000)
        upload_mb = upload.sum().item() / (1000 * 1000)
        total_download += download_mb
        total_upload += upload_mb

        test_loss, test_acc, _, _ = run_batches(
            model, None, None, test_loader, False, 1, args,
        )
        test_time = timer()

        if getattr(args, "skip_map", False):
            map_val = None
            map_time = 0.0
        else:
            map_val = evaluate_map(model, args)
            map_time = timer()

        try:
            rounded_down = round(download_mb)
        except Exception:
            rounded_down = np.nan
        try:
            rounded_up = round(upload_mb)
        except Exception:
            rounded_up = np.nan

        if lr_scheduler is not None:
            lr = lr_scheduler.get_last_lr()[0]
        else:
            lr = args.lr if args.lr is not None else args.lr_scale

        comm_cum_mib = total_download + total_upload
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            # "train_time": train_time,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "down (MiB)": rounded_down,
            "up (MiB)": rounded_up,
            "total_time": timer.total_time,
        }
        if map_val is not None:
            row["MAP"] = round(map_val, 4)
        summary = row
        for logger in loggers:
            logger.append(summary)
        if args.use_tensorboard and writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Acc/test", test_acc, epoch)
            if map_val is not None:
                writer.add_scalar("MAP", map_val, epoch)
            writer.add_scalar("Lr", lr, epoch)
            writer.add_scalar("Comm/cumulative_mib", comm_cum_mib, epoch)

        tgt = getattr(args, "target_test_acc", None)
        if tgt is not None and not target_acc_hit and test_acc >= tgt:
            target_acc_hit = True
            print("=" * 64)
            print(
                "MFedMC-style communication overhead (Table 2(ii) spirit): "
                "cumulative train download+upload = {:.4f} MiB "
                "when test_acc first reached ≥ {:.6f} (epoch {})".format(
                    comm_cum_mib, tgt, epoch + 1,
                )
            )
            print(
                "  (Accounting: fed_aggregator per-epoch train comm; "
                "binary MiB; see paper for their exact compressor.)"
            )
            print("=" * 64)
            if getattr(args, "stop_on_target_acc", False):
                break

        if comm_tracker and comm_tracker.get("exhausted"):
            u = comm_tracker["used_mb"]
            b = comm_tracker["budget_mb"]
            tb = comm_tracker["train_batches"]
            print("=" * 64)
            print(
                "COMM BUDGET STOP — target {:g} MiB; used {:.6f} MiB over "
                "{} training batches".format(b, u, tb)
            )
            print(
                "Test accuracy at stop (comm-limited run): "
                "{:.6f}  (fraction; ×100 for %)".format(test_acc)
            )
            print("=" * 64)
            break

    print("Total Download (MiB): {:0.2f}".format(total_download))
    print("Total Upload (MiB): {:0.2f}".format(total_upload))
    print("Avg Download Per Client: {:0.2f}".format(
        total_download / train_loader.dataset.num_clients
    ))
    print("Avg Upload Per Client: {:0.2f}".format(
        total_upload / train_loader.dataset.num_clients
    ))
    print(
        "Cumulative train comm (download+upload): {:.4f} MiB".format(
            total_download + total_upload
        )
    )
    tgt = getattr(args, "target_test_acc", None)
    if tgt is not None and not target_acc_hit:
        print(
            "Note: target_test_acc {:.4f} was never reached; overhead line "
            "above not triggered.".format(tgt)
        )
    return summary


def run_batches(model, opt, lr_scheduler, loader,
                training, epoch_fraction, args, comm_tracker=None):
    if not training and epoch_fraction != 1:
        raise ValueError("Must do full epochs for val")
    if epoch_fraction > 1 or epoch_fraction <= 0:
        raise ValueError(f"Invalid epoch_fraction {epoch_fraction}")

    model.train(training)
    losses, accs = [], []
    client_download = client_upload = None

    if training:
        num_clients = loader.dataset.num_clients
        client_download = torch.zeros(num_clients)
        client_upload = torch.zeros(num_clients)
        spe = steps_per_epoch(args.local_batch_size, loader.dataset,
                              args.num_workers)
        mle = max(float(getattr(args, "mm_local_epochs", 1.0)), 1e-8)
        step_limit = spe * epoch_fraction * mle

        i = 0
        done = False
        while not done:
            for batch in loader:
                if i > step_limit:
                    done = True
                    break

                opt.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                if args.local_batch_size == -1:
                    expected = args.num_workers
                    if torch.unique(batch[0]).numel() < expected:
                        msg = "SKIPPING BATCH: NOT ENOUGH CLIENTS ({} < {})"
                        print(msg.format(torch.unique(batch[0]).numel(),
                                          expected))
                        continue
                else:
                    expected_numel = args.num_workers * args.local_batch_size
                    if batch[0].numel() < expected_numel:
                        msg = "SKIPPING BATCH: NOT ENOUGH DATA ({} < {})"
                        print(msg.format(batch[0].numel(), expected_numel))
                        continue

                loss, acc, download, upload = model(batch)
                if np.any(np.isnan(loss)):
                    print(f"NAN LOSS ({np.mean(loss)}), TERMINATING")
                    return np.nan, np.nan, np.nan, np.nan

                client_download += download
                client_upload += upload
                losses.extend(loss)
                accs.extend(acc)
                i += 1

                if comm_tracker is not None and not comm_tracker.get(
                        "exhausted", False):
                    batch_bytes = float(
                        download.sum().item() + upload.sum().item())
                    batch_mb = batch_bytes / (1024.0 * 1024.0)
                    comm_tracker["used_mb"] += batch_mb
                    comm_tracker["train_batches"] += 1
                    if comm_tracker["used_mb"] >= comm_tracker["budget_mb"]:
                        comm_tracker["exhausted"] = True
                        done = True
                        print(
                            "max_comm_megabytes: budget {:.6g} MiB reached "
                            "(cumulative train comm {:.6f} MiB). "
                            "Stopping further training batches.".format(
                                comm_tracker["budget_mb"],
                                comm_tracker["used_mb"],
                            ))
                        break

                if args.do_test:
                    done = True
                    break
    else:
        for batch in loader:
            if batch[0].numel() < args.valid_batch_size:
                print("SKIPPING VAL BATCH: TOO SMALL")
                continue
            loss, acc = model(batch)
            losses.extend(loss)
            accs.extend(acc)
            if args.do_test:
                break

    return np.mean(losses), np.mean(accs), client_download, client_upload


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def get_data_loaders(args):
    alpha = getattr(args, 'dirichlet_alpha', 0.1)
    train_dataset = FedMultiModal(
        args.dataset_dir, args.dataset_name, transform=None,
        do_iid=args.do_iid, num_clients=args.num_clients,
        train=True, download=True,
        dirichlet_alpha=alpha,
    )
    test_dataset = FedMultiModal(
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
    # drop_last: the last batch is often smaller than test_batch_size; the
    # val loop skips any batch with fewer than valid_batch_size samples
    # (batch[0] is client-id tensor of shape (B,) → numel()==B) to avoid
    # FedModel._call_val failing when one shard cannot be split across
    # num_workers processes.  drop_last omits that partial batch instead
    # of printing "SKIPPING VAL BATCH: TOO SMALL" every epoch.
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
        print(f"Validation drop_last=True: omitting last {n_drop} test samples "
              f"(evaluating {n_te - n_drop}/{n_te}; batch_size={test_batch_size})")

    print(len(train_loader), len(test_loader))
    return train_loader, test_loader


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    print("MY PID:", os.getpid())

    args = parse_args()
    print(args)

    timer = Timer()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- data ----
    train_loader, test_loader = get_data_loaders(args)

    # auto-detect multi-label from the dataset
    multi_label = train_loader.dataset.multi_label
    if multi_label:
        print("Detected MULTI-LABEL dataset — using BCEWithLogitsLoss")
        compute_loss_fn = compute_loss_multi_label
    else:
        print("Detected single-label dataset — using CrossEntropyLoss")
        compute_loss_fn = compute_loss_single_label

    # override num_classes from dataset metadata if not set explicitly
    if args.num_classes is None and train_loader.dataset.mm_num_classes:
        args.num_classes = train_loader.dataset.mm_num_classes
    num_classes = args.num_classes

    # ---- model ----
    model_config = {
        "img_dim": args.img_dim,
        "txt_dim": args.txt_dim,
        "feat_dim": args.feat_dim,
        "num_classes": num_classes,
        "dropout": args.mm_dropout,
        "sketch_r": getattr(args, "sketch_r", 4),
        "sketch_c": getattr(args, "sketch_c", 128),
        "lstm_hidden": getattr(args, "lstm_hidden", 128),
    }
    ModelClass = getattr(models, args.model)
    print(f"{args.model} config: {model_config}")
    model = ModelClass(**model_config)

    if getattr(args, "mm_sketch_separated", False):
        from mm_sketch_separated import store_index_maps_on_args
        store_index_maps_on_args(model, args)
        print(f"Modality-separated sketch: d_img={args._mm_sep_d_img}, "
              f"d_txt={args._mm_sep_d_txt}")

    param_groups = model.parameters()
    opt = optim.SGD(param_groups, lr=1)

    model = FedModel(model, compute_loss_fn, args, compute_loss_fn)
    opt = FedOptimizer(opt, args)

    # ---- LR schedule ----
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

    # ---- output ----
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

    # ---- train ----
    train(
        model, opt, lr_scheduler, train_loader, test_loader, args,
        writer, loggers=(TableLogger(),), timer=timer,
    )
    model.finalize()
