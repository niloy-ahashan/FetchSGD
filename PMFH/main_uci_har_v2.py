"""PMFH-v2 on UCI HAR: baseline-style gate + trained classifier head.

A variant of ``main_uci_har.py`` (which runs official PMFH). Differences:

* model ``LayersUCI`` (``layers_uci.py``): per-modality gate shaped like the
  IC / MFedMC v2 / Hybrid v2 ``FeaRefiner``, and a ``head`` Linear(common_dim ->
  classes) on ``nec_vec`` instead of upstream's untrained ``classify``;
* ``LocalUpdateUCI``: upstream ``update_weights_UCCH_new`` verbatim, plus
  ``clf_weight * CrossEntropy(head logits, y)`` in the **global** model's loss.
  Private training, distillation, ConLoss and prototypes are unchanged;
* ``test_acc`` = argmax of the trained head, 8-sample shard mean, i.e. read out
  exactly like the baselines' ``test_acc``.

Data, client exclusion, drop-last-batch-of-1, FastConLoss / anomaly mode,
and comm accounting are the same as ``main_uci_har.py``, whose helpers are
imported. The log is the baseline table: head ``test_acc`` and communication.
This is not official PMFH; label results "PMFH-v2".

``--variant fedhead`` (option B) instead keeps official PMFH untouched
(upstream ``Layers`` for both models, upstream gate) and adds only a federated
classifier head: ``LayersFedHead`` puts Linear(common_dim -> classes) on
``nec_vec.detach()``; the head is trained with CE during local training,
uploaded and FedAvg'd with the global model, and PMFH's own parameters are
clipped exactly as upstream (head clipped separately). PMFH's training is
therefore identical to ``main_uci_har.py``; label results
"PMFH + federated head".

``--variant fedhead --head_grad joint`` (option C) is the same, except the
head's CE gradient also flows into PMFH's network (no ``detach``) and all
global-model parameters are clipped together, like the baselines' jointly
trained classifier. PMFH's features then change; label results
"PMFH + jointly trained head".
"""

import argparse
import copy
import io
import os
import sys
import time

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader

from layers import Layers
from layers_uci import LayersFedHead, LayersUCI
from main import LocalUpdate
from main_uci_har import (
    FastConLoss,
    TableLogger,
    _Tee,
    _load_hybrid_v2_dataset_module,
    _nbytes,
    _one_hot,
    _shard_mean_acc,
)
from utils import (
    affinity_tag_multi,
    agg_func,
    average_weights,
    proto_aggregation,
    seed_setting,
)


class LocalUpdateUCI(LocalUpdate):
    """Upstream ``LocalUpdate.update_weights_UCCH_new`` plus a CE head loss.

    Every line is upstream ``main.py`` except the one marked ``PMFH-v2``.
    """

    def update_weights_UCCH_new(self, args, global_protos, global_avg_protos, model_img, private_modal):
        global_modal = model_img
        device = self.device
        loss_mse = nn.MSELoss().to(device)
        global_modal.to(device)
        private_modal.to(device)
        if args.variant == "fedhead":
            # Head in its own param group: PMFH's lr (1e-5) is too small to train a fresh layer.
            head_ids = {id(p) for p in global_modal.head.parameters()}
            optimizer_global = torch.optim.Adam([
                {"params": [p for p in global_modal.parameters() if id(p) not in head_ids]},
                {"params": list(global_modal.head.parameters()), "lr": args.head_lr},
            ], lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer_global = torch.optim.Adam(global_modal.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer_private = torch.optim.Adam(private_modal.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


        private_loss_sum, private_steps = 0.0, 0
        global_loss_sum, global_steps = 0.0, 0

        private_modal.train()
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(args.train_ep_private):
            for batch_idx, (img_F, txt_F, labels,index) in enumerate(self.dataloader_train):
                _, aff_norm, aff_label = affinity_tag_multi(labels.numpy(), labels.numpy())
                img = Variable(img_F.to(device).float())
                txt = Variable(torch.FloatTensor(txt_F.numpy()).to(device))
                labels = Variable(labels.to(device).float())
                labels = labels.to(device)
                fuseJ, H, pred = private_modal(img, txt)
                H_norm = F.normalize(H)
                clf_loss = loss_mse(torch.sigmoid(pred), labels)
                similarity_loss = loss_mse(H_norm.mm(H_norm.t()), torch.from_numpy(aff_label).float().to(device))
                B = torch.sign(H)
                lb = loss_mse(B, H)
                bal_loss = (torch.sum(B) / B.size(0))
                loss = similarity_loss * args.bb + lb * args.cc
                optimizer_private.zero_grad()
                loss.backward()
                clip_grad_norm_(private_modal.parameters(), 1.)
                optimizer_private.step()
                private_loss_sum += loss.item()
                private_steps += 1

        global_modal.train()
        private_modal.eval()

        for epoch in range(args.train_ep):
            for batch_idx, (img_F,txt_F,labels,index) in enumerate(self.dataloader_train):
                _, aff_norm, aff_label = affinity_tag_multi(labels.numpy(), labels.numpy())
                img = Variable(img_F.to(device))
                txt = Variable(torch.FloatTensor(txt_F.numpy()).to(device))
                labels = Variable(labels.to(device).float())

                bsz = labels.shape[0]

                fuseJg, Hg, predg = global_modal(img, txt)
                Hg_norm = F.normalize(Hg)
                clf_loss = loss_mse(torch.sigmoid(predg), labels)
                Lr = loss_mse(Hg_norm.mm(Hg_norm.t()), torch.from_numpy(aff_label).float().to(device))
                Bg = torch.sign(Hg)
                lb = loss_mse(Hg, Bg)
                bal_loss = (torch.sum(Bg) / Bg.size(0))
                loss = Lr * args.bb + lb * args.cc

                kd_loss = 0
                loss_cont = 0
                if len(global_protos)!=0:
                    fuseJp, private_modal_features, _ = private_modal(img, txt)
                    kd_loss = loss_mse(Hg, private_modal_features)

                if len(global_protos) == args.num_users:
                    fuseJp, private_modal_features, _ = private_modal(img, txt)
                    features = fuseJp.unsqueeze(2)
                    for i in range(args.num_users):
                        for label in global_avg_protos.keys():
                            if label not in global_protos[i].keys():
                                global_protos[i][label] = global_avg_protos[label]
                        loss_cont += self.criterion_CL(features, labels, global_protos[i])

                loss = loss + kd_loss * args.ee + loss_cont * args.ff
                # PMFH-v2 / fedhead: trained classifier head, CE like IC / MFedMC v2 / Hybrid v2.
                loss = loss + args.clf_weight * F.cross_entropy(predg, labels.argmax(dim=1))
                optimizer_global.zero_grad()
                loss.backward()
                if args.variant == "fedhead" and args.head_grad == "detached":
                    # PMFH's parameters clipped exactly as upstream; the head on its own.
                    clip_grad_norm_([p for p in global_modal.parameters() if id(p) not in head_ids], 1.)
                    clip_grad_norm_(global_modal.head.parameters(), 1.)
                else:
                    clip_grad_norm_(global_modal.parameters(), 1.)
                optimizer_global.step()
                global_loss_sum += loss.item()
                global_steps += 1

        with torch.no_grad():
            private_modal.eval()
            private_modal.to(device)

            proj_bank_fuse = []
            n_samples = 0
            for bs, (images, texts, labels, indd) in enumerate(self.dataloader_train):
                img = Variable(images.to(self.device))
                txt = Variable(torch.FloatTensor(texts.numpy()).to(self.device))
                labels = Variable(labels.to(device).float())

                if n_samples >= self.m_size:
                    break
                fuseJ, H, _ = private_modal(img, txt)
                H = F.normalize(H)
                proj_bank_fuse.append(H)
                n_samples += len(indd)

            proj_bank_img = torch.cat(proj_bank_fuse, dim=0).contiguous()
            if n_samples > self.m_size:
                proj_bank_img = proj_bank_img[:self.m_size]

            local_protos = {}
            for i in range(args.classes):
                local_protos[i] = proj_bank_img[proj_bank_img.shape[0] // args.classes * i: proj_bank_img.shape[0] // args.classes * (i+1)].mean(0)
        private_loss = private_loss_sum / max(private_steps, 1)
        global_loss = global_loss_sum / max(global_steps, 1)
        return global_modal.cpu().state_dict(), private_modal.cpu().state_dict(), local_protos, private_loss, global_loss


def args_parser():
    parser = argparse.ArgumentParser()
    # data / federation
    parser.add_argument("--dataset_dir", type=str,
                        default=os.path.join(ROOT, "datasets", "uci_har_mm"))
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    # PMFH (same names and defaults as main_uci_har.py)
    parser.add_argument("--mlpdrop", type=float, default=0.01)
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--train_ep", type=int, default=10)
    parser.add_argument("--train_ep_private", type=int, default=10)
    parser.add_argument("--bit", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_dim", type=int, default=348, help="Acc feature dim")
    parser.add_argument("--text_dim", type=int, default=213, help="Gyro feature dim")
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--common_dim", type=int, default=128)
    parser.add_argument("--aa", type=float, default=0.000001)
    parser.add_argument("--bb", type=float, default=0.01)
    parser.add_argument("--cc", type=float, default=0.001)
    parser.add_argument("--dd", type=float, default=1)
    parser.add_argument("--ee", type=float, default=0.001)
    parser.add_argument("--ff", type=float, default=1)
    # PMFH-v2 / fedhead
    parser.add_argument("--variant", type=str, default="v2", choices=["v2", "fedhead"],
                        help="v2: baseline gate + jointly trained head; "
                             "fedhead: official PMFH + federated head on detached nec_vec")
    parser.add_argument("--head_grad", type=str, default="detached", choices=["detached", "joint"],
                        help="fedhead only: detached = head cannot change PMFH (option B); "
                             "joint = head's CE also trains PMFH's network (option C)")
    parser.add_argument("--head_lr", type=float, default=1e-3,
                        help="fedhead only: Adam lr of the classifier head (PMFH keeps --learning_rate)")
    parser.add_argument("--clf_weight", type=float, default=1.0,
                        help="weight of the CrossEntropy head loss added to the global model")
    # evaluation
    parser.add_argument("--valid_batch_size", type=int, default=8,
                        help="shard size for the IC-style shard-mean accuracy")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--slow_upstream", action="store_true",
                        help="use upstream's loop ConLoss and autograd anomaly mode (same numbers, much slower)")
    return parser.parse_args()


def _prefix(args):
    if args.variant == "fedhead":
        return "PMFHjh" if args.head_grad == "joint" else "PMFHfh"
    return "PMFHv2"


def _txt_stem(args, cum_up_mb, n_active):
    dataset = os.path.basename(os.path.normpath(args.dataset_dir or "unknown"))
    up_str = f"{cum_up_mb:.2f}MB" if cum_up_mb is not None else "NAMB"
    prefix = _prefix(args)
    return (
        f"{prefix}_{dataset}_mm_{up_str}_dirichlet{args.dirichlet_alpha:g}_"
        f"it{args.rounds}_bit{args.bit}_h{args.hidden_dim}_c{args.common_dim}_"
        f"clf{args.clf_weight:g}_lr{args.learning_rate:g}_"
        f"E{args.train_ep}_Ep{args.train_ep_private}_"
        f"clients{args.num_clients}_workers{n_active}_seed{args.seed}"
    )


@torch.no_grad()
def _head_logits(model, xs_img, xs_txt, device, batch_size=1024):
    """Trained head on nec_vec (LayersUCI.forward's third output)."""
    model.eval()
    model.to(device)
    out = []
    for start in range(0, len(xs_img), batch_size):
        img = torch.as_tensor(xs_img[start:start + batch_size], dtype=torch.float32, device=device)
        txt = torch.as_tensor(xs_txt[start:start + batch_size], dtype=torch.float32, device=device)
        _, _, logits = model(img, txt)
        out.append(logits)
    return torch.cat(out, dim=0)


def _run(args):
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"Requested {args.device} but CUDA is unavailable — using {device}")
    args.device = device
    seed_setting(args.seed)
    if not args.slow_upstream:
        # Upstream calls torch.autograd.set_detect_anomaly(True) every local update.
        # Anomaly mode only adds NaN checks / stack traces to backward; it changes no values.
        torch.autograd.set_detect_anomaly = lambda *a, **k: None
    timer_start = time.perf_counter()

    print("Loading SketchFusionB UCI HAR split (Acc / Gyro feature vectors)...")
    ds = _load_hybrid_v2_dataset_module()
    clients, global_test, meta = ds.load_uci_har_mm(
        args.dataset_dir,
        num_clients=args.num_clients,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=42,
    )
    acc_dim, gyro_dim = meta["mod_dims"]
    if (acc_dim, gyro_dim) != (args.image_dim, args.text_dim):
        print(f"Warning: data dims Acc={acc_dim}, Gyro={gyro_dim} differ from args. Using data dims.")
        args.image_dim, args.text_dim = acc_dim, gyro_dim
    if meta["num_classes"] != args.classes:
        args.classes = int(meta["num_classes"])

    # Clients too small for PMFH's positional per-class prototypes (n // classes == 0).
    n_per = [int(len(c["y"])) for c in clients]
    active = [cid for cid, n in enumerate(n_per) if n >= args.classes]
    excluded = [cid for cid, n in enumerate(n_per) if n < args.classes]
    if excluded:
        print(
            f"Excluding clients {excluded} (samples {[n_per[c] for c in excluded]} < "
            f"classes={args.classes}): PMFH prototypes would be NaN."
        )
    n_active = len(active)
    args.num_users = n_active  # upstream enables the contrastive loss when all users have protos

    # Upstream datasetall layout; client j owns a contiguous index range.
    train_x = np.concatenate([clients[c]["xs"][0] for c in active]).astype(np.float32)
    train_y = np.concatenate([clients[c]["xs"][1] for c in active]).astype(np.float32)
    train_L = _one_hot(np.concatenate([clients[c]["y"] for c in active]), args.classes)
    empty_x = np.empty((0, args.image_dim), dtype=np.float32)
    empty_y = np.empty((0, args.text_dim), dtype=np.float32)
    empty_L = np.empty((0, args.classes), dtype=np.int64)
    datasetall = [train_L, train_x, train_y, empty_L, empty_x, empty_y, empty_L, empty_x, empty_y]
    offsets = np.cumsum([0] + [n_per[c] for c in active])
    client_idcs_train = [list(range(offsets[j], offsets[j + 1])) for j in range(n_active)]
    client_idcs_empty = [[] for _ in range(n_active)]

    q_img, q_txt = global_test["xs"][0], global_test["xs"][1]
    q_lab = np.asarray(global_test["y"], dtype=np.int64)

    args.img_hidden_dim = [args.image_dim, args.hidden_dim, args.common_dim]
    args.txt_hidden_dim = [args.text_dim, args.hidden_dim, args.common_dim]
    if args.variant == "fedhead":
        # Same construction order and classes as main_uci_har.py; only the global model gets the head.
        private_modal_list = [Layers(args=args) for _ in range(n_active)]
        global_modal = LayersFedHead(args=args)
    else:
        private_modal_list = [LayersUCI(args=args) for _ in range(n_active)]
        global_modal = LayersUCI(args=args)

    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(HERE, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    model_bytes = _nbytes(global_modal.state_dict())
    n_params = sum(p.numel() for p in global_modal.parameters())
    print("Finished initializing in {:.2f} seconds".format(time.perf_counter() - timer_start))
    print("Hyperparameters (resolved):")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    if args.variant == "fedhead":
        print(f"PMFH + federated head: official Layers bit={args.bit} img={args.img_hidden_dim} "
              f"txt={args.txt_hidden_dim}, head={args.common_dim}->{args.classes} on "
              f"{'nec_vec (joint)' if args.head_grad == 'joint' else 'detached nec_vec'}")
    else:
        print(f"PMFH-v2 LayersUCI: bit={args.bit} img={args.img_hidden_dim} txt={args.txt_hidden_dim} "
              f"gate={args.common_dim}->{args.common_dim}->{args.common_dim} head={args.common_dim}->{args.classes}")
    print(f"Total params: {n_params}  (state_dict bytes per model: {model_bytes})")
    print(f"Active clients: {active} (n={n_active}), samples: {[n_per[c] for c in active]}")
    if args.variant == "fedhead":
        if args.head_grad == "joint":
            print(f"Engine: PMFH training + jointly trained federated head (CE x {args.clf_weight:g}, head lr "
                  f"{args.head_lr:g}, CE gradient also trains PMFH; head uploaded and FedAvg'd with the global model)")
        else:
            print(f"Engine: official PMFH training + federated head (CE x {args.clf_weight:g}, head lr "
                  f"{args.head_lr:g}, gradient stops at nec_vec; head uploaded and FedAvg'd with the global model)")
    else:
        print(f"Engine: PMFH-v2 (upstream LocalUpdate + CE head x {args.clf_weight:g} on global model "
              f"+ FedAvg of global model + prototype exchange)")
    print("Speed: " + ("upstream ConLoss loop + anomaly mode" if args.slow_upstream
                       else "FastConLoss (same values) + anomaly mode off"))
    print(f"Eval: test_acc = trained head argmax on test ({len(q_lab)}); shard={args.valid_batch_size}")

    logger = TableLogger()
    history = {k: [] for k in (
        "test_acc", "private_loss", "global_loss",
        "upload_bytes", "download_bytes",
    )}
    global_avg_protos = {}
    global_protos = {}
    local_protos = {}
    total_upload = 0.0
    total_download = 0.0
    train_start = time.perf_counter()
    prev_elapsed = 0.0

    for rnd in range(args.rounds):
        # Download billed before local training; round 1 is shared init (IC convention).
        if rnd == 0:
            download_bytes = 0
        else:
            download_bytes = n_active * (
                _nbytes(global_modal.state_dict())
                + _nbytes(global_protos)
                + _nbytes(global_avg_protos)
            )

        local_weights = []
        private_weights = []
        upload_bytes = 0
        total_private_loss = 0.0
        total_global_loss = 0.0

        for idx in range(n_active):
            local_model = LocalUpdateUCI(
                args=args, id=idx, dataset=datasetall, idxs=client_idcs_train[idx],
                idcs_retrieval=client_idcs_empty, idcs_query=client_idcs_empty,
            )
            if not args.slow_upstream:
                local_model.criterion_CL = FastConLoss(temperature=0.07)
            # Same loader as upstream, but never leave a final batch of 1 (BatchNorm1d).
            train_ds = local_model.dataloader_train.dataset
            local_model.dataloader_train = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                drop_last=(len(train_ds) % args.batch_size == 1),
            )
            w_global, w_private, protos, private_loss, global_loss = local_model.update_weights_UCCH_new(
                args, global_protos, global_avg_protos,
                model_img=global_modal, private_modal=private_modal_list[idx],
            )
            total_private_loss += private_loss
            total_global_loss += global_loss

            agg_protos = agg_func(protos)
            local_protos[idx] = copy.deepcopy(agg_protos)
            local_weights.append(w_global)
            private_weights.append(w_private)
            upload_bytes += _nbytes(w_global) + _nbytes(agg_protos)

        local_weights_list = average_weights(local_weights)
        global_avg_protos = proto_aggregation(local_protos)
        global_protos = copy.deepcopy(local_protos)
        for idx in range(n_active):
            private_modal_list[idx].load_state_dict(private_weights[idx])
        global_modal.load_state_dict(local_weights_list[0], strict=True)

        # ---- eval: trained head, read out like IC / MFedMC v2 / Hybrid v2 ----
        pred = _head_logits(global_modal, q_img, q_txt, device).argmax(dim=1).cpu().numpy()
        test_acc = _shard_mean_acc((pred == q_lab).astype(np.float64), args.valid_batch_size)
        global_modal.cpu()

        elapsed_s = time.perf_counter() - train_start
        round_s = elapsed_s - prev_elapsed
        prev_elapsed = elapsed_s
        upload_mb = upload_bytes / (1000 * 1000)
        download_mb = download_bytes / (1000 * 1000)
        total_upload += upload_mb
        total_download += download_mb

        logger.append({
            "iter": rnd + 1,
            "test_acc": test_acc,
            "up_MB": upload_mb,
            "cum_up_MB": total_upload,
            "down_MB": download_mb,
            "cum_down_MB": total_download,
            "round_s": round_s,
            "time_s": elapsed_s,
        })
        history["test_acc"].append(test_acc)
        history["private_loss"].append(total_private_loss / n_active)
        history["global_loss"].append(total_global_loss / n_active)
        history["upload_bytes"].append(upload_bytes)
        history["download_bytes"].append(download_bytes)

    print("Total Download (MiB): {:0.2f}".format(total_download))
    print("Total Upload (MiB): {:0.2f}".format(total_upload))
    print("Avg Download Per Client: {:0.2f}".format(total_download / max(args.num_clients, 1)))
    print("Avg Upload Per Client: {:0.2f}".format(total_upload / max(args.num_clients, 1)))
    print("Cumulative train comm (download+upload): {:.4f} MiB".format(total_download + total_upload))
    print(
        f"Final test accuracy: {history['test_acc'][-1]:.4f} "
        f"({100.0 * history['test_acc'][-1]:.2f}%)"
    )

    prefix = _prefix(args)
    file_name = os.path.join(results_dir, f"{prefix}_UCI_HAR_bit{args.bit}_h{args.hidden_dim}.npz")
    np.savez(
        file_name,
        test_acc=np.asarray(history["test_acc"]),
        private_loss=np.asarray(history["private_loss"]),
        global_loss=np.asarray(history["global_loss"]),
        upload_bytes=np.asarray(history["upload_bytes"]),
        download_bytes=np.asarray(history["download_bytes"]),
        total_upload_mib=total_upload,
        total_download_mib=total_download,
        samples_per_client=np.asarray(n_per),
        active_clients=np.asarray(active),
        excluded_clients=np.asarray(excluded, dtype=np.int64),
        dataset_dir=np.array(meta["dataset_dir"]),
        bit=np.array(args.bit),
        hidden_dim=np.array(args.hidden_dim),
        common_dim=np.array(args.common_dim),
        clf_weight=np.array(args.clf_weight),
        variant=np.array(args.variant),
        head_lr=np.array(args.head_lr),
        head_grad=np.array(args.head_grad),
    )
    print(f"Results saved to {file_name}")
    return total_upload, n_active


def main():
    args = args_parser()
    log_buf = io.StringIO()
    stdout_orig = sys.stdout
    sys.stdout = _Tee(stdout_orig, log_buf)
    cum_up_mb, n_active = None, args.num_clients
    try:
        cum_up_mb, n_active = _run(args)
    finally:
        sys.stdout = stdout_orig
        updated_dir = os.path.join(HERE, "result_updated")
        os.makedirs(updated_dir, exist_ok=True)
        txt_path = os.path.join(updated_dir, _txt_stem(args, cum_up_mb, n_active) + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(log_buf.getvalue())
        print(f"Console log saved to {txt_path}")


if __name__ == "__main__":
    main()
