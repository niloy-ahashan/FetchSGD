import argparse
import os


def args_parser():
    parser = argparse.ArgumentParser(
        description="HybridSketchMFedMC_v2: FetchSGD engine (example-weighted "
        "gradient or delta sketch) + MFedMC client/modality selection."
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument(
        "--dataset",
        choices=["uci_har", "mhealth"],
        default="uci_har",
        help="Which prepared feature cache to load. "
        "uci_har: Acc/Gyro (2 modalities). mhealth: Acc/Gyro/Mag/ECG (4 modalities).",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="",
        help="Directory with data.npz (default: <repo>/datasets/uci_har_mm or "
        "<repo>/datasets/mhealth_mm, matching --dataset).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Defaults to 6 for --dataset uci_har, 12 for --dataset mhealth.",
    )
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.1)
    parser.add_argument(
        "--acc_dim", type=int, default=None,
        help="Defaults to 348 (uci_har) or 177 (mhealth).",
    )
    parser.add_argument(
        "--gyro_dim", type=int, default=None,
        help="Defaults to 213 (uci_har) or 118 (mhealth).",
    )
    parser.add_argument(
        "--mag_dim", type=int, default=None,
        help="mhealth only. Defaults to 118.",
    )
    parser.add_argument(
        "--ecg_dim", type=int, default=None,
        help="mhealth only. Defaults to 43.",
    )
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument("--mm_dropout", type=float, default=0.3)
    parser.add_argument("--sketch_r", type=int, default=4)
    parser.add_argument("--sketch_c", type=int, default=128)
    parser.add_argument(
        "--fusion_mode",
        choices=["sketch", "sum"],
        default="sketch",
        help="Feature fusion before the integrator. "
        "'sketch' = SketchFusionB (Count Sketch then add). "
        "'sum' = IndependentCompression (add refined feat_dim vectors).",
    )

    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="1 = no local SGD (one gradient on global weights, FetchSGD/IC). "
        ">1 = local SGD steps before the compressed upload.",
    )
    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=-1,
        help="Local batch size. -1 uses each client's full dataset.",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=8,
        help="IC val shard size for logged test_loss/test_acc "
        "(unweighted mean over shards of this many examples).",
    )
    parser.add_argument("--lr_scale", type=float, default=0.1)
    parser.add_argument("--pivot_epoch", type=float, default=10)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="IC / FetchSGD default. Applied as wd / n_participating on the "
        "dense gradient (CommEfficient get_grad).",
    )
    parser.add_argument(
        "--upload_object",
        choices=["gradient", "delta"],
        default="gradient",
        help="gradient = sketch the local gradient (IC / FetchSGD mode=sketch). "
        "delta = sketch (global - local) after local SGD. "
        "delta with local_epochs=1 falls back to gradient.",
    )

    parser.add_argument(
        "--mode",
        choices=["sketch"],
        default="sketch",
        help="Gradient compression (FetchSGD Count Sketch).",
    )
    parser.add_argument("--k", type=int, default=20000)
    parser.add_argument("--num_rows", type=int, default=3)
    parser.add_argument("--num_cols", type=int, default=5000)
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--virtual_momentum", type=float, default=0.9)
    parser.add_argument(
        "--error_type",
        choices=["virtual", "none"],
        default="virtual",
    )

    parser.add_argument(
        "--num_select_modalities",
        "--top_shap",
        dest="num_select_modalities",
        type=int,
        default=0,
        help="Keep this many highest-priority modalities per selected client. "
        "0 means all modalities. For 3+ modalities this is the MFedMC top-k.",
    )
    parser.add_argument(
        "--modality_weights",
        nargs="+",
        type=float,
        default=[1 / 3, 1 / 3, 1 / 3],
        help="Weights for SHAP, model_size, and recency in modality priority.",
    )
    parser.add_argument(
        "--random_modality",
        action="store_true",
        help="Select modalities uniformly at random instead of priority scores.",
    )

    parser.add_argument(
        "--client_select",
        choices=["loss", "random"],
        default="loss",
        help="loss = MFedMC (percentage of clients by loss). "
        "random = FetchSGD-style uniform sample.",
    )
    parser.add_argument(
        "--client_select_ratio",
        type=float,
        default=0.5,
        help="Fraction of clients that upload each epoch (e.g. 0.5 = 5/10, "
        "matching SketchFusionB --num_workers 5).",
    )
    parser.add_argument(
        "--client_weights",
        nargs="+",
        type=float,
        default=[1.0, 0.0],
        help="Weights for (loss, staleness) in MFedMC client priority.",
    )
    parser.add_argument(
        "--prefer-higher-loss",
        dest="prefer_higher_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, prefer high-loss clients (MFedMC paper default). "
        "Default is least-loss selection.",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory for the output npz (relative to this package unless absolute).",
    )

    args = parser.parse_args()

    _DATASET_DEFAULTS = {
        "uci_har": {"num_classes": 6, "acc_dim": 348, "gyro_dim": 213},
        "mhealth": {
            "num_classes": 12,
            "acc_dim": 177,
            "gyro_dim": 118,
            "mag_dim": 118,
            "ecg_dim": 43,
        },
    }
    for field, default in _DATASET_DEFAULTS[args.dataset].items():
        if getattr(args, field) is None:
            setattr(args, field, default)

    if not args.dataset_dir:
        here = os.path.abspath(os.path.dirname(__file__))
        repo = os.path.abspath(os.path.join(here, ".."))
        subdir = "uci_har_mm" if args.dataset == "uci_har" else "mhealth_mm"
        args.dataset_dir = os.path.join(repo, "datasets", subdir)
    return args
