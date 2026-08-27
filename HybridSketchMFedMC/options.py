import argparse
import os


def args_parser():
    parser = argparse.ArgumentParser(
        description="Hybrid SketchFusionB + MFedMC for UCI HAR "
        "(fusion sketch B, FetchSGD gradient sketch, client/modality selection)."
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="",
        help="Directory with data.npz (default: <repo>/datasets/uci_har_mm).",
    )
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.1)
    parser.add_argument("--acc_dim", type=int, default=348)
    parser.add_argument("--gyro_dim", type=int, default=213)
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument("--mm_dropout", type=float, default=0.3)
    parser.add_argument("--sketch_r", type=int, default=4)
    parser.add_argument("--sketch_c", type=int, default=128)

    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=-1,
        help="Local SGD batch size. -1 uses each client's full dataset.",
    )
    parser.add_argument("--lr_scale", type=float, default=0.1)
    parser.add_argument("--pivot_epoch", type=float, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)

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
    if not args.dataset_dir:
        here = os.path.abspath(os.path.dirname(__file__))
        repo = os.path.abspath(os.path.join(here, ".."))
        args.dataset_dir = os.path.join(repo, "datasets", "uci_har_mm")
    return args
