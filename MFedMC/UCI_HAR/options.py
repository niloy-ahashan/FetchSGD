import argparse


def args_parser():
    parser = argparse.ArgumentParser(
        description="MFedMC on the same UCI HAR split as SketchFusionB "
        "(datasets/uci_har_mm: Acc 348-D + Gyro 213-D)."
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="",
        help="Directory with data.npz and optional client*.npz from SketchFusionB. "
        "Default: <repo>/datasets/uci_har_mm",
    )
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.1)
    parser.add_argument("--acc_dim", type=int, default=348)
    parser.add_argument("--gyro_dim", type=int, default=213)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--mm_dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="If < 1, stratified split of each client for a local test set. "
        "Default 1.0 uses the official UCI HAR test set (same as SketchFusionB).",
    )
    parser.add_argument(
        "--eval_on_global_test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate each client's fusion on the official global test set.",
    )

    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=5)

    parser.add_argument(
        "--top_shap",
        type=int,
        default=1,
        help="Keep this many highest-priority modalities per client "
        "(same rule as ActionSense MFedMC; the rest are not uploaded).",
    )
    parser.add_argument(
        "--modality_weights",
        nargs="+",
        type=float,
        default=[1 / 3, 1 / 3, 1 / 3],
        help="Weights for SHAP, model_size, and recency in modality selection",
    )
    parser.add_argument(
        "--random_modality",
        action="store_true",
        help="Use random modality selection instead of Priority",
    )

    parser.add_argument(
        "--client_select_ratio",
        type=float,
        default=0.2,
        help="Ratio of clients to select per round",
    )
    parser.add_argument(
        "--client_weights",
        nargs="+",
        type=float,
        default=[1.0, 0.0],
        help="Weights for loss and staleness in client selection",
    )
    parser.add_argument("--random_clients", action="store_true")
    parser.add_argument(
        "--prefer-higher-loss",
        dest="prefer_higher_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory for the output npz (relative to this package unless absolute).",
    )

    return parser.parse_args()
