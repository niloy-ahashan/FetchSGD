import argparse


def args_parser():
    parser = argparse.ArgumentParser(
        description="MFedMC paper reproduction on raw UCI HAR Inertial Signals "
        "(Acc=total_acc 128x3, Gyro=body_gyro 128x3) with per-modality LSTM "
        "encoders and the natural 30-subject client partition."
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--uci_root",
        type=str,
        default="",
        help="Inner UCI HAR folder (contains train/subject_train.txt and "
        "train/Inertial Signals/). Default: auto-discovered.",
    )
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument(
        "--hidden_size", type=int, default=128,
        help="LSTM hidden units (paper Sec 4.2).",
    )
    parser.add_argument(
        "--class_non_iid_rate",
        type=float,
        default=1.0,
        help="Optional extra Dirichlet label-skew on top of the natural subject "
        "clients. 1.0 = off (default, matches the paper's Natural Distribution).",
    )

    parser.add_argument("--lr", type=float, default=0.1, help="Paper Sec 4.2.")
    parser.add_argument("--batch_size", type=int, default=32, help="Paper Sec 4.2.")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Per-client stratified local-holdout train fraction (paper protocol: "
        "always local holdout, never a shared global test set).",
    )

    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--local_epochs", type=int, default=5, help="Paper E=5.")

    parser.add_argument(
        "--top_shap",
        type=int,
        default=1,
        help="gamma: keep this many highest-priority modalities per client (paper gamma=1).",
    )
    parser.add_argument(
        "--modality_weights",
        nargs="+",
        type=float,
        default=[1 / 3, 1 / 3, 1 / 3],
        help="alpha_s, alpha_c, alpha_r (paper: 1/3 each).",
    )
    parser.add_argument(
        "--random_modality",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use random modality selection instead of the priority rule.",
    )

    parser.add_argument(
        "--client_select_ratio",
        type=float,
        default=0.2,
        help="delta: ratio of clients to select per round (paper delta=0.2).",
    )
    parser.add_argument(
        "--client_weights",
        nargs="+",
        type=float,
        default=[1.0, 0.0],
        help="Weights for loss and staleness in client selection.",
    )
    parser.add_argument(
        "--random_clients",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--prefer_higher_loss",
        dest="prefer_higher_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Paper's Eq. (18) always selects the LOWEST-loss clients (default: "
        "False here, correct from the start). Pass --prefer_higher_loss to opt "
        "into the old, paper-inverted highest-loss behavior for comparison.",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory for the output npz (relative to this package unless absolute).",
    )

    return parser.parse_args()
