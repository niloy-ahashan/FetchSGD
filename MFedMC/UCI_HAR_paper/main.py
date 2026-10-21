import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch

from options import args_parser
from dataset import (
    MODALITIES,
    dirichlet_partition_data,
    load_subject_clients,
    stratified_split_client_data,
)
from federated import federated_learning
from models import LSTMEncoder


def main():
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), results_dir)
    args.results_dir = results_dir
    os.makedirs(results_dir, exist_ok=True)

    print("Loading and restructuring data...")
    client_data, meta = load_subject_clients(uci_root=args.uci_root or None)
    args.num_clients = meta["num_clients"]
    if args.class_non_iid_rate < 1.0:
        print("Using Dirichlet partitioning...")
        client_data = dirichlet_partition_data(
            client_data, alpha=args.class_non_iid_rate, seed=args.seed
        )

    print("Splitting client data...")
    client_data_train, client_data_test = stratified_split_client_data(
        client_data, train_ratio=args.train_ratio, seed=args.seed
    )

    print("Initializing global modality encoders...")
    k = args.num_classes
    global_modality_encoders = [
        LSTMEncoder(input_size=3, hidden_size=args.hidden_size, num_classes=k),
        LSTMEncoder(input_size=3, hidden_size=args.hidden_size, num_classes=k),
    ]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for m in global_modality_encoders:
        m.to(device)

    print("Starting federated learning...")
    acc, mod_counts = federated_learning(
        args=args,
        client_data_train=client_data_train,
        client_data_test=client_data_test,
        global_models=global_modality_encoders,
        modalities=MODALITIES,
        device=device,
    )

    mw_str = "_".join(f"{w:g}" for w in args.modality_weights)
    file_name = os.path.join(
        results_dir,
        f"MFedMC_UCI_HAR_paper_Top_{args.top_shap}_ShapCommRec_{mw_str}_"
        f"Client_{args.client_select_ratio:.1f}_seed{args.seed}.npz",
    )
    np.savez(file_name, acc=acc, mod=mod_counts)
    print(f"Results saved to {file_name}")


if __name__ == "__main__":
    main()
