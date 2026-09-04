import os
import sys
import torch
import numpy as np

from options import args_parser
from dataset import attach_global_test, dirichlet_partition_data, load_and_restructure_uci_har_data
from federated import federated_learning
from models import Acc_MLP, Gyro_MLP

def main():
    args = args_parser()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs('results', exist_ok=True)

    print("Loading and restructuring data...")
    filepath = args.data_path
    if not os.path.isabs(filepath):
        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), filepath))
    else:
        filepath = os.path.abspath(filepath)
    if os.path.isdir(filepath):
        data_ok = os.path.isfile(os.path.join(filepath, 'data.npz'))
    else:
        data_ok = os.path.isfile(filepath)
    if not data_ok:
        print(
            f"Error: UCI HAR data not found:\n  {filepath}\n"
            "Pass --data_path to datasets/uci_har_mm/data.npz (same as run_uci_har_sketch_fusion_B.sh).",
            file=sys.stderr,
        )
        sys.exit(1)
    client_data, global_test = load_and_restructure_uci_har_data(filepath)

    if args.class_non_iid_rate < 1.0:
        print("Using Dirichlet partitioning...")
        client_data = dirichlet_partition_data(client_data, alpha=args.class_non_iid_rate)

    # Full official train split per client (no local 80/20 hold-out).
    client_data_train = client_data
    print("Evaluating on the official UCI HAR test set (global test_acc + per-client RF fusion).")
    client_data_test = attach_global_test(client_data_train.keys(), global_test)

    print("Initializing global modality encoders...")
    global_modality_encoders = [Acc_MLP(), Gyro_MLP()]

    print("Starting federated learning...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    accuracy_matrix, modality_counts, test_acc, acc_test, gyro_test = federated_learning(
        args=args,
        client_data_train=client_data_train,
        client_data_test=client_data_test,
        global_models=global_modality_encoders,
        global_test=global_test,
        device=device
    )

    # Save results
    # Filename format: MFedMC_UCI_HAR_Top_{args.top_shap}_ShapCommRec_{modality_weights}_Client_{client_select_ratio}.npz
    mw_str = "_".join([f"{w:.1f}" for w in args.modality_weights])
    file_name = f"results/MFedMC_UCI_HAR_Top_{args.top_shap}_ShapCommRec_{mw_str}_Client_{args.client_select_ratio:.1f}.npz"
    np.savez(
        file_name,
        acc=accuracy_matrix,
        mod=modality_counts,
        test_acc=test_acc,
        acc_test=acc_test,
        gyro_test=gyro_test,
    )
    print(
        f"Final test_acc: {float(test_acc[-1]):.4f} ({100.0 * float(test_acc[-1]):.2f}%)"
    )
    print(f"Results saved to {file_name}")

if __name__ == '__main__':
    main()
