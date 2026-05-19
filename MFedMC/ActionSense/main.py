import os
import sys
import torch
import numpy as np

from options import args_parser
from dataset import load_and_restructure_hdf5_data, dirichlet_partition_data, stratified_split_client_data
from federated import federated_learning
from models import Eye_LSTM, EMG_LSTM, Tactile_LSTM, IMU_LSTM

def main():
    args = args_parser()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs('results', exist_ok=True)

    print("Loading and restructuring data...")
    filepath = os.path.abspath(args.data_path)
    if not os.path.isfile(filepath):
        # print(
        #     f"Error: ActionSense data file not found:\n  {filepath}\n"
        #     "Pass --data_path to your HDF5, e.g.\n"
        #     "  python main.py --data_path ./data/data_processed_allStreams_10s_10hz_5subj_ex20-20_allActs.hdf5",
        #     file=sys.stderr,
        # )
        sys.exit(1)
    client_data = load_and_restructure_hdf5_data(filepath)

    if args.class_non_iid_rate < 1.0:
        print("Using Dirichlet partitioning...")
        client_data = dirichlet_partition_data(client_data, alpha=args.class_non_iid_rate)
    
    print("Splitting client data...")
    client_data_train, client_data_test = stratified_split_client_data(client_data, train_ratio=args.train_ratio)

    print("Initializing global modality encoders...")
    k = args.num_classes
    global_modality_encoders = [
        Eye_LSTM(k),
        EMG_LSTM(k),
        EMG_LSTM(k),
        Tactile_LSTM(k),
        Tactile_LSTM(k),
        IMU_LSTM(k),
    ]

    print("Starting federated learning...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    accuracy_matrix, modality_counts, upload_bytes_round = federated_learning(
        args=args,
        client_data_train=client_data_train,
        client_data_test=client_data_test,
        global_models=global_modality_encoders,
        device=device
    )
    upload_bytes_cumulative = np.cumsum(upload_bytes_round)

    # Save results
    # Filename format: MFedMC_ActionSense_Top_{args.top_shap}_ShapCommRec_{modality_weights}_Client_{client_select_ratio}.npz
    mw_str = "_".join([f"{w:.1f}" for w in args.modality_weights])
    file_name = f"results/MFedMC_ActionSense_Top_{args.top_shap}_ShapCommRec_{mw_str}_Client_{args.client_select_ratio:.1f}.npz"
    np.savez(
        file_name,
        acc=accuracy_matrix,
        mod=modality_counts,
        upload_bytes_round=upload_bytes_round,
        upload_bytes_cumulative=upload_bytes_cumulative,
    )
    print(f"Results saved to {file_name}")

if __name__ == '__main__':
    main()
