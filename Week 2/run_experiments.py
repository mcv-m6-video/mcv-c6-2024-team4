import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from ultralytics import YOLO
from collections import defaultdict
from sklearn.model_selection import KFold
import json
import torch
# NECESSARY to prevent package errors
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def create_split_files(train_idxs, test_idxs):
    """
        Given the indices for the test and train samples, create the required split file
        by "ultralytics".
    """
    dataset_folder = "data/sequence_dataset"
    with open(f"{dataset_folder}/train.txt", "w+") as f:
        f.writelines(
            [ f"{dataset_folder}/images/{str(id)}.png\n"
                for id in train_idxs
            ])
        
    with open(f"{dataset_folder}/test.txt", "w+") as f:
        f.writelines(
            [ f"{dataset_folder}/images/{str(id)}.png\n"
                for id in test_idxs
            ])
        
def run_model(train_idxs, test_idxs, epochs=10, batch_size=4, iou_thr=0.5, results_name=None):
    """
        Create split files and train YOLOv8.
    """
    # Create files train.txt and test.txt files, which will indicate which frames belong to 
    # each split.
    create_split_files(train_idxs, test_idxs)
    
    # Load pretrained model
    model = YOLO("yolov8n.pt")

    if torch.cuda.is_available():
        device = 0
    else:
        device = "cpu"

    # Start training
    output = model.train(data="dataset.yaml", plots=True, epochs=epochs, batch=batch_size, iou=iou_thr, project="results", name=results_name, device=device)
    results_dict = output.results_dict
    return results_dict

if __name__ == '__main__':
    n_frames = len(os.listdir("data/sequence_dataset/images"))
    idx_list = np.arange(0, n_frames)
    K = 4
    n_epochs = 100

    random_splits_idxs = {} # Save random split indices
    results_experiments = {}
    for strategy in ["A", "B", "C"]:
        if strategy == "A":
            # For strategy A, we use the first 25% of the data for training and the
            # rest for test.
            train_idxs = idx_list[:(n_frames//4)]
            test_idxs = idx_list[(n_frames//4):]

            # Run model
            res = run_model(train_idxs, test_idxs, epochs=n_epochs, results_name=f"strategy_{strategy}")
            results_experiments[strategy] = res

        else:
            # For both strategy "B" and "C" we do 4-Fold cross-validation.
            #    - B: Split video into 4 sequences (folds).
            #    - C: The folds are randombly sampled from all the frames.
            metrics = defaultdict(list) # Save the evaluation metrics for each fold
            shuffle = strategy == "C"
            kfold = KFold(n_splits=K, shuffle=shuffle)
            for i, (test_index, train_index) in enumerate(kfold.split(idx_list)):
                print(f"Fold {i}:")
                print(f"  Train: index={train_index}")
                print(f"  Test:  index={test_index}")
                if strategy == "C":
                    # We save what samples belong to each random fold, just in case.
                    random_splits_idxs[i] = {"Train": train_index.tolist(), "Test": test_index.tolist()}

                # Run model
                res = run_model(train_index, test_index, epochs=n_epochs, results_name=f"strategy_{strategy}_fold_{i}")
                for k, v in res.items():
                    metrics[k].append(v)

            results_experiments[strategy] = {k:[np.mean(v), v] for k, v in metrics.items()}
            
    print("FINAL RESULTS:\n\n", results_experiments)

    # Save experiment results
    with open("results/experiment_results.json", "w") as f:
        json.dump(results_experiments, f)

    # Save split information of strategy C
    with open("results/C_random_indices_splits.json", "w") as f:
        json.dump(random_splits_idxs, f)