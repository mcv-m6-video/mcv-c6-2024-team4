import argparse, pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Iterator

from torch.utils.data import DataLoader

from datasets.HMDB51Dataset import HMDB51Dataset
from models import model_creator
from utils import statistics

def create_datasets(
        frames_dir: str,
        annotations_dir: str,
        split: HMDB51Dataset.Split,
        clip_length: int,
        crop_size: int,
        temporal_stride: int,
        num_segments: int, 
        num_spatial_crops: int
) -> Dict[str, HMDB51Dataset]:
    """
    Creates datasets for training, validation, and testing.

    Args:
        frames_dir (str): Directory containing the video frames (a separate directory per video).
        annotations_dir (str): Directory containing annotation files.
        split (HMDB51Dataset.Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.

    Returns:
        Dict[str, HMDB51Dataset]: A dictionary containing the datasets for training, validation, and testing.
    """
    datasets = {}
    for regime in HMDB51Dataset.Regime:
        datasets[regime.name.lower()] = HMDB51Dataset(
            frames_dir,
            annotations_dir,
            split,
            regime,
            clip_length,
            crop_size,
            temporal_stride,
            num_segments, 
            num_spatial_crops
        )
    
    return datasets


def create_dataloaders(
        datasets: Dict[str, HMDB51Dataset],
        batch_size: int,
        batch_size_eval: int = 8,
        num_workers: int = 2,
        pin_memory: bool = True
    ) -> Dict[str, DataLoader]:
    """
    Creates data loaders for training, validation, and testing datasets.

    Args:
        datasets (Dict[str, HMDB51Dataset]): A dictionary containing datasets for training, validation, and testing.
        batch_size (int, optional): Batch size for the data loaders. Defaults to 8.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 2.
        pin_memory (bool, optional): Whether to pin memory in DataLoader for faster GPU transfer. Defaults to True.

    Returns:
        Dict[str, DataLoader]: A dictionary containing data loaders for training, validation, and testing datasets.
    """
    dataloaders = {}
    for key, dataset in datasets.items():
        dataloaders[key] = DataLoader(
            dataset,
            batch_size=(batch_size if key == 'training' else batch_size_eval),
            shuffle=(key == 'training'),  # Shuffle only for training dataset
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
            
    return dataloaders

def aggregation_function(scores, num_segments, num_crops):
    total_n, n_classes = scores.shape
    batch_size = total_n // (num_segments * num_crops)

    aggregated_outputs = torch.zeros((batch_size, n_classes), device=scores.device)
    for i in range(batch_size):
        # Extract scores for each segment-crop combination
        segment_crop_scores = scores[i * num_segments * num_crops: (i + 1) * num_segments * num_crops]

        # Reshape scores to separate segments and crops
        segment_crop_scores = segment_crop_scores.reshape((num_segments, num_crops, n_classes))

        # Average scores across crops for each segment
        segment_scores = segment_crop_scores.mean(axis=1)

        # Aggregate scores across segments
        aggregated_outputs[i, :] = segment_scores.mean(axis=0)

    return aggregated_outputs

# def evaluate(
#         model: nn.Module, 
#         valid_loader: DataLoader, 
#         loss_fn: nn.Module,
#         type_data: str,
#         num_segments: int,
#         num_spatial_crops: int,
#         device: str,
#         description: str = ""
#     ) -> None:
#     """
#     Evaluates the given model using the provided data loader and loss function.

#     Args:
#         model (nn.Module): The neural network model to be validated.
#         valid_loader (DataLoader): The data loader containing the validation dataset.
#         loss_fn (nn.Module): The loss function used to compute the validation loss (not used for backpropagation)
#         device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
#         description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

#     Returns:
#         None
#     """
#     model.eval()
#     pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
#     loss_valid_mean = statistics.RollingMean(window_size=len(valid_loader))
#     hits = count = 0 # auxiliary variables for computing accuracy
#     all_labels = []
#     all_predictions = []
#     for batch in pbar:
#         # Gather batch and move to device
#         if type_data == "rgb":
#             clips, labels = batch['clips'].to(device), batch['labels'].to(device)
#         elif type_data == "of":
#             clips, labels = valid_loader.dataset.get_of_batch(batch['clips']).to(device), batch['labels'].to(device)
#         elif type_data == "both":
#             clips_rgb, labels = batch['clips'].to(device), batch['labels'].to(device)
#             clips_of = valid_loader.dataset.get_of_batch(clips_rgb).to(device)
#             clips = torch.cat((clips_rgb, clips_of), dim=1)
#         # Forward pass
#         with torch.no_grad():
#             outputs = model(clips)
#             scores = torch.nn.functional.log_softmax(outputs, dim=-1)
#             # Agregate outputs per video
#             aggregated_outputs = aggregation_function(scores, num_segments, num_spatial_crops)
#             # Compute loss (just for logging, not used for backpropagation)
#             loss = loss_fn(aggregated_outputs, labels) 
#             # Compute metrics
#             loss_iter = loss.item()
#             hits_iter = torch.eq(aggregated_outputs.argmax(dim=1), labels).sum().item()
#             hits += hits_iter
#             count += len(labels)
#             # Update progress bar with metrics
#             pbar.set_postfix(
#                 loss=loss_iter,
#                 loss_mean=loss_valid_mean(loss_iter),
#                 acc=(float(hits_iter) / len(labels)),
#                 acc_mean=(float(hits) / count)
#             )
#             # Save labels and predictions
#             all_labels.extend(labels.cpu().numpy())
#             all_predictions.extend(aggregated_outputs.argmax(dim=1).cpu().numpy())
#     return all_labels, all_predictions, loss_valid_mean(loss_iter), float(hits) / count

def evaluate(
        model: nn.Module, 
        valid_loader: DataLoader, 
        loss_fn: nn.Module,
        num_segments: int,
        num_spatial_crops: int,
        device: str,
        description: str = ""
    ) -> None:
    """
    Evaluates the given model using the provided data loader and loss function.

    Args:
        model (nn.Module): The neural network model to be validated.
        valid_loader (DataLoader): The data loader containing the validation dataset.
        loss_fn (nn.Module): The loss function used to compute the validation loss (not used for backpropagation)
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        None
    """
    model.eval()
    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    loss_valid_mean = statistics.RollingMean(window_size=len(valid_loader))
    hits = count = 0 # auxiliary variables for computing accuracy
    all_labels = []
    all_predictions = []
    for batch in pbar:
        # Gather batch and move to device
        clips_rgb, labels = batch['clips'].to(device), batch['labels'].to(device)
        clips_of = valid_loader.dataset.get_of_batch(clips_rgb).to(device)

        # Forward pass
        with torch.no_grad():
            scores = model(clips_rgb, clips_of)
            #scores = torch.nn.functional.log_softmax(outputs, dim=-1)
            # Agregate outputs per video
            aggregated_outputs = aggregation_function(scores, num_segments, num_spatial_crops)
            # Compute loss (just for logging, not used for backpropagation)
            loss = loss_fn(aggregated_outputs, labels) 
            # Compute metrics
            loss_iter = loss.item()
            hits_iter = torch.eq(aggregated_outputs.argmax(dim=1), labels).sum().item()
            hits += hits_iter
            count += len(labels)
            # Update progress bar with metrics
            pbar.set_postfix(
                loss=loss_iter,
                loss_mean=loss_valid_mean(loss_iter),
                acc=(float(hits_iter) / len(labels)),
                acc_mean=(float(hits) / count)
            )
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(aggregated_outputs.argmax(dim=1).cpu().numpy())
    return all_labels, all_predictions, loss_valid_mean(loss_iter), float(hits) / count

parser = argparse.ArgumentParser(description='Train a video classification model on HMDB51 dataset.')

parser.frames_dir = "/ghome/group08/c6/week5/frames"
parser.annotations_dir = "/ghome/group08/c6/week5/data/hmdb51/testTrainMulti_601030_splits"
parser.clip_length = 16
parser.crop_size = 224
parser.temporal_stride = 5
parser.model_name = "x3d_m"
parser.load_pretrain = False
parser.num_workers = 2
parser.device = "cuda"
parser.batch_size = 16
parser.batch_size_eval = 16
parser.num_segments = 2
parser.num_spatial_crops = 3

args = parser

# Create datasets
datasets = create_datasets(
    frames_dir=args.frames_dir,
    annotations_dir=args.annotations_dir,
    split=HMDB51Dataset.Split.TEST_ON_SPLIT_1, # hardcoded
    clip_length=args.clip_length,
    crop_size=args.crop_size,
    temporal_stride=args.temporal_stride,
    num_segments=args.num_segments,
    num_spatial_crops=args.num_spatial_crops
)

# Create data loaders
loaders = create_dataloaders(
    datasets,
    args.batch_size,
    batch_size_eval=args.batch_size_eval,
    num_workers=args.num_workers
)

# model = model_creator.create('x3d_m', False, 51, 5)
# model.load_state_dict(torch.load('../pretrained/x3d_m_both.pth', map_location='cpu'))
# Init model, optimizer, loss function and the early stopping strategy
model_rgb = model_creator.create(args.model_name, args.load_pretrain, datasets["training"].get_num_classes(), 3)
model_of = model_creator.create(args.model_name, args.load_pretrain, datasets["training"].get_num_classes(), 2)

class JointModel(nn.Module):
    def __init__(self, model_rgb, model_of): 
        super().__init__()
        self.model_rgb = model_rgb
        self.model_of = model_of
    def forward(self, rgb, of):
        # Perform aggregation at class score level
        scores_rbg = nn.functional.log_softmax(self.model_rgb(rgb))
        scores_of = nn.functional.log_softmax(self.model_of(of))
        output = torch.stack([scores_rbg, scores_of], dim=-1).mean(-1)
        return output
        
model = JointModel(model_rgb=model_rgb, model_of=model_of)
model.load_state_dict(torch.load('../pretrained/aggregation.pth', map_location='cpu'))
model.to(args.device)
model.eval()

loss_fn = nn.CrossEntropyLoss()
all_labels, all_predictions, mean_loss, mean_acc  = evaluate(model, loaders['testing'], loss_fn, 2, 3, args.device, description=f"Testing")
print(f"Mean loss: {mean_loss:.4f}, Mean accuracy: {mean_acc:.4f}")

with open('test_predictions_agg.pkl', 'wb') as f:
    pickle.dump(all_predictions, f)

with open('test_labels_agg.pkl', 'wb') as f:
    pickle.dump(all_labels, f)