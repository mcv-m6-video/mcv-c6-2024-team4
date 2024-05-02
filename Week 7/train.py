""" Main script for training a video classification model on HMDB51 dataset. """

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Iterator

from torch.utils.data import DataLoader

from datasets.HMDB51Dataset import HMDB51Dataset
from models import model_creator
from utils import model_analysis
from utils import statistics

import wandb

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

def train(
        model: nn.Module,
        train_loader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        loss_fn: nn.Module,
        type_data: str,
        device: str,
        description: str = ""
    ) -> None:
    """
    Trains the given model using the provided data loader, optimizer, and loss function.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): The data loader containing the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        loss_fn (nn.Module): The loss function used to compute the training loss.
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        None
    """
    model.train()
    pbar = tqdm(train_loader, desc=description, total=len(train_loader))
    loss_train_mean = statistics.RollingMean(window_size=len(train_loader))
    hits = count = 0 # auxiliary variables for computing accuracy
    for batch in pbar:
        # Gather batch and move to device
        if type_data == "rgb":
            clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        elif type_data == "of":
            clips, labels = train_loader.dataset.get_of_batch(batch['clips']).to(device), batch['labels'].to(device)
        elif type_data == "both":
            clips_rgb, labels = batch['clips'].to(device), batch['labels'].to(device)
            clips_of = train_loader.dataset.get_of_batch(clips_rgb).to(device)
            clips = torch.cat((clips_rgb, clips_of), dim=1)
            
        # Forward pass
        outputs = model(clips)
        # Compute loss
        loss = loss_fn(outputs, labels)
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Update progress bar with metrics
        loss_iter = loss.item()
        hits_iter = torch.eq(outputs.argmax(dim=1), labels).sum().item()
        hits += hits_iter
        count += len(labels)
        pbar.set_postfix(
            loss=loss_iter,
            loss_mean=loss_train_mean(loss_iter),
            acc=(float(hits_iter) / len(labels)),
            acc_mean=(float(hits) / count)
        )
    return loss_train_mean(loss_iter), float(hits) / count


def evaluate(
        model: nn.Module, 
        valid_loader: DataLoader, 
        loss_fn: nn.Module,
        type_data: str,
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
    for batch in pbar:
        # Gather batch and move to device
        if type_data == "rgb":
            clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        elif type_data == "of":
            clips, labels = valid_loader.dataset.get_of_batch(batch['clips']).to(device), batch['labels'].to(device)
        elif type_data == "both":
            clips_rgb, labels = batch['clips'].to(device), batch['labels'].to(device)
            clips_of = valid_loader.dataset.get_of_batch(clips_rgb).to(device)
            clips = torch.cat((clips_rgb, clips_of), dim=1)
        # Forward pass
        with torch.no_grad():
            outputs = model(clips)
            scores = torch.nn.functional.log_softmax(outputs, dim=-1)
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
    return loss_valid_mean(loss_iter), float(hits) / count


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
            num_spatial_crops,
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


def create_optimizer(optimizer_name: str, parameters: Iterator[nn.Parameter], lr: float = 1e-4) -> torch.optim.Optimizer:
    """
    Creates an optimizer for the given parameters.
    
    Args:
        optimizer_name (str): Name of the optimizer (supported: "adam" and "sgd" for now).
        parameters (Iterator[nn.Parameter]): Iterator over model parameters.
        lr (float, optional): Learning rate. Defaults to 1e-4.

    Returns:
        torch.optim.Optimizer: The optimizer for the model parameters.
    """
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")


def print_model_summary(
        model: nn.Module,
        clip_length: int,
        crop_size: int,
        input_channel: int,
        print_model: bool = True,
        print_params: bool = True,
        print_FLOPs: bool = True
    ) -> None:
    """
    Prints a summary of the given model.

    Args:
        model (nn.Module): The model for which to print the summary.
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        print_model (bool, optional): Whether to print the model architecture. Defaults to True.
        print_params (bool, optional): Whether to print the number of parameters. Defaults to True.
        print_FLOPs (bool, optional): Whether to print the number of FLOPs. Defaults to True.

    Returns:
        None
    """
    if print_model:
        print(model)

    if print_params:
        num_params = sum(p.numel() for p in model.parameters())
        #num_params = model_analysis.calculate_parameters(model) # should be equivalent
        print(f"Number of parameters (M): {round(num_params / 1e6, 2)}")

    if print_FLOPs:
        num_FLOPs = model_analysis.calculate_operations(model, clip_length, crop_size, crop_size, input_channel)
        print(f"Number of FLOPs (G): {round(num_FLOPs / 1e9, 2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a video classification model on HMDB51 dataset.')
    parser.add_argument('--frames_dir', type=str, default='/ghome/group08/c6/week5/frames',
                        help='Directory containing video files')
    parser.add_argument('--annotations-dir', type=str, default="/ghome/group08/c6/week5/data/hmdb51/testTrainMulti_601030_splits",
                        help='Directory containing annotation files')
    parser.add_argument('--num_segments', type=int, default=2,
                        help='Number of segments')
    parser.add_argument('--num_spatial_crops', type=int, default=3,
                        help='Number of spatial crops')
    parser.add_argument('--clip-length', type=int, default=16,
                        help='Number of frames of the clips')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='Size of spatial crops (squares)')
    parser.add_argument('--temporal-stride', type=int, default=5,
                        help='Receptive field of the model will be (clip_length * temporal_stride) / FPS')
    parser.add_argument('--model-name', type=str, default='x3d_m',
                        help='Model name as defined in models/model_creator.py')
    parser.add_argument('--type_data', type=str, default='rgb',
                        help='Type of data to use (rgb, of, both)')
    parser.add_argument('--load-pretrain', action='store_true', default=False,
                    help='Load pretrained weights for the model (if available)')
    parser.add_argument('--optimizer-name', type=str, default="adam",
                        help='Optimizer name (supported: "adam" and "sgd" for now)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for the training data loader')
    parser.add_argument('--batch-size-eval', type=int, default=4,
                        help='Batch size for the evaluation data loader')
    parser.add_argument('--validate-every', type=int, default=1,
                        help='Number of epochs after which to validate the model')
    parser.add_argument('--patience', type=int, default=25,
                        help='Number of epochs to wait after last time validation loss improved')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (cuda or cpu)')

    args = parser.parse_args()
    args.input_channel = 2 if args.type_data == 'of' else 3 if args.type_data == 'rgb' else 5


    # Create datasets
    datasets = create_datasets(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_1, # hardcoded
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        temporal_stride=args.temporal_stride,
        num_segments=args.num_segments, 
        num_spatial_crops=args.num_spatial_crops,
    )

    # Create data loaders
    loaders = create_dataloaders(
        datasets,
        args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers
    )

    # Init model, optimizer, loss function and the early stopping strategy
    model = model_creator.create(args.model_name, args.load_pretrain, datasets["training"].get_num_classes(), args.input_channel)
    optimizer = create_optimizer(args.optimizer_name, model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    early_stopping = statistics.EarlyStopping(patience=args.patience, verbose=True)
 
    print_model_summary(model, args.clip_length, args.crop_size, args.input_channel, print_model=False)
    model = model.to(args.device)

    # Init WandB
    WANDB_PROJECT = "c6-week6"
    WANDB_ENTITY = "luisgogu2001"
    run_name = f"x3d_m_rgb"
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name, config=args)

    best_acc = 0
    for epoch in range(args.epochs):
        # Training
        description = f"Training [Epoch: {epoch+1}/{args.epochs}]"
        mean_train_loss, train_acc = train(model, loaders['training'], optimizer, loss_fn, args.type_data, args.device, description=description)

        # Validation
        if epoch % args.validate_every == 0:
            description = f"Validation [Epoch: {epoch+1}/{args.epochs}]"
            mean_val_loss, val_acc = evaluate(model, loaders['testing'], loss_fn, args.type_data, args.num_segments, args.num_spatial_crops, args.device, description=description)

            wandb.log({
                'Mean Training Loss': mean_train_loss, 
                'Mean Validation Loss': mean_val_loss, 
                'Training Acc': train_acc,
                'Validation Acc': val_acc})

            # Call early stopping
            early_stopping(mean_val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        else:
            wandb.log({
                'Mean Training Loss': mean_train_loss,
                'Training Acc': train_acc})

        if val_acc > best_acc:
            best_loss = val_acc
            torch.save(model.state_dict(), f'../pretrained/{run_name}.pth')
        
    # Testing
    model.load_state_dict(torch.load(f'../pretrained/{run_name}.pth', map_location=args.device))
    mean_test_loss, test_acc = evaluate(model, loaders['testing'], loss_fn, args.type_data, args.num_segments, args.num_spatial_crops, args.device, description=f"Testing")
    wandb.log({"Accuracy": test_acc})
    print(f"Test Loss: {mean_test_loss} | Test Accuracy: {test_acc}")

    exit()
