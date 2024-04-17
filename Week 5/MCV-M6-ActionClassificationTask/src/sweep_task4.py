""" Main script for training a video classification model on HMDB51 dataset. """

import argparse
import os
import torch
import torch.nn as nn

from tqdm import tqdm
from typing import Dict, Iterator

from torch.utils.data import DataLoader

from datasets.HMDB51Dataset_task4 import HMDB51Dataset
from models import model_creator
from utils import model_analysis
from utils import statistics

from train_task4 import *

import wandb
WANDB_PROJECT = ""
WANDB_ENTITY = ""


def run_model(config):
        SAVE_FOLDER = "./c6/week5/pretrained/sweep3"
        current_trial = len(os.listdir(SAVE_FOLDER)) + 1

        # Update config with sweep's selected hyperparameters
        args = parse_arguments()
        args.num_segments = config["num_segments"]
        args.lr = config["learning_rate"]
        args.optimizer_name = config["optimizer"]
        args.clip_length = config["clip_length"]
        args.temporal_stride = config["temporal_stride"]

        # Create datasets
        datasets = create_datasets(
            frames_dir=args.frames_dir,
            annotations_dir=args.annotations_dir,
            split=HMDB51Dataset.Split.TEST_ON_SPLIT_1, # hardcoded
            clip_length=args.clip_length,
            crop_size=args.crop_size,
            temporal_stride=args.temporal_stride,
            num_segments=args.num_segments
        )

        # Create data loaders
        loaders = create_dataloaders(
            datasets,
            args.batch_size,
            batch_size_eval=args.batch_size_eval,
            num_workers=args.num_workers
        )

        # Init model, optimizer, and loss function
        model = model_creator.create(args.model_name, args.load_pretrain, datasets["training"].get_num_classes())
        optimizer = create_optimizer(args.optimizer_name, model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()

        print_model_summary(model, args.clip_length, args.crop_size)

        model = model.to(args.device)

        # Init WandB
        run_name = f"model_{current_trial}"
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name, config=args)
        
        best_acc = 0
        for epoch in range(args.epochs):
            # Training
            description = f"Training [Epoch: {epoch+1}/{args.epochs}]"
            mean_train_loss, train_acc = train(model, loaders['training'], optimizer, nn.NLLLoss(), args.num_segments, args.device, description=description, agg_type=args.agg_type)

            # Validation
            if epoch % args.validate_every == 0:
                description = f"Validation [Epoch: {epoch+1}/{args.epochs}]"
                mean_val_loss, val_acc = evaluate(model, loaders['validation'], loss_fn, args.device, description=description)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f'{SAVE_FOLDER}/{run_name}.pth')
            
            wandb.log({
                'Mean Training Loss': mean_train_loss, 
                'Mean Validation Loss': mean_val_loss, 
                'Training Acc': train_acc,
                'Validation Acc': val_acc})

        # Testing
        model.load_state_dict(torch.load(f'{SAVE_FOLDER}/{run_name}.pth', map_location=args.device))
        _, test_acc = evaluate(model, loaders['testing'], loss_fn, args.device, description=f"Testing")
        return test_acc

    
def main():
    wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT)
    score = run_model(wandb.config)
    wandb.log({"Accuracy": score})

if __name__ == "__main__":
    sweep_configuration = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "Accuracy"},
        "parameters": {
            "num_segments": {"values": [3, 4, 5]},
            "learning_rate": {"values": [5e-4, 2e-4, 1e-4, 5e-5]},
            "optimizer": {"values": ["adam"]},
            "clip_length": {"values": [4, 6, 8, 10]},
            "temporal_stride": {"values": [8, 12, 16]}
        }
    }
    
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configuration, entity=WANDB_ENTITY, project=WANDB_PROJECT)
    wandb.agent(sweep_id, function=main, count=25)
    
