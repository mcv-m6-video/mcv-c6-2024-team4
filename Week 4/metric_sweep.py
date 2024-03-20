import torch
from torch.utils.data import ConcatDataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm
from pytorch_metric_learning import miners, losses, distances, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from utils import train, inference, MetricLearningTransforms, DatasetFromSubset
import torch.nn as nn
import os


import wandb
WANDB_ENTITY = None
WANDB_PROJECT = None

def run_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_trial = len(os.listdir("sweep_weights")) + 1

    ### DATASET INITIALIZATION ###
    # initialize transforms
    transform_train = MetricLearningTransforms(config, mode='train', angle=config["angle"], brightness=config["brightness"])
    transform_test = MetricLearningTransforms(config, mode='test', angle=config["angle"], brightness=config["brightness"])

    # Sequence 3 for testing
    test_dataset = ImageFolder("metric_cars_dataset/S03", transform=transform_test)

    # Sequence 1 and 4 for training (Train/Validation - 80%/20% split)
    dataset_s01 = ImageFolder("metric_cars_dataset/S01")
    dataset_s04 = ImageFolder("metric_cars_dataset/S04")
    train_dataset, val_dataset = random_split(ConcatDataset([dataset_s01, dataset_s04]), [0.8, 0.2])
    # Train/validation transforms (We need to convert the Subsets to Dataset be able to define them)
    train_dataset = DatasetFromSubset(train_dataset, transform_train)
    val_dataset = DatasetFromSubset(val_dataset, transform_test)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config["batch_size"])

    ### METRIC LEARNING ###
    distance = distances.LpDistance(power=2)
    loss_func = losses.ContrastiveLoss(pos_margin=config["pos_margin"], neg_margin=1-config["pos_margin"], distance=distance)
    miner = miners.PairMarginMiner(pos_margin=config["pos_margin"], neg_margin=1-config["pos_margin"])
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1"), k=config['n_neighbors'])

    # Model initializatin
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.DEFAULT')
            self.model.fc = nn.Identity()
            
        def forward(self, x):
            return self.model(x)
    model = Net()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    best_acc = 0
    # Adjusted Training Loop with Miner
    for epoch in range(config['epochs']):
        train_loss = train(model, train_loader, optimizer, loss_func, miner, device)
        accuracies = inference(model, train_dataset, val_dataset, accuracy_calculator, mode="val")
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Acc: {accuracies['precision_at_1']}")

        wandb.log({'Train Loss': train_loss, 'Validation Acc': accuracies['precision_at_1']})

        if accuracies['precision_at_1'] > best_acc:
            best_acc = accuracies['precision_at_1']
            torch.save(model.state_dict(), f'./sweep_weights/siamese_model_{current_trial}.pth')

    print(f"Best Validation Accuracy: {best_acc}")
    print("Testing the model...")
    model.load_state_dict(torch.load(f'./sweep_weights/siamese_model_{current_trial}.pth', map_location=device))
    test_accuracies = inference(model, train_dataset, test_dataset, accuracy_calculator, mode="test")
    print(f"Test Accuracy: {test_accuracies['precision_at_1']}")
    return test_accuracies['precision_at_1']

def main():
    wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT)
    score = run_model(wandb.config)
    wandb.log({"Accuracy": score})

if __name__ == "__main__":
    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "minimize", "name": "Accuracy"},
        "parameters": {
            "IMG_WIDTH": {"value": 224},
            "IMG_HEIGHT": {"value": 224},
            "batch_size": {"value": 32},
            "learning_rate": {"values": [1e-2, 1e-3, 1e-4]},
            "n_neighbors": {"value": 5},
            "epochs": {"value": 10},
            "pos_margin": {"values": [0, 0.1, 0.2]},
            "angle": {"values": [90, 0]},
            "brightness": {"values": [0, 0.5]}
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, entity=WANDB_ENTITY, project=WANDB_PROJECT)
    wandb.agent(sweep_id, function=main, count=25)

