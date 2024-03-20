import torch
from torch.utils.data import ConcatDataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm
from pytorch_metric_learning import miners, losses, distances, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from utils import train, inference, MetricLearningTransforms, DatasetFromSubset
import torch.nn as nn
import wandb
WANDB_ENTITY = None
WANDB_PROJECT = None

if __name__ == "__main__":
    # Configuration
    config = {
        "IMG_WIDTH": 224,
        "IMG_HEIGHT": 224,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "n_neighbors": 5,
        'epochs': 10,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "neg_margin": 0.8,
        "pos_margin": 0.2
    }

    ### DATASET INITIALIZATION ###
    # initialize transforms
    transform_train = MetricLearningTransforms(config, mode='train')
    transform_test = MetricLearningTransforms(config, mode='test')

    # Sequence 3 for testing
    test_dataset = ImageFolder("metric_cars_dataset/S03", transform=transform_test)

    # Sequence 1 and 4 for training (Train/Validation - 80%/20% split)
    dataset_s01 = ImageFolder("metric_cars_dataset/S01")
    dataset_s04 = ImageFolder("metric_cars_dataset/S04")
    train_dataset, val_dataset = random_split(ConcatDataset([dataset_s01, dataset_s04]), [0.8, 0.2])
    # Train/validation transforms (We need to convert the Subsets to Dataset be able to define them)
    train_dataset = DatasetFromSubset(train_dataset, transform_train)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config["batch_size"])
    val_dataset = DatasetFromSubset(val_dataset, transform_test)

    ### METRIC LEARNING ###
    distance = distances.LpDistance(power=2)
    loss_func = losses.ContrastiveLoss(pos_margin=config["pos_margin"], neg_margin=config["neg_margin"], distance=distance)
    miner = miners.PairMarginMiner(pos_margin=config["pos_margin"], neg_margin=config["neg_margin"])
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1", 'mean_average_precision'), k=config['n_neighbors'])

    # Model
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.DEFAULT')
            self.model.fc = nn.Identity()
            
        def forward(self, x):
            return self.model(x)
    model = Net()
    model.to(config['device'])

    MODE = "Test"
    if MODE == "Train":
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=config)
        best_acc = 0
        # Adjusted Training Loop with Miner
        for epoch in range(config['epochs']):
            train_loss = train(model, train_loader, optimizer, loss_func, miner, config['device'])
            accuracies = inference(model, train_dataset, val_dataset, accuracy_calculator, mode="val")
            
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Acc: {accuracies['precision_at_1']}, MAP: {accuracies['mean_average_precision']}")

            wandb.log({'Train Loss': train_loss, 'Validation Acc': accuracies['precision_at_1'], 'MAP': accuracies['mean_average_precision']})

            if accuracies['precision_at_1'] > best_acc:
                best_acc = accuracies['precision_at_1']
                torch.save(model.state_dict(), './pretrained/siamese_model.pth')
        # Load best model found
        print(f"Best Validation Accuracy: {best_acc}")
        model.load_state_dict(torch.load(f'./pretrained/siamese_model.pth', map_location=config['device']))
    else:
        model.load_state_dict(torch.load(f'./sweep_weights/siamese_model_6.pth', map_location=config['device']))

    print("Testing the model...")
    test_accuracies = inference(model, train_dataset, test_dataset, accuracy_calculator, mode="test")
    print(f"Test Accuracy: {test_accuracies['precision_at_1']}, MAP: {test_accuracies['mean_average_precision']}")