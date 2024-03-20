from torchvision import transforms
from tqdm import tqdm
from pytorch_metric_learning import miners, losses, distances, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.data import Dataset
    
class MetricLearningTransforms:
    def __init__(self, config, mode, angle=90, brightness=0.5):
        if mode == 'train':
            # Rotation
            # change in illuminance
            self.transform = transforms.Compose([
                transforms.Resize((config['IMG_WIDTH'], config['IMG_HEIGHT'])),
                transforms.RandomRotation((angle)),
                transforms.ColorJitter(brightness=(1-brightness, 1+brightness)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config['IMG_WIDTH'], config['IMG_HEIGHT'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    def __call__(self, img):
        return self.transform(img)

# By @Andreas K. from https://stackoverflow.com/questions/51782021/how-to-use-different-data-augmentation-for-subsets-in-pytorch
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def train(model, dataloader, optimizer, loss_func, miner, device):
    model.train()
    training_loss = 0.0
    for idx, (data, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        targets = targets.float()  # Ensure targets are float for contrastive loss
        
        miner_out = miner(embeddings, targets)
        loss = loss_func(embeddings, targets, miner_out)
        
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

        if idx % 50 == 0:
            num = miner.num_pos_pairs+miner.num_neg_pairs
            print("Iteration {}: Loss = {}, Number of mined pairs = {}".format(idx, loss, num))
    
    training_loss /= len(dataloader.dataset)
    return training_loss

### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def inference(model, train_set, test_set, accuracy_calculator, mode="val"):
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    test_labels = test_labels.squeeze(1)
    if mode == "test":
        # Train and test sets do not share labels (they are secuences with different
        # vehicles), so we compute accuracy using only the test set
        ref_includes_query = True
        train_embeddings, train_labels = None, None       
    else:
        ref_includes_query = False
        train_embeddings, train_labels = get_all_embeddings(train_set, model)
        train_labels = train_labels.squeeze(1)

    accuracies = accuracy_calculator.get_accuracy(test_embeddings, test_labels, train_embeddings, train_labels, ref_includes_query)
    return accuracies