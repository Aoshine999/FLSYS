"""flsys: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner,DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip

from torch.amp import autocast, GradScaler

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def get_transforms():
    # pytorch_transforms = Compose(
    #     [ToTensor(), Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    # )
    pytorch_transforms = Compose(
        [RandomCrop(32), RandomHorizontalFlip(), ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] 
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch
    return apply_transforms

fds = None  # Cache FederatedDataset




def load_data(partition_id: int, num_partitions: int):
    """Load partition fashion_mnist data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions,partition_by="label",alpha=0.5,shuffle=True,seed=42,self_balancing=True)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)


    partition_train_test = partition_train_test.with_transform(get_transforms())
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader

def train(net, trainloader, epochs, lr,device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5,weight_decay=1e-5)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=1e-5)
    
    net.train()
    running_loss = 0.0
    

    for _ in range(epochs):
        for i,batch in enumerate(trainloader):
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()


            outputs = net(images.to(device))
            loss = criterion(outputs, labels.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def train1(net, trainloader, epochs, lr,device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=1e-5)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=1e-5)

    scaler = None
    if device.type == "cuda":  # Check if GPU is available
        scaler = GradScaler("cuda")
    
    net.train()
    running_loss = 0.0
    
    num_batches = len(trainloader) * epochs
    if num_batches == 0:
        return 0.0
    

    for _ in range(epochs):
        for i,batch in enumerate(trainloader):
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()

            # Use autocast for mixed precision if on CUDA
            if device.type == 'cuda':
                with autocast():
                    outputs = net(images.to(device))
                    loss = criterion(outputs, labels.to(device))
            else: # Standard training on CPU
                outputs = net(images.to(device))
                loss = criterion(outputs, labels.to(device))

            # Backward pass and optimizer step
            if device.type == 'cuda':
                # Scale loss and perform backward pass
                scaler.scale(loss).backward()
                # Unscale gradients and call optimizer.step()
                scaler.step(optimizer)
                # Update the scale for next iteration
                scaler.update()
            else: # Standard backward and step on CPU
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0


    # net.eval()
    
    with torch.no_grad(): # Disable gradient calculation for inference
        # Use autocast for mixed precision inference if on CUDA
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()


    accuracy = correct / len(testloader.dataset) if len(testloader.dataset) > 0 else 0.0
    loss = loss / len(testloader) if len(testloader) > 0 else 0.0

    return loss, accuracy

def test1(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0


    net.eval()

    with torch.no_grad(): # Disable gradient calculation for inference
        # Use autocast for mixed precision inference if on CUDA
        if device.type == 'cuda':
            with autocast():
                 for batch in testloader:
                    images = batch["img"].to(device)
                    labels = batch["label"].to(device)
                    outputs = net(images)
                    # Note: criterion typically outputs float32 even within autocast
                    loss += criterion(outputs, labels).item()
                    correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        else: # Standard inference on CPU
            for batch in testloader:
                 images = batch["img"].to(device)
                 labels = batch["label"].to(device)
                 outputs = net(images)
                 loss += criterion(outputs, labels).item()
                 correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()


    accuracy = correct / len(testloader.dataset) if len(testloader.dataset) > 0 else 0.0
    loss = loss / len(testloader) if len(testloader) > 0 else 0.0

    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

if __name__ == "__main__":

    x = torch.randn(1,3,32,32)


    