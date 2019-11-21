import copy
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        # Defining the layers, 128, 64, 10 units each
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        """ Forward pass through the network, returns the output logits """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = 1 - (
        (2.0 * intersection + smooth)
        / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    )

    return loss.mean()


def calc_loss(pred, target, metrics):
    criterion = nn.NLLLoss()
    loss = criterion(pred, target)
    metrics["loss"] += loss.data.cpu().numpy() * target.size(0)
    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def view_classify(img, ps, version="MNIST"):
    """ Function for viewing an image and it's predicted classes.
    """
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis("off")
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(
            [
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle Boot",
            ],
            size="small",
        )
    ax2.set_title("Class Probability")
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def sequential_model():
    # Hyperparameters for our network
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10
    # Build a feed-forward network
    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], output_size),
        nn.Softmax(dim=1),
    )
    return model


def main():
    # Define a transform to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
    )
    # Download and load the training data
    trainset = datasets.MNIST(
        "~/.pytorch/MNIST_data/", download=True, train=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True,)

    testset = datasets.MNIST(
        "~/.pytorch/MNIST_data/", download=True, train=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True,)

    # Init model
    # model = FCN()
    # print(model.fc1)
    model = sequential_model()
    # print(model[0])
    # print(summary(model, input_size=(1, 64, 64)))

    # Train model ====================
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    best_model = train(
        model=model,
        optimizer=optimizer,
        num_epochs=3,
        train_loader=trainloader,
        valid_loader=testloader,
    )
    print(best_model)
    # print(best_model.state_dict())


def train(model, optimizer, num_epochs, train_loader, valid_loader):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    # Freeze backbone layers
    # for l in model.base_layers:
    #    for param in l.parameters():
    #        param.requires_grad = False

    device = torch.device("cpu")

    dataloaders = {
        "train": train_loader,
        "val": valid_loader,
    }
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                for param_group in optimizer.param_groups:
                    print("LR", param_group["lr"])
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.view(inputs.shape[0], -1)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics["loss"] / epoch_samples

            # deep copy the model
            if phase == "val" and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print("{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    print("Best val loss: {:4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


if __name__ == "__main__":
    """
    cd ~/Downloads/Theses/Repos/segmentation_models.pytorch
    source ~/ml-env3/bin/activate
    python -m segmentation_models_pytorch.fcn.model
    """
    main()
