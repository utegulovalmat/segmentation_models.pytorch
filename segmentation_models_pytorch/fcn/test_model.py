"""
Run script with:

cd ~/Downloads/Theses/Repos/segmentation_models.pytorch
source ~/ml-env3/bin/activate
python -m segmentation_models_pytorch.fcn.test_model
"""
import copy
from collections import defaultdict

import time
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from .utils import view_classify
from .utils import print_metrics
from .utils import calc_loss
from .model import FCN


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
    model = FCN()
    # print(model.fc1)

    # Train model
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    best_model = train(
        model=model,
        optimizer=optimizer,
        num_epochs=1,
        train_loader=trainloader,
        valid_loader=testloader,
    )
    print(best_model)
    # print(best_model.state_dict())

    # View prediction
    images, labels = next(iter(testloader))
    images.resize_(64, 1, 784)
    img_idx = 0
    ps = best_model.forward(images[img_idx, :])
    print(ps)  # softmax output probabilities
    ps = torch.exp(ps)  # back to percents
    view_classify(images[0].view(1, 28, 28), ps)


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
                inputs = inputs.view(inputs.shape[0], -1)  # TODO: remove for conv
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    logps = model(inputs)
                    loss = calc_loss(logps, labels, metrics)

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
    main()
