"""
Run script with:

cd ~/Downloads/Theses/Repos/segmentation_models.pytorch
source ~/ml-env3/bin/activate
python -m segmentation_models_pytorch.fcn.test_model
"""
import copy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

from segmentation_models_pytorch.utils.losses import DiceLoss

from .model import FCN
from .utils import print_metrics
from .utils import masks_to_colorimg
from .utils import plot_side_by_side
from .simulation import generate_random_data


def main():
    # Generate some random images
    input_images, target_masks = generate_random_data(192, 192, count=1)
    for x in [input_images, target_masks]:
        print(x.shape)
        print(x.min(), x.max())

    # Change channel-order and make 3 channels for matplot
    # input_images_rgb = [
    #     (x.swapaxes(0, 2).swapaxes(0, 1)  # * -255 + 255
    #     ).astype(np.uint8) for x in input_images
    # ]
    # target_masks_rgb = [masks_to_colorimg(x) for x in input_images_rgb]
    # plot_side_by_side([input_images_rgb, target_masks_rgb])

    # Init model
    device = torch.device("cpu")
    model = FCN(classes=6).to(device)
    # print(model.fc1)

    # Train model
    # criterion = nn.BCELoss()
    criterion = DiceLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    best_model = train_model(model, criterion, optimizer, num_epochs=1)

    # optimizer = optim.Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    # )
    # criterion = nn.BCELoss()
    # best_model = train(
    #     model=model,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     num_epochs=1,
    # )
    # print(best_model)
    # print(best_model.state_dict())

    # View prediction
    images, labels = input_images, target_masks
    images, labels = torch.from_numpy(images).float(), torch.from_numpy(labels).float()
    # preds = best_model.forward(images).detach().numpy()
    preds = best_model.predict(images).detach().numpy()
    images = images.numpy()

    print("PREDS SHAPE", type(preds), preds.shape)
    # preds = preds[0, :]
    # import matplotlib.pyplot as plt
    # plt.imshow(preds[0])
    # plt.show()

    # Change channel-order and make 3 channels for matplot
    input_images_rgb = [
        (x.swapaxes(0, 2).swapaxes(0, 1)).astype(np.uint8) for x in images
    ]
    # Map each channel (i.e. class) to each color
    target_masks_rgb = [masks_to_colorimg(x) for x in preds]
    # # Left: Input image, Right: Target mask
    plot_side_by_side([input_images_rgb, target_masks_rgb])


# def train(model, optimizer, criterion, num_epochs):
#     best_model_weights = copy.deepcopy(model.state_dict())
#     best_loss = 1e10
#
#     # Freeze backbone layers
#     # for l in model.base_layers:
#     #    for param in l.parameters():
#     #        param.requires_grad = False
#
#     device = torch.device("cpu")
#
#     for epoch in range(num_epochs):
#         print("Epoch {}/{}".format(epoch, num_epochs - 1))
#         print("-" * 10)
#
#         since = time.time()
#
#         # Each epoch has a training and validation phase
#         for phase in ["train", "val"]:
#             if phase == "train":
#                 for param_group in optimizer.param_groups:
#                     print("Learning rate", param_group["lr"])
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()  # Set model to evaluate mode
#
#             metrics = defaultdict(float)
#             epoch_samples = 0
#
#             input_images, target_masks = generate_random_data(192, 192, count=2)
#             inputs = torch.from_numpy(input_images).float().to(device)
#             labels = torch.from_numpy(target_masks).float().to(device)
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward
#             # track history if only in train
#             with torch.set_grad_enabled(phase == "train"):
#                 preds = model(inputs)
#                 loss = criterion(preds, labels)
#                 # metrics["loss"] += loss.data.cpu().numpy() * labels.size(0)
#
#                 # backward + optimize only if in training phase
#                 if phase == "train":
#                     loss.backward()
#                     optimizer.step()
#
#             # statistics
#             epoch_samples += inputs.size(0)
#
#             print_metrics(metrics, epoch_samples, phase)
#             epoch_loss = metrics["loss"] / epoch_samples
#
#             # deep copy the model
#             if phase == "val" and epoch_loss < best_loss:
#                 print("saving best model")
#                 best_loss = epoch_loss
#                 best_model_weights = copy.deepcopy(model.state_dict())
#
#         time_elapsed = time.time() - since
#         print("{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
#
#     print("Best val loss: {:4f}".format(best_loss))
#
#     # load best model weights
#     model.load_state_dict(best_model_weights)
#     return model


def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        device = torch.device("cpu")
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            batch_size = 2
            epoch_steps = 10
            for i in range(epoch_steps):
                input_images, target_masks = generate_random_data(
                    192, 192, count=batch_size
                )
                print(input_images.shape, target_masks.shape)
                inputs = torch.from_numpy(input_images).float()
                labels = torch.from_numpy(target_masks).float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    print("outputs shape", outputs.shape)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / (batch_size * epoch_steps)

            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            # deep copy the model
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val loss: {:4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    main()
