import warnings

warnings.filterwarnings("ignore")

import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.custom_functions import get_train_augmentation
from segmentation_models_pytorch.utils.custom_functions import get_test_augmentation
from segmentation_models_pytorch.utils.custom_functions import get_preprocessing
from segmentation_models_pytorch.utils.data import MriDataset

import numpy as np
import argparse
import logging
import sys
import os
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (7, 7)

from torch.utils.data import DataLoader
import traceback

logger = None


class NoMatchingModelException(Exception):
    pass


def new_print(*args):
    global logger
    return logger.info(" ".join(str(a) for a in args))


def get_fns(input_dir: str):
    """Get paths to volumes

    :param input_dir: directory with MRI volumes
    :return: sorted list of paths to masks and volume images
    """
    fns = []
    mask_fns = []
    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if "nrrd" in filename and "seg" not in filename:
                if "label.nrrd" in filename:
                    mask_fns.append(os.path.join(dirname, filename))
                else:
                    fns.append(os.path.join(dirname, filename))
    mask_fns = sorted(mask_fns)
    fns = sorted(fns)
    return mask_fns, fns


def get_fn(mask_filename):
    """Extract volume path from mask path"""
    return mask_filename.replace("S-label", "")


def train_model(
    model_name: str,
    encoder: str,
    input_dir: str,
    output_dir: str,
    train_all: bool,
    axis: str = "012",
    extract_slices: bool = True,
    epochs: int = 1,
):
    """Script to train networks on MRI dataset

    :param model_name: unet/pspnet/fpn/linknet
    :param encoder: see encoders list at https://github.com/utegulovalmat/segmentation_models.pytorch
    :param input_dir: path to folder with volumes and masks
    :param output_dir: output folder for model predictions
    :param axis: which axis should be used for training [0|1|2]
    :param train_all: False means use 1 volume for training
    :param extract_slices: True - extract slices from volumes
    :return:
    """
    global logger
    print = new_print
    use_axis = axis

    # Get paths to volumes and masks
    mask_fns, fns = get_fns(input_dir)
    n_volumes = 12 if train_all else 1
    train_masks = mask_fns[0:n_volumes]
    train_volumes = [get_fn(fn) for fn in train_masks]
    valid_masks = mask_fns[12:13]
    valid_volumes = [get_fn(fn) for fn in valid_masks]
    test_masks = mask_fns[13:14]
    test_volumes = [get_fn(fn) for fn in test_masks]
    print(
        train_volumes, train_masks, valid_volumes, valid_masks, test_volumes, test_masks
    )

    # Extract slices from volumes
    dataset_dir = "/".join(input_dir.split("/")[:-1])
    exported_slices_dir_train = dataset_dir + "/tif_slices_train/"
    exported_slices_dir_valid = dataset_dir + "/tif_slices_valid/"
    exported_slices_dir_test = dataset_dir + "/tif_slices_test/"
    print("Extract slices:", extract_slices)
    if extract_slices:
        smp.utils.custom_functions.extract_slices_from_volumes(
            images=train_volumes,
            masks=train_masks,
            output_dir=exported_slices_dir_train,
            skip_empty_mask=True,
            use_dimensions=use_axis,
        )
        smp.utils.custom_functions.extract_slices_from_volumes(
            images=valid_volumes,
            masks=valid_masks,
            output_dir=exported_slices_dir_valid,
            skip_empty_mask=True,
            use_dimensions=use_axis,
        )
        smp.utils.custom_functions.extract_slices_from_volumes(
            images=test_volumes,
            masks=test_masks,
            output_dir=exported_slices_dir_test,
            skip_empty_mask=True,
            use_dimensions=use_axis,
        )

    # Define datasets
    train_dataset = MriDataset(
        path=exported_slices_dir_train,
        augmentation=get_train_augmentation(),
        preprocessing=get_preprocessing(),
    )
    print(len(train_dataset))
    valid_dataset = MriDataset(
        path=exported_slices_dir_valid,
        augmentation=get_test_augmentation(),
        preprocessing=get_preprocessing(),
    )
    print(len(valid_dataset))
    test_dataset = MriDataset(
        path=exported_slices_dir_test,
        augmentation=get_test_augmentation(),
        preprocessing=get_preprocessing(),
    )
    print(len(test_dataset))

    image, mask = train_dataset[150]
    print("Image and mask dimensions")
    print(type(image), image.shape, mask.shape)

    # %% [code]
    w, h = 10, 10
    fig = plt.figure(figsize=(8, 8))
    rows, columns = 1, 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(image[0])
    fig.add_subplot(rows, columns, 2)
    plt.imshow(mask[0])
    # plt.show()
    plt.savefig(fname=output_dir + "/input_sample.png")

    classes = ["1"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Use device: " + device)
    activation = "sigmoid" if len(classes) == 1 else "softmax"
    logger.info("Activation: " + activation)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # create segmentation model with pretrained encoder
    if model_name == "unet":
        encoder_weights = "imagenet"
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=len(classes),
            activation=activation,
            in_channels=1,
        )
    elif model_name == "fcn":
        model = smp.FCN(classes=len(classes),)
    elif model_name == "fpn":
        model = smp.FPN("resnet34", in_channels=1)
    else:
        raise NoMatchingModelException

    # %% [code]
    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
    loss = smp.utils.losses.DiceLoss(eps=1.0)
    metrics = [
        smp.utils.metrics.IoU(eps=1.0),
        smp.utils.metrics.Fscore(eps=1.0),
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.Adam([
    #     {'params': model.decoder.parameters(), 'lr': 1e-4},
    #     # decrease lr for encoder in order not to ruin
    #     # pre-trained weights with large gradients on training start
    #     {'params': model.encoder.parameters(), 'lr': 1e-6},
    # ])

    # %% [code]
    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model, loss=loss, metrics=metrics, device=device, verbose=True,
    )
    return

    # %% [code]
    max_score = 0
    train_history = []
    valid_history = []
    for i in range(0, epochs):
        print("\nEpoch: {}".format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_history.append(train_logs)
        valid_history.append(valid_logs)
        # do something (save model, change lr, etc.)
        if max_score < valid_logs["iou_score"]:
            max_score = valid_logs["iou_score"]
            torch.save(model, "./best_model.pth")
            print("Model saved!")

        if i == 25:
            optimizer.param_groups[0]["lr"] = 1e-5
            print("Decrease decoder learning rate to 1e-5!")

    # %% [code]
    history = {
        "loss": [],
        "iou_score": [],
        "f-score": [],
        "val_loss": [],
        "val_iou_score": [],
        "val_f-score": [],
    }
    for train_log, valid_log in zip(train_history, valid_history):
        history["loss"].append(train_log["dice_loss"])
        history["iou_score"].append(train_log["iou_score"])
        history["f-score"].append(train_log["fscore"])
        history["val_loss"].append(valid_log["dice_loss"])
        history["val_iou_score"].append(valid_log["iou_score"])
        history["val_f-score"].append(valid_log["fscore"])

    # %% [code]
    best_train_loss = 1e10
    best_train_row = ""
    for idx, (loss, iou, dice) in enumerate(
        zip(history["loss"], history["iou_score"], history["f-score"])
    ):
        if loss < best_train_loss:
            best_train_loss = loss
            epoch = "epoch: " + str(idx)
            loss = "loss: {:.5}".format(loss)
            iou = "iou: {:.5}".format(iou)
            dice = "dice: {:.5}".format(dice)
            best_train_row = " ".join([epoch, loss, iou, dice])

    best_valid_loss = 1e10
    best_valid_row = ""
    for idx, (loss, iou, dice) in enumerate(
        zip(history["val_loss"], history["val_iou_score"], history["val_f-score"])
    ):
        if loss < best_valid_loss:
            best_valid_loss = loss
            epoch = "epoch: " + str(idx)
            loss = "loss: {:.5}".format(loss)
            iou = "iou: {:.5}".format(iou)
            dice = "dice: {:.5}".format(dice)
            best_valid_row = " ".join([epoch, loss, iou, dice])

    print("Train", best_train_row)
    print("Valid", best_valid_row)

    # %% [code]
    # Plot training & validation iou_score values
    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.plot(history["iou_score"])
    plt.plot(history["val_iou_score"])
    plt.title("Model iou_score")
    plt.ylabel("iou_score")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.show()

    # %% [markdown]
    # # Evaluate model on test set

    # %% [code]
    # load best saved checkpoint
    best_model = torch.load("./best_model.pth")

    # %% [code]
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_epoch = smp.utils.train.ValidEpoch(
        best_model, loss=loss, metrics=metrics, device=device, verbose=True,
    )
    logs = test_epoch.run(test_loader)

    # %% [code]
    print(logs)

    # %% [markdown]
    # # Visualize predictions

    # %% [code]
    for i in range(0, 5):
        n = np.random.choice(len(test_dataset))
        image, gt_mask = test_dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy().round()

        smp.utils.custom_functions.visualize(
            image=image[0], gt_mask=gt_mask, pr_mask=pr_mask,
        )

    # TODO: remove exported slices


def arg_parser():
    parser = argparse.ArgumentParser(
        description="split 3d image into multiple 2d images"
    )
    parser.add_argument(
        "-m", "--model_name", type=str, help="one of unet/pspnet/fpn/linknet"
    )
    parser.add_argument(
        "-e",
        "--encoder",
        type=str,
        help="encoder name resnet34/resnet50/resnext50_32x4d/densenet121/efficientnet-b0/...",
    )
    parser.add_argument(
        "-in",
        "--input_dir",
        type=str,
        default="/home/a/Thesis/datasets/mri/final_dataset",
        help="path to NRRD image/volume directory",
    )
    parser.add_argument(
        "-t",
        "--train_all",
        type=str,
        default="one",
        help="use 'all' to train model on 12 volumes, else it will use 1 volume",
    )
    parser.add_argument(
        "-a",
        "--axis",
        type=str,
        default="012",
        help="axis of the 3d image array on which to sample the slices",
    )
    parser.add_argument(
        "--extract_slices",
        type=int,
        default=1,
        help="1 - extract slices, 0 - skip this step",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs",
    )
    return parser


def get_datetime_str():
    from datetime import datetime

    now = datetime.now()  # current date and time
    date = now.strftime("%Y%m%d")
    time = now.strftime("%H:%M:%S")
    return date + "-" + time


def main():
    """
    source ~/ml-env3/bin/activate
    python -m segmentation_models_pytorch.experiments.train_model -m unet -e resnet34 -in /home/segnet/dataset -a 012 -ex 0
    """
    global logger

    args = arg_parser().parse_args()

    base_path = "segmentation_models_pytorch/experiments/"
    cur_datetime = get_datetime_str()
    output_dir = (
        base_path
        + args.model_name
        + "-"
        + args.encoder
        + "-"
        + args.axis
        + "-"
        + cur_datetime
    )
    os.mkdir(output_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_filename = output_dir + "/train.log"
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    try:
        logger.info("Start")
        train_model(
            model_name=args.model_name,
            encoder=args.encoder,
            input_dir=args.input_dir,
            output_dir=output_dir,
            train_all=args.train_all == "all",
            axis=args.axis,
            extract_slices=args.extract_slices == 1,
            epochs=args.epochs,
        )
        logger.info("Finish")
        return 0
    except Exception as e:
        logger.error("Exception")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    """
    Encoders: https://github.com/qubvel/segmentation_models.pytorch

    Encoder	    Weights	    Params, M
    resnet18	imagenet	11M
    resnet34	imagenet	21M
    resnet50	imagenet	23M
    resnet101	imagenet	42M
    resnet152	imagenet	58M
    resnext50_32x4d	imagenet	22M
    resnext101_32x8d	imagenet
    instagram	86M
    resnext101_32x16d	instagram	191M
    resnext101_32x32d	instagram	466M
    resnext101_32x48d	instagram	826M
    dpn68	imagenet	11M
    dpn68b	imagenet+5k	11M
    dpn92	imagenet+5k	34M
    dpn98	imagenet	58M
    dpn107	imagenet+5k	84M
    dpn131	imagenet	76M
    vgg11	imagenet	9M
    vgg11_bn	imagenet	9M
    vgg13	imagenet	9M
    vgg13_bn	imagenet	9M
    vgg16	imagenet	14M
    vgg16_bn	imagenet	14M
    vgg19	imagenet	20M
    vgg19_bn	imagenet	20M
    senet154	imagenet	113M
    se_resnet50	imagenet	26M
    se_resnet101	imagenet	47M
    se_resnet152	imagenet	64M
    se_resnext50_32x4d	imagenet	25M
    se_resnext101_32x4d	imagenet	46M
    densenet121	imagenet	6M
    densenet169	imagenet	12M
    densenet201	imagenet	18M
    densenet161	imagenet	26M
    inceptionresnetv2	imagenet
    imagenet+background	54M
    inceptionv4	imagenet
    imagenet+background	41M
    efficientnet-b0	imagenet	4M
    efficientnet-b1	imagenet	6M
    efficientnet-b2	imagenet	7M
    efficientnet-b3	imagenet	10M
    efficientnet-b4	imagenet	17M
    efficientnet-b5	imagenet	28M
    efficientnet-b6	imagenet	40M
    efficientnet-b7	imagenet	63M
    mobilenet_v2	imagenet	2M
    xception	imagenet	22M
    """
    sys.exit(main())
