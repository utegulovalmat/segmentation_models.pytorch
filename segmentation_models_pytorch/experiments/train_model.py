import traceback
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import argparse
import logging
import sys
import os
import matplotlib.pyplot as plt
import warnings
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.custom_functions import get_train_augmentation
from segmentation_models_pytorch.utils.custom_functions import get_test_augmentation
from segmentation_models_pytorch.utils.custom_functions import get_preprocessing
from segmentation_models_pytorch.utils.data import MriDataset

from .helpers import get_volume_fn_from_mask_fn
from .helpers import NoMatchingModelException
from .helpers import save_sample_image
from .helpers import get_volume_paths
from .helpers import format_history
from .helpers import plot_graphs
from .helpers import get_best_metrics
from .helpers import get_datetime_str
from .helpers import send_email
from .helpers import format_test_result_metrics

plt.rcParams["figure.figsize"] = (7, 7)
warnings.filterwarnings("ignore")

logger = None


def new_print(*args):
    global logger
    return logger.info(" ".join(str(a) for a in args))


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

    :param model_name: one of unet/pspnet/fpn/linknet/fcn
    :param encoder: see encoders list at https://github.com/utegulovalmat/segmentation_models.pytorch
    :param input_dir: path to folder with volumes and masks
    :param output_dir: output folder for model predictions
    :param axis: which axis should be used for training [0|1|2]
    :param train_all: False means use 1 volume for training
    :param extract_slices: True - extract slices from volumes
    :param epochs: number of epochs to train
    :return:
    """
    global logger
    print = new_print
    use_axis = axis

    # Get paths to volumes and masks
    mask_fns, fns = get_volume_paths(input_dir)
    n_volumes = 12 if train_all else 1
    train_masks = mask_fns[0:n_volumes]
    train_volumes = [get_volume_fn_from_mask_fn(fn) for fn in train_masks]
    valid_masks = mask_fns[12:13]
    valid_volumes = [get_volume_fn_from_mask_fn(fn) for fn in valid_masks]
    test_masks = mask_fns[13:14]
    test_volumes = [get_volume_fn_from_mask_fn(fn) for fn in test_masks]
    print("train", train_volumes, train_masks)
    print("valid", valid_volumes, valid_masks)
    print("test", test_volumes, test_masks)

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

    # Show sample image
    save_sample_image(image[0], mask[0], output_dir)

    # Create segmentation model with pretrained encoder
    classes = ["1"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Use device: " + device)
    activation = "sigmoid" if len(classes) == 1 else "softmax"
    logger.info("Activation: " + activation)
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

    # Define metrics, loss and optimizer
    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
    metrics = [
        smp.utils.metrics.IoU(eps=1.0),
        smp.utils.metrics.Fscore(eps=1.0),
    ]
    loss = smp.utils.losses.DiceLoss(eps=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.Adam([
    #     {'params': model.decoder.parameters(), 'lr': 1e-4},
    #     # decrease lr for encoder in order not to ruin
    #     # pre-trained weights with large gradients on training start
    #     {'params': model.encoder.parameters(), 'lr': 1e-6},
    # ])

    # Create DataLoaders
    subset_sampler = SubsetRandomSampler(indices=[150, 160])
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=12,
        # shuffle=True,
        sampler=subset_sampler,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        num_workers=4,
        # shuffle=False,
        sampler=subset_sampler,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        # shuffle=False,
        sampler=subset_sampler,
    )
    # Create epoch runners
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

    max_score = 0
    best_epoch = 0
    train_history = []
    valid_history = []
    early_stop_epochs = 0
    for epoch in range(0, epochs):
        print("\nEpoch: {}".format(epoch))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_history.append(train_logs)
        valid_history.append(valid_logs)
        # do something (save model, change lr, etc.)
        if max_score < valid_logs["fscore"]:
            max_score = valid_logs["fscore"]
            torch.save(model, output_dir + "/best_model.pth")
            print("Model saved at epoch:", epoch)
            best_epoch = epoch
            early_stop_epochs = 0
        else:
            early_stop_epochs += 1
            if early_stop_epochs == 3:
                logger.info("Early stopping at epoch: " + str(epoch))
                break

        # if epoch == 25:
        #     optimizer.param_groups[0]["lr"] = 1e-5
        #     print("Decrease decoder learning rate to 1e-5!")

    history = format_history(train_history, valid_history)
    best_train_row, best_valid_row = get_best_metrics(history)
    plot_graphs(history, output_dir)

    # Evaluate model on test set, load best saved checkpoint
    model = torch.load(output_dir + "/best_model.pth")
    test_epoch = smp.utils.train.ValidEpoch(
        model, loss=loss, metrics=metrics, device=device, verbose=True
    )
    best_test_row = test_epoch.run(test_loader)
    best_test_row = format_test_result_metrics(best_test_row)
    print("Test dataset performance metrics")
    print(best_test_row)

    # Visualize predictions
    for idx in range(0, 5):
        n = np.random.choice(len(test_dataset))
        image, gt_mask = test_dataset[n]
        gt_mask = gt_mask.squeeze()
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy().round()
        overlay_prediction = image[0] * pr_mask
        smp.utils.custom_functions.visualize(
            output_path=output_dir + "/" + str(idx) + ".png",
            image=image[0],
            gt_mask=gt_mask,
            pr_mask=pr_mask,
            overlay_prediction=overlay_prediction,
            overlay_masks=get_overlay_masks(gt_mask, pr_mask),
        )
    message = "Model: " + model_name + "-" + encoder + "\n"
    message += "Extract slices: " + str(extract_slices) + "\n"
    message += "Use axis: " + use_axis + "\n"
    message += "Training volumes: " + str(len(train_masks)) + "\n"
    message += "Best valid epoch: " + str(best_epoch) + "\n"
    message += "Train_|_" + best_train_row + "\n"
    message += "Valid_|_" + best_valid_row + "\n"
    message += "Test__|_" + best_test_row + "\n"
    return message


def get_overlay_masks(gt_mask, pr_mask):
    pr_mask[pr_mask > 0.5] = 1
    mask = gt_mask * pr_mask
    return mask


def main():
    """
    source ~/ml-env3/bin/activate
    python -m segmentation_models_pytorch.experiments.train_model -in /home/a/Thesis/datasets/mri/final_dataset --train_all all --extract_slices 0
    """
    global logger
    args = arg_parser().parse_args()
    base_path = "segmentation_models_pytorch/experiments/"
    pipeline = pd.read_csv(
        "segmentation_models_pytorch/experiments/pipeline.csv",
        dtype={"axis": str, "epochs": int},
    )
    for idx, (done, model, encoder, axis, epochs) in pipeline.iterrows():
        print(done, model, encoder, axis, epochs)
        if done == "yes":
            print("skip")
            continue
        cur_datetime = get_datetime_str()
        output_dir = base_path + "-".join([model, encoder, axis, cur_datetime])
        os.mkdir(output_dir)
        logger = get_logger(output_dir)
        try:
            logger.info("Start")
            result = train_model(
                model_name=model,
                encoder=encoder,
                output_dir=output_dir,
                axis=axis,
                epochs=epochs,
                input_dir=args.input_dir,
                train_all=args.train_all == "all",
                extract_slices=args.extract_slices == 1,
            )
            logger.info("Finish")
            logger.info("Send email")
            send_email(title=model + "-" + encoder + " SUCCESS", message=result)
        except Exception as e:
            logger.error("Exception")
            logger.error(str(e))
            logger.info("Send email")
            logger.error(traceback.format_exc())
            send_email(
                title=model + "-" + encoder + " FAILED", message=traceback.format_exc()
            )
    return 0


def arg_parser():
    parser = argparse.ArgumentParser(
        description="split 3d image into multiple 2d images"
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
        "--extract_slices",
        type=int,
        default=1,
        help="1 - extract slices, 0 - skip this step",
    )
    return parser


def get_logger(output_dir):
    global logger
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
    return logger


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
