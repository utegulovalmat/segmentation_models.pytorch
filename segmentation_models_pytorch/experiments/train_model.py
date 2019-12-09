import traceback
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import gc
import time
import numpy as np
import pandas as pd
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

from segmentation_models_pytorch.experiments.helpers import get_volume_fn_from_mask_fn
from segmentation_models_pytorch.experiments.helpers import NoMatchingModelException
from segmentation_models_pytorch.experiments.helpers import save_sample_image
from segmentation_models_pytorch.experiments.helpers import get_volume_paths
from segmentation_models_pytorch.experiments.helpers import format_history
from segmentation_models_pytorch.experiments.helpers import plot_graphs
from segmentation_models_pytorch.experiments.helpers import get_best_metrics
from segmentation_models_pytorch.experiments.helpers import get_datetime_str
from segmentation_models_pytorch.experiments.helpers import send_email
from segmentation_models_pytorch.experiments.helpers import arg_parser
from segmentation_models_pytorch.experiments.helpers import get_overlay_masks
from segmentation_models_pytorch.experiments.helpers import format_test_result_metrics

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
    batch_size: int,
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
    :param batch_size: batch size
    :param train_all: False means use 1 volume for training
    :param extract_slices: True - extract slices from volumes
    :param epochs: number of epochs to train
    :return:
    """
    global logger
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
    new_print("train", train_volumes, train_masks)
    new_print("valid", valid_volumes, valid_masks)
    new_print("test", test_volumes, test_masks)

    # Extract slices from volumes
    dataset_dir = "/".join(input_dir.split("/")[:-1])
    exported_slices_dir_train = dataset_dir + "/tif_slices_train/"
    exported_slices_dir_valid = dataset_dir + "/tif_slices_valid/"
    exported_slices_dir_test = dataset_dir + "/tif_slices_test/"
    new_print("Extract slices:", extract_slices)
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

    path, dirs, files = next(os.walk(exported_slices_dir_train))
    logger.info("exported_slices_dir_train: " + str(len(files) / 2))
    path, dirs, files = next(os.walk(exported_slices_dir_valid))
    logger.info("exported_slices_dir_valid: " + str(len(files) / 2))
    path, dirs, files = next(os.walk(exported_slices_dir_test))
    logger.info("exported_slices_dir_test: " + str(len(files) / 2))

    # Define datasets
    train_dataset = MriDataset(
        path=exported_slices_dir_train,
        augmentation=get_train_augmentation(),
        preprocessing=get_preprocessing(),
    )
    new_print(len(train_dataset))
    valid_dataset = MriDataset(
        path=exported_slices_dir_valid,
        augmentation=get_test_augmentation(),
        preprocessing=get_preprocessing(),
    )
    new_print(len(valid_dataset))
    test_dataset = MriDataset(
        path=exported_slices_dir_test,
        augmentation=get_test_augmentation(),
        preprocessing=get_preprocessing(),
    )
    new_print(len(test_dataset))

    image, mask = train_dataset[150]
    new_print("Image and mask dimensions")
    new_print(type(image), image.shape, mask.shape)

    # Show sample image
    save_sample_image(image[0], mask[0], output_dir)

    # Create segmentation model with pretrained encoder
    classes = ["1"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Use device: " + device)
    activation = "sigmoid" if len(classes) == 1 else "softmax"
    logger.info("Activation: " + activation)
    encoder_weights = "imagenet"
    if model_name == "unet":
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=len(classes),
            activation=activation,
            in_channels=1,
        )
    elif model_name == "fcn":
        # TODO: add flexibility with encoder selection
        model = smp.FCN(encoder_name=encoder, classes=len(classes),)
    elif model_name == "fpn":
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=len(classes),
            activation=activation,
            in_channels=1,
        )
    elif model_name == "linknet":
        model = smp.Linknet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=len(classes),
            activation=activation,
            in_channels=1,
        )
    elif model_name == "pspnet":
        model = smp.PSPNet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=len(classes),
            activation=activation,
            in_channels=1,
        )
    else:
        raise NoMatchingModelException

    # Define metrics, loss and optimizer
    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
    metrics = [
        smp.utils.metrics.IoU(eps=1.0),
        smp.utils.metrics.Fscore(eps=1.0),
    ]
    # TODO: try BCEDiceLoss
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
        batch_size=batch_size,
        num_workers=12,
        shuffle=True,
        # sampler=subset_sampler,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        # sampler=subset_sampler,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        # sampler=subset_sampler,
    )
    # Create epoch runners, it is a simple loop of iterating over DataLoader's samples
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
        new_print("\nEpoch: {}".format(epoch))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_history.append(train_logs)
        valid_history.append(valid_logs)

        if max_score < valid_logs["iou_score"]:
            max_score = valid_logs["iou_score"]
            torch.save(model, output_dir + "/best_model.pth")
            logger.info("Model saved at epoch: " + str(epoch))
            best_epoch = epoch
            early_stop_epochs = 0
        else:
            early_stop_epochs += 1
            logger.info("Early stop epochs = " + str(early_stop_epochs))
            if early_stop_epochs == 3:
                optimizer.param_groups[0]["lr"] = 1e-5
                new_print("Decrease learning rate to 1e-5")
            if early_stop_epochs == 5:
                logger.info("Early stopping at epoch: " + str(epoch))
                break

    history = format_history(train_history, valid_history)
    best_train_row, best_valid_row = get_best_metrics(history)
    plot_graphs(history, output_dir)
    logger.info("Train performance metrics")
    logger.info(best_train_row)
    logger.info("Validation performance metrics")
    logger.info(best_valid_row)

    # Evaluate model on test set, load best saved checkpoint
    model = torch.load(output_dir + "/best_model.pth")
    test_epoch = smp.utils.train.ValidEpoch(
        model, loss=loss, metrics=metrics, device=device, verbose=True
    )
    best_test_row = test_epoch.run(test_loader)
    best_test_row = format_test_result_metrics(best_test_row)
    logger.info("Test performance metrics")
    logger.info(best_test_row)

    # Visualize predictions
    for idx in range(0, len(test_dataset), 10):
        image, gt_mask = test_dataset[idx]
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
    message = "Model: <strong>" + model_name + "-" + encoder + "</strong><br>"
    message += "Extract slices: " + str(extract_slices) + "<br>"
    message += "Use axis: " + use_axis + "<br>"
    message += "Training volumes: " + str(len(train_masks)) + "<br>"
    message += "Best valid epoch: " + str(best_epoch) + "<br>"
    message += "Train_|_" + best_train_row + "<br>"
    message += "Valid_|_" + best_valid_row + "<br>"
    message += "Test__|_" + best_test_row + "<br>"
    return message


def main():
    """
    source ~/ml-env3/bin/activate
    python -m segmentation_models_pytorch.experiments.train_model -in /home/a/Thesis/datasets/mri/final_dataset --train_all all --extract_slices 0

    nohup python -m segmentation_models_pytorch.experiments.train_model -in /datastore/home/segnet/datasets --train_all all --extract_slices 0 &
    echo 8083 >> last_pid.txt
    tail nohup.out -f
    """
    global logger
    args = arg_parser().parse_args()
    base_path = "segmentation_models_pytorch/experiments/"
    pipline_file = "segmentation_models_pytorch/experiments/pipeline.csv"
    pipeline = pd.read_csv(pipline_file, dtype={"axis": str, "epochs": int},)
    for idx, (done, model, encoder, axis, epochs, batch) in pipeline.iterrows():
        if done == "yes":
            continue
        cur_datetime = get_datetime_str()
        output_dir = base_path + "-".join([model, encoder, axis, cur_datetime])
        os.mkdir(output_dir)
        logger = get_logger(output_dir)
        logger.info("done, model, encoder, axis, epochs, batch")
        logger.info(
            " ".join([str(i) for i in [done, model, encoder, axis, epochs, batch]])
        )
        try:
            logger.info("\n\n\nStart training " + output_dir + "\n\n")
            result = train_model(
                model_name=model,
                encoder=encoder,
                output_dir=output_dir,
                axis=axis,
                epochs=epochs,
                input_dir=args.input_dir,
                train_all=args.train_all == "all",
                extract_slices=args.extract_slices == 1,
                batch_size=batch,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            logger.info("Finish")
            title = model + "-" + encoder + " SUCCESS"
            prefix = "Training finished with status: " + title + "\n\n"
            message = prefix + result
            logger.info(result + "\n\n" + "=" * 100)
            mask = (pipeline["model"] == model) & (pipeline["encoder"] == encoder)
            pipeline["done"][mask] = "yes"
            pipeline.to_csv(pipline_file, index=False)
            logger.info("Send email")
            print(message)
            send_email(title=title, message=message)
        except Exception as e:
            logger.error("Exception")
            logger.error(str(e))
            logger.error(traceback.format_exc())
            title = model + "-" + encoder + " FAILED"
            logger.info("Send email")
            send_email(title=title, message=traceback.format_exc())
        break
    return 0


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

    This is for axis 0, all volumes
    ls tif_slices_valid | wc -l ## 490
    ls tif_slices_test | wc -l ## 352
    ls tif_slices_train | wc -l ## 5372


    """
    sys.exit(main())
