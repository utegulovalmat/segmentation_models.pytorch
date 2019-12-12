import argparse
import os

import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.experiments.helpers import get_volume_fn_from_mask_fn
from segmentation_models_pytorch.experiments.helpers import get_volume_paths
from segmentation_models_pytorch.utils.custom_functions import read_pil_image
from segmentation_models_pytorch.utils.custom_functions import normalize_0_1


def export_volume_to_dir(
    patient: str, volume: str, mask: str, use_axis: str, skip_empty_mask: bool,
):
    print(patient)
    smp.utils.custom_functions.extract_slices_from_volumes(
        images=[volume],
        masks=[mask],
        output_dir=patient,
        skip_empty_mask=skip_empty_mask,
        use_dimensions=use_axis,
    )


def load_model(model_path):
    pass


def run_inference():
    args = inference_arg_parser().parse_args()
    input_dir = args.input_dir

    # Get paths to volumes and masks
    mask_fns, fns = get_volume_paths(input_dir)
    n_volumes = 1  # 14 or 1
    masks = mask_fns[0:n_volumes]
    volumes = [get_volume_fn_from_mask_fn(fn) for fn in masks]

    for volume, mask in zip(volumes, masks):
        dataset_dir = "/".join(input_dir.split("/")[:-1])
        patient = volume.split("/")[-1]
        patient = patient.split(".")[0]
        exported_slices_dir = dataset_dir + "/" + patient + "/"
        export_volume_to_dir(
            patient=exported_slices_dir,
            volume=volume,
            mask=mask,
            use_axis="0",  # 012
            skip_empty_mask=True,
        )

        # TODO: Run model inference on exported dataset
        # model = load_model(model_path)

        # Export to PNG to view exported slices
        images, masks = get_slices_paths(exported_slices_dir)
        exported_slices_dir_png = exported_slices_dir + "png/"
        os.mkdir(exported_slices_dir_png)
        for idx, (image_path, mask_path) in enumerate(zip(images, masks)):
            image = read_pil_image(image_path)
            # image = normalize_0_1(image)
            mask = read_pil_image(mask_path)
            mask[mask > 0] = 1
            save_sample_image(
                image.T, mask.T, exported_slices_dir_png + str(idx).zfill(5) + ".png"
            )


def save_sample_image(image, mask, output_file):
    fig = plt.figure(figsize=(14, 6))
    rows, columns = 1, 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(image, cmap="gray")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(mask, cmap="gray")
    # plt.show()
    plt.savefig(fname=output_file)
    plt.close(fig)


def inference_arg_parser():
    parser = argparse.ArgumentParser(
        description="split 3d image into multiple 2d images"
    )
    parser.add_argument(
        "-in",
        "--input_dir",
        type=str,
        default="/home/a/Thesis/datasets/mri/final_dataset/",
        help="path to NRRD image/volume directory",
    )
    return parser


def get_slices_paths(input_dir: str):
    """Get paths to volumes

    :param input_dir: directory with MRI volumes
    :return: sorted list of paths to masks and volume images
    """
    fns = []
    mask_fns = []
    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if "seg" in filename:
                mask_fns.append(os.path.join(dirname, filename))
            else:
                fns.append(os.path.join(dirname, filename))
    fns = sorted(fns)
    mask_fns = sorted(mask_fns)
    return fns, mask_fns


if __name__ == "__main__":
    run_inference()
