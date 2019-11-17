import torch
import os
import cv2
import shutil
import albumentations as A
import matplotlib.pyplot as plt
import nrrd
import numpy as np
from PIL import Image

ORIENTATION = {'coronal': "COR", 'axial': "AXI", 'sagital': "SAG"}


def iou(pr, gt, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Source:
        https://github.com/catalyst-team/catalyst/
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union

jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()


    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score

#################################################
#### Useful functions ###########################
#################################################

def combine_masks(image, mask):
    """
    Overlay mask layers into single mask
    Add mask to the image into single image
    """
    masksum = np.zeros(image.shape[:2])
    masked = np.zeros(image.shape[:2])
    for idx in range(mask.shape[2]):
        masksum += mask[:, :, idx] * 2 ** (idx + 1)
        masked += image[:, :, 0] * mask[:, :, idx]
    return masksum, masked


def get_train_augmentation(hw_len=512):
    transform = [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomScale(p=0.5),
        A.PadIfNeeded(min_height=hw_len, min_width=hw_len, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
        A.CenterCrop(height=hw_len, width=hw_len, always_apply=True),
    ]
    return A.Compose(transform)


def get_test_augmentation(hw_len=512):
    transform = [
        A.PadIfNeeded(min_height=hw_len, min_width=hw_len, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
        A.CenterCrop(height=hw_len, width=hw_len, always_apply=True),
    ]
    return A.Compose(transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        # A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)


def visualize(**images):
    """Plot images in one row.
    Helper function for data visualization"""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, idx + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def rotate_orientation(volume_data, volume_label, orientation=ORIENTATION['coronal']):
    """Return rotated matrix to get differnt views ralative to submited 3D volumes"""
    if orientation == ORIENTATION['coronal']:
        return volume_data.transpose((2, 0, 1)), volume_label.transpose((2, 0, 1))
    elif orientation == ORIENTATION['axial']:
        return volume_data.transpose((1, 2, 0)), volume_label.transpose((1, 2, 0))
    elif orientation == ORIENTATION['sagital']:
        return volume_data, volume_label
    else:
        raise ValueError("Invalid value for orientation. Pleas see help")


def remove_black(data, labels, only_with_target=False):
    clean_data, clean_labels = [], []
    for i, frame in enumerate(labels):
        unique, counts = np.unique(frame, return_counts=True)
        # if only_with_target and len(unique) == 1:
        #    continue
        if counts[0] / sum(counts) < .99:
            clean_labels.append(frame)
            clean_data.append(data[i])
    return np.array(clean_data), np.array(clean_labels)


def normilize_mean_std(volume):
    # now normalize each modality with its mean and standard deviation (computed within the brain mask)
    mean = volume.mean()
    std = volume.std()
    volume = (volume - mean) / (std + 1e-8)
    # imgs_npy[i][brain_mask == 0] = 0
    return volume


def remove_all_blacks(image, mask, only_with_target=False):
    image, mask = rotate_orientation(image, mask, orientation=ORIENTATION['coronal'])
    image, mask = remove_black(image, mask, only_with_target)
    image, mask = rotate_orientation(image, mask, orientation=ORIENTATION['axial'])
    image, mask = remove_black(image, mask, only_with_target)
    image, mask = rotate_orientation(image, mask, orientation=ORIENTATION['sagital'])
    image, mask = remove_black(image, mask, only_with_target)
    return image, mask


def round_clip_0_1(x):
    """Remove values gt and lt 0 and 1"""
    return x.round().clip(0, 1)


def normalize_0_1(x):
    x_max = np.percentile(x, 99)
    x_min = np.percentile(x, 1)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def read_volume(filepath):
    img_data, header = nrrd.read(filepath)
    #     img = nib.load(filepath)
    #     img = nib.as_closest_canonical(img)
    #     img_data = img.get_fdata()
    return img_data


def read_pil_image(filepath):
    new_img = Image.open(filepath)
    new_img = np.array(new_img.getdata()).reshape(new_img.size[1], new_img.size[0])
    return new_img


def read_slices(images, masks):
    _images, _masks = [], []
    for image_fn, mask_fn in zip(images, masks):
        # Get volume and mask files by filepath
        image = read_volume(image_fn)
        mask = np.uint8(read_volume(mask_fn))
        # Remove black slices from all sides
        image, mask = remove_all_blacks(image, mask, only_with_target=True)
        _images.append(image)
        _masks.append(mask)
    return _images, _masks


def load_image_and_mask(slice_index, single_dimension, use_dimension, image_volumes, mask_volumes):
    """ Extracts volume slice with index `slice_index`

    Volumes must be corresponding orders.

    image_volumes:      3D numpy array
    mask_volumes:      3D numpy array
    single_dimension:   Use all 3D volume or 1 view dimension
                        True / False
    use_dimension:      Use only one of dimensions of 3D volume
                        can be use_dimension_0/use_dimension_1/use_dimension_2

    """
    image, mask = None, None
    for _image, _mask in zip(image_volumes, mask_volumes):
        if single_dimension:  # <-------------------------------- single dimension
            if use_dimension == 'use_dimension_0':
                img_shape = _image.shape[0]
                if slice_index >= img_shape:
                    slice_index -= img_shape
                    continue
                if slice_index >= img_shape:
                    slice_index -= img_shape
                else:
                    image = _image[slice_index, :, :]
                    mask = _mask[slice_index, :, :]
                    break
            elif use_dimension == 'use_dimension_1':
                img_shape = _image.shape[1]
                if slice_index >= img_shape:
                    slice_index -= img_shape
                    continue
                if slice_index >= img_shape:
                    slice_index -= img_shape
                else:
                    image = _image[:, slice_index, :]
                    mask = _mask[:, slice_index, :]
                    break
            elif use_dimension == 'use_dimension_2':
                img_shape = _image.shape[2]
                if slice_index >= img_shape:
                    slice_index -= img_shape
                    continue
                if slice_index >= img_shape:
                    slice_index -= img_shape
                else:
                    image = _image[:, :, slice_index]
                    mask = _mask[:, :, slice_index]
                    break
        else:  # <---------------------------------------------------- 3 dimensions
            img_shape = _image.shape
            # Get target volume
            if slice_index >= sum(img_shape):
                slice_index -= sum(img_shape)
                continue
            # Get target dimension and slice
            if slice_index >= img_shape[0]:
                slice_index -= img_shape[0]
            else:
                image = _image[slice_index, :, :]
                mask = _mask[slice_index, :, :]
                break
            if slice_index >= img_shape[1]:
                slice_index -= img_shape[1]
            else:
                image = _image[:, slice_index, :]
                mask = _mask[:, slice_index, :]
                break

            if slice_index > img_shape[2]:
                slice_index -= img_shape[2]
            else:
                image = _image[:, :, slice_index]
                mask = _mask[:, :, slice_index]
                break
    return image, mask


def save_slice_as_tiff_image(npy_orig, convert_format, output_dir: str, new_title: str, sanity_check=False):
    """Export numpy array to TIFF image.

    npy_orig:       numpy array containing image
    convert_format: must be 'F' for grayscale, 'L' for int values
    """
    assert new_title.endswith('.tiff')

    out_fname = output_dir + new_title
    new_img = Image.fromarray(npy_orig)
    new_img = new_img.convert(convert_format)
    new_img.save(out_fname)
    new_img = read_pil_image(out_fname)
    assert new_img.shape == npy_orig.shape

    if sanity_check:
        new_img = read_pil_image(out_fname)
        print(out_fname)
        print(np.sum(npy_orig), '<sum>', np.sum(new_img))
        print(np.unique(npy_orig), '<np.unique>', np.unique(new_img))
        print(np.max(npy_orig), '<max>', np.max(npy_orig))
        print(np.min(new_img), '<min>', np.min(new_img))
        print(type(npy_orig), '<type>', type(new_img))
        print(npy_orig.dtype, '<dtype>', new_img.dtype)
        print(npy_orig.shape, '<shape>', new_img.shape)
        print(npy_orig)
        print(new_img)
        print('\n')

    return npy_orig, new_img, out_fname


def extract_slices_from_volumes(
        images,
        masks,
        output_dir,
        skip_empty_mask=True,
        use_dimensions='012',
):
    """ Export volumes slices as separate TIFF images

    :param images: list of volume paths
    :param masks: list of volume paths
    :param output_dir: target folder path suffix
    :param skip_empty_mask: default True
    :param use_dimensions: which views to extract "012"

    # Usage example

    EXPORTED_SLICES_DIR = '/content/export_slices/'
    if os.path.isdir(EXPORTED_SLICES_DIR):
        print('rmtree folder', EXPORTED_SLICES_DIR)
        shutil.rmtree(EXPORTED_SLICES_DIR)
    os.mkdir(EXPORTED_SLICES_DIR)
    extract_slices_from_volumes(TRAIN, TRAIN_MASKS, output_dir=EXPORTED_SLICES_DIR)
    """
    if os.path.isdir(output_dir):
        print('rmtree before extracting slices:', output_dir)
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    image_volumes, mask_volumes = read_slices(images, masks)
    volume_shapes = [i.shape for i in image_volumes]
    print('volumes shapes', volume_shapes)

    # Export slices from dimension 0
    start_idx = 0
    slices_cnt_dim_0 = sum([x for x, y, z in volume_shapes])
    with_masks_dim_0 = slices_cnt_dim_0
    if '0' in use_dimensions:
        for idx in range(start_idx, slices_cnt_dim_0):
            npy_image, npy_mask = load_image_and_mask(
                idx,
                single_dimension=True,
                use_dimension='use_dimension_0',
                image_volumes=image_volumes,
                mask_volumes=mask_volumes,
            )
            if skip_empty_mask and len(np.unique(npy_mask)) == 1:
                with_masks_dim_0 -= 1
                continue
            base_fn = str(idx).zfill(4)
            image_fn = base_fn + '.tiff'
            mask_fn =  base_fn + '_seg.tiff'
            save_slice_as_tiff_image(npy_image, convert_format='F', output_dir=output_dir, new_title=image_fn)
            save_slice_as_tiff_image(npy_mask, convert_format='L', output_dir=output_dir, new_title=mask_fn)
        print('exported slices dim 0:', slices_cnt_dim_0)
        print('with mask dim 0:', with_masks_dim_0)

    # Export slices from dimension 1
    start_idx = slices_cnt_dim_0
    slices_cnt_dim_1 = sum([y for x, y, z in volume_shapes])
    with_masks_dim_1 = slices_cnt_dim_1
    if '1' in use_dimensions:
        for idx in range(start_idx, slices_cnt_dim_1 + start_idx):
            npy_image, npy_mask = load_image_and_mask(
                idx,
                single_dimension=True,
                use_dimension='use_dimension_1',
                image_volumes=image_volumes,
                mask_volumes=mask_volumes,
            )
            if skip_empty_mask and len(np.unique(npy_mask)) == 1:
                with_masks_dim_1 -= 1
                continue
            base_fn = str(idx).zfill(4)
            image_fn = base_fn + '.tiff'
            mask_fn = base_fn + '_seg.tiff'
            save_slice_as_tiff_image(npy_image, convert_format='F', output_dir=output_dir, new_title=image_fn)
            save_slice_as_tiff_image(npy_mask, convert_format='L', output_dir=output_dir, new_title=mask_fn)
        print('exported slices dim 1:', slices_cnt_dim_1)
        print('with mask dim 1:', with_masks_dim_1)

    # Export slices from dimension 2
    start_idx = slices_cnt_dim_0 + slices_cnt_dim_1
    slices_cnt_dim_2 = sum([z for x, y, z in volume_shapes])
    with_masks_dim_2 = slices_cnt_dim_2
    if '2' in use_dimensions:
        for idx in range(start_idx, slices_cnt_dim_2 + start_idx):
            npy_image, npy_mask = load_image_and_mask(
                idx,
                single_dimension=True,
                use_dimension='use_dimension_2',
                image_volumes=image_volumes,
                mask_volumes=mask_volumes,
            )
            if skip_empty_mask and len(np.unique(npy_mask)) == 1:
                with_masks_dim_2 -= 1
                continue
            base_fn = str(idx).zfill(4)
            image_fn = base_fn + '.tiff'
            mask_fn = base_fn + '_seg.tiff'
            save_slice_as_tiff_image(npy_image, convert_format='F', output_dir=output_dir, new_title=image_fn)
            save_slice_as_tiff_image(npy_mask, convert_format='L', output_dir=output_dir, new_title=mask_fn)
        print('exported slices dim 2:', slices_cnt_dim_2)
        print('with mask dim 2:', with_masks_dim_2)
    return True
