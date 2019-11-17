from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import os
import numpy as np
from .functions import read_pil_image
from .functions import normalize_0_1


EXPORTED_SLICES_DIR_TRAIN = './export_slices_train/'
EXPORTED_SLICES_DIR_VALID = './export_slices_valid/'
EXPORTED_SLICES_DIR_TEST = './export_slices_test/'


class MriDataset(BaseDataset):
    """
    # Usage example:

    %time
    get_training_augmentation = smp.utils.functions.get_training_augmentation
    get_test_augmentation = smp.utils.functions.get_test_augmentation

    train_dataset = MriDataset(mode='train', augmentation=get_training_augmentation(), preprocessing=None)
    print(len(train_dataset))

    valid_dataset = MriDataset(mode='valid', augmentation=get_training_augmentation(), preprocessing=None)
    print(len(valid_dataset))

    test_dataset = MriDataset(mode='test', augmentation=get_test_augmentation(), preprocessing=None)
    print(len(test_dataset))
    """
    CLASSES = ['1']

    def __init__(
            self,
            mode,
            augmentation=None,
            preprocessing=None,
    ):
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mode = mode  # train valid test
        if self.mode == 'train':
            self.exported_slices_dir = EXPORTED_SLICES_DIR_TRAIN
        elif self.mode == 'valid':
            self.exported_slices_dir = EXPORTED_SLICES_DIR_VALID
        elif self.mode == 'test':
            self.exported_slices_dir = EXPORTED_SLICES_DIR_TEST
        else:
            raise

        all_fns = sorted(os.listdir(self.exported_slices_dir))
        self.image_fns = [fn for fn in all_fns if 'seg' not in fn]
        self.mask_fns = [fn for fn in all_fns if 'seg' in fn]
        assert len(self.mask_fns) == len(self.image_fns)

        self.slices_cnt = len(self.mask_fns)

    def __getitem__(self, image_id):
        image = self.load_image(image_id)
        image = np.dstack([image, image, image])

        mask = self.load_mask(image_id)
        mask = np.expand_dims(mask, axis=0)  # add unit dimension

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def load_mask(self, image_id):
        mask_fn = self.exported_slices_dir + self.mask_fns[image_id]
        mask = read_pil_image(mask_fn)
        mask[mask > 0] = 1
        return mask.astype("uint8").T

    def load_image(self, image_id):
        image_fn = self.exported_slices_dir + self.image_fns[image_id]
        image = read_pil_image(image_fn)

        # Normalize image
        # image = normilize_mean_std(image)
        image = normalize_0_1(image)
        # Bin values to 0/255 range
        # image = exposure.equalize_hist(image) # put it 0/255 range
        # image = exposure.equalize_adapthist(data) [TODO:] try it
        # print(np.unique(image), image.min(), image.max())
        # img.view(width, height, 1).expand(-1, -1, 3)
        # print(image.shape, image.min(), image.max())
        return image.T

    def __len__(self):
        return self.slices_cnt
