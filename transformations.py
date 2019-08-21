import random

import cv2
import numpy as np
import torch
from albumentations import (CLAHE, Blur, CenterCrop, Compose, ElasticTransform,
                            GaussNoise, GridDistortion, HorizontalFlip,
                            HueSaturationValue, IAAAdditiveGaussianNoise,
                            JpegCompression, MedianBlur, MotionBlur, Normalize,
                            OneOf, OpticalDistortion, RandomBrightness,
                            RandomContrast, RandomGamma,
                            RandomSizedBBoxSafeCrop, RandomSizedCrop, Resize,
                            RGBShift, Rotate, ShiftScaleRotate, ToFloat)

from albumentations.augmentations.bbox_utils import (
    convert_bbox_to_albumentations)

from albumentations.imgaug.transforms import IAASharpen
from albumentations.pytorch.transforms import ToTensor as AlbToTensor

from config import CONFIG





# INPUTS_TRANSFORMATIONS = Compose([
#     LongestMaxSize(256),
#     Flip(p=0.3),
#     Rotate(p=0.3),
#     CLAHE(p=0.5),
#     RandomBrightnessContrast(p=0.5),
#     RandomGamma(p=0.5),
#     AlbToTensor()])
# ]:

AUGM_TRANSFORMATIONS = Compose([
    HorizontalFlip(p=0.5),
    Rotate(p=0.3, limit=10),
    Blur(blur_limit=1, p=0.1),
    OneOf([
        RandomContrast(),
        RandomGamma(),
        RandomBrightness(),
    ], p=0.3),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    RandomSizedCrop(min_max_height=(176, 256),
                    height=CONFIG.im_size, width=CONFIG.im_size, p=0.25)
], p=1)


INPUTS_TRANSFORMATIONS = Compose([Resize(height=CONFIG.im_size, width=CONFIG.im_size)], p=1)
# INPUTS_TRANSFORMATIONS = Compose([], p=1)
OUTPUTS_TRANSFORMATIONS = Compose([AlbToTensor()], p=1)


def get_bboxes(mask):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    contours, _ = cv2.findContours((mask > 0).astype(
        np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [convert_bbox_to_albumentations(
        cv2.boundingRect(contour), 'coco', rows=mask.shape[0], cols=mask.shape[1])
        for contour in contours]
    return bboxes


def handle_input_image(image, mask=None):
    if len(image.shape) == 2:
        image = np.tile(image[..., None], (1, 1, 3))
    assert image.shape[2] == 3, image.shape
    if mask is not None:
        if len(mask.shape) == 2:
            mask = np.tile(mask[..., None], (1, 1, 3))
        assert mask.shape[2] == 3, mask.shape
        res = INPUTS_TRANSFORMATIONS(image=image, mask=mask)
        return res['image'], res['mask']
    else:
        res = INPUTS_TRANSFORMATIONS(image=image)
        return res['image'], None


def create_augmented_image(image, mask):
    res = AUGM_TRANSFORMATIONS(image=image,
                               mask=mask)

    return res


def handle_transformations(image, mask, augment):
    image, mask = handle_input_image(image, mask)
    if augment:
        res = create_augmented_image(image=image, mask=mask)

    else:
        if mask is not None:
            res = dict(image=image, mask=mask)
        else:
            res = dict(image=image)
    res = OUTPUTS_TRANSFORMATIONS(**res)
    if 'mask' in res:

        res['mask'] = torch.from_numpy(
            (res['mask'].cpu().data.numpy() > 0).astype(
                np.uint8)).float().to(res['mask'].device)
        assert len(np.unique(res['mask'].cpu().data.numpy())) <= 2, \
            np.unique(res['mask'].cpu().data.numpy())
    # assert np.all(res['image'].cpu().data.numpy() <= 1)
    # assert np.all(res['image'].cpu().data.numpy() >= 0)

    return res
