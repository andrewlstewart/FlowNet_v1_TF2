"""
"""


from typing import Tuple, List, Union
from pathlib import Path

import numpy as np
import tensorflow as tf

import utils_io as uio


def get_training_min_max(root_path: Path) -> List[np.ndarray]:
    flo_list = []
    for idx, flo in enumerate(root_path.glob('*.flo')):
        # idx isn't required but makes troubleshooting easier
        v = uio.read(str(flo))
        flo_list.append([tf.math.reduce_min(v), tf.math.reduce_max(v)])
    flo_list_stacked = tf.stack(flo_list, axis=0)
    return [tf.math.reduce_min(flo_list_stacked), tf.math.reduce_max(flo_list_stacked)]


def normalize_images(img: np.array) -> np.ndarray:
    # this, tf.image.convert_image_dtype(img, tf.float16), is producing strange results, ie all zero values...
    return tf.cast(img, tf.float16) / 255  # Casting to float16 may reduce precision but hopefully it increases speed


def normalize_flo(flo: np.array, scale_factors: Tuple[float, float]) -> np.ndarray:
    # range -> [-1, 1]
    if isinstance(flo, list):
        flo = tf.stack(flo, axis=0)
    return ((flo - scale_factors[0]) / (scale_factors[1] - scale_factors[0]) - 0.5) * 2


def denormalize_flo(flo: np.ndarray, scale_factors: Tuple[float, float]) -> np.ndarray:
    return (scale_factors[1] - scale_factors[0]) * (0.5 + flo/2) + scale_factors[0]


def get_train_val_test(image_names: List[Path],
                       train_ratio: Union[float, int],
                       test_ratio: Union[float, int],
                       shuffle: bool = True) -> Tuple[List[Path], List[Path], List[Path]]:
    """ Get the train, val, and test sets from a list of all image paths.
        The test set is the last block and shouldn't be handled until after hyperparameter tuning.
        This function is sloppy and can easily be broken.  Reasonable values, such as train_ratio=0.7 and test_ratio=0.1
        will return a train_ratio of 0.7, a validation_ratio of 0.2, and a test_ratio of 0.1 and work fine.
    """
    if (not 0 < train_ratio < 1) or (not 0 < test_ratio < 1) or (train_ratio + test_ratio >= 1):
        raise Exception(f"Why have you done this. Train ratio: {train_ratio}, val ratio: {1-train_ratio-test_ratio}, Test ratio: {test_ratio}.")

    n_images = len(image_names)
    test = image_names[int(-test_ratio*n_images):]  # Don't use the last set of images until done hyperparameter tuning

    image_names = image_names[:int(-test_ratio*n_images)]

    n_train = int(train_ratio * n_images)
    if shuffle:
        np.random.shuffle(image_names)
    train = image_names[:n_train]
    val = image_names[n_train:]

    return train, val, test
