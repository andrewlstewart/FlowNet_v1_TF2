"""
"""


from typing import Tuple
from pathlib import Path

import numpy as np
import tensorflow as tf

import utils_io as uio


def get_training_min_max(root_path: Path):
    flo_list = []
    for idx, flo in enumerate(root_path.glob('*.flo')):
        # idx isn't required but makes troubleshooting easier
        v = uio.read(str(flo))
        flo_list.append([tf.math.reduce_min(v), tf.math.reduce_max(v)])
    flo_list_stacked = tf.stack(flo_list, axis=0)
    return [tf.math.reduce_min(flo_list_stacked), tf.math.reduce_max(flo_list_stacked)]


def normalize_images(img: np.array):
    # this, tf.image.convert_image_dtype(img, tf.float16), is producing strange results, ie all zero values... 
    return tf.cast(img, tf.float16) / 255  # Casting to float16 may reduce precision but hopefully it increases speed


def normalize_flo(flo: np.array, scale_factors: Tuple[float, float]):
    # range -> [-1, 1]
    return ((tf.stack(flo, axis=0) - scale_factors[0]) / (scale_factors[1] - scale_factors[0]) - 0.5) * 2
