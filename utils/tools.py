import os

import commentjson
import numpy as np
import random
import tensorflow as tf
from nets import ssd_vgg_300, ssd_common, np_methods
from PIL import Image

from functools import wraps
from datetime import datetime

from itertools import permutations


def load_configger(config_path):
    if not os.path.isfile(config_path):
        raise FileNotFoundError('File %s does not exist.' % config_path)
    with open(config_path, 'r') as fp:
        config = commentjson.loads(fp.read())
    return config


def is_image_valid(pic_path):
    try:
        Image.open(pic_path).verify()
        return True
    except:
        return False


def set_gpu_options(config_settings):

    gpu_options = tf.GPUOptions(allow_growth=config_settings['allow_growth'])
    config = tf.ConfigProto(
        log_device_placement=config_settings['log_device_placement'],
        gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=config)
    return isess


def gen_random_mask(h, w, M):
    masks = []
    for _ in range(M):
        pixels = h * w
        mask = np.asarray([random.randint(0, 1) for _ in range(pixels)])
        mask = np.tile(np.reshape(mask, (h, w, 1)), [1, 1, 3])
        active_locs = [
            (y, x)
            for (y, x) in zip(np.where(mask == 1)[0],
                              np.where(mask == 1)[1])
        ]

        permut = [i for i in range(len(active_locs))]
        permut = permutations(permut)
        masks.append(mask)

    return masks
