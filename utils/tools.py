import os
from datetime import datetime
from functools import wraps
import commentjson
from PIL import Image
import tensorflow as tf
from nets import ssd_vgg_300, ssd_common, np_methods


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
