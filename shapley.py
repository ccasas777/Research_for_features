import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import tensorflow.contrib.slim as slim
import argparse
import math
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

from pprint import pprint
from pathlib import Path
from utils.tools import set_gpu_options, is_image_valid, load_configger, gen_random_mask
from PIL import Image
from itertools import permutations

#TODO: put it in utils


def img_gen(img_root, config, isess):
    # Main image processing routine.
    def process_image(img,
                      select_threshold=0.5,
                      nms_threshold=.45,
                      net_shape=(300, 300)):
        # Run SSD network.
        rimg, rpredictions, rlocalisations, rbbox_img = isess.run(
            [image_4d, predictions, localisations, bbox_img],
            feed_dict={img_input: img})

        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions,
            rlocalisations,
            ssd_anchors,
            select_threshold=select_threshold,
            img_shape=net_shape,
            num_classes=21,
            decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses,
                                                            rscores,
                                                            rbboxes,
                                                            top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(
            rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes

    net_input_shape = (config['input_shape']['height'],
                       config['input_shape']['width'])
    save_path = os.path.join(img_root, 'results')
    # create file if it not exits
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    data_format = 'NHWC'
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    # Evaluation pre-processing: resize to SSD net shape.
    # links this method copied from[links:https://github.com/balancap/SSD-Tensorflow]
    # in order to follow model's backbone input, so here we resize the image
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input,
        None,
        None,
        net_input_shape,
        data_format,
        resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)

    image_4d = tf.expand_dims(image_pre, 0)
    # Define the SSD model.
    reuse = True if 'ssd_net' in locals() else None
    ssd_net = ssd_vgg_300.SSDNet()
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d,
                                                       is_training=False,
                                                       reuse=reuse)

    # Restore SSD model.
    ckpt_filename = config['model_path']
    isess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)
    # SSD default anchor boxes.
    ssd_anchors = ssd_net.anchors(net_input_shape)
    img_names = list(filter(lambda x: 'jpg' in x, os.listdir(img_root)))
    img_paths = list(map(lambda x: os.path.join(img_root, x), img_names))
    M = 1
    for img_path, img_name in zip(img_paths, img_names):
        origin_img = mpimg.imread(img_path)
        rclasses, rscores, rbboxes = process_image(origin_img)
        h, w, c = origin_img.shape
        mask_images, permuts = gen_random_mask(h, w, M)
        for masked_image in mask_images:
            m_rclasses, m_rscores, m_rbboxes = process_image(masked_image)
        #TODO: latter, it will add shapley based on activated pixel
        # I assumed when pixel value = 0, it will not contribure Shapley values.

        # img = visualization.gen_bboxes(img, rclasses, rscores, rbboxes)
        # print('writing %s' % os.path.join(save_path, img_name))
        # cv2.imwrite(os.path.join(save_path, img_name), img)
    #TODO: calculate shapley values.....


def parser():
    parser = argparse.ArgumentParser(
        description='Interpret model based on Shapley values')
    parser.add_argument('--img_root', default='./imgs')
    parser.add_argument('--cfg_path', default='./configs/shapley.json')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parser()
    config = load_configger(args.cfg_path)

    print(config)
    isess = set_gpu_options(config['gpu_options'])

    img_gen(args.img_root, config['predictor'], isess)
    print('Processing  SHAP...')
    #gen_shap_img(args.img_root, args.cfg_path)
