# Copyright 2017 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import cv2
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm
import numpy as np


# =========================================================================== #
# Some colormaps.
# =========================================================================== #
def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i * dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors


colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232),
                  (255, 127, 14), (255, 187, 120), (44, 160, 44),
                  (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75),
                  (196, 156, 148), (227, 119, 194), (247, 182, 210),
                  (127, 127, 127), (199, 199, 199), (188, 189, 34),
                  (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# =========================================================================== #
# OpenCV drawing.
# =========================================================================== #

# =========================================================================== #
# OpenCV drawing.
# =========================================================================== #


def gen_bboxes(img, classes, scores, bboxes, clr=(0, 255, 0)):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random() * 160, random.random() * 125,
                                  random.random() * 150)
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            tl = (xmin, ymin)
            br = (xmax, ymax)
            img = cv2.rectangle(img, tl, br, colors[cls_id], 1)
            class_name = str(cls_id)
            text_info = '{:s} | {:.3f}'.format(class_name, score)
            img = cv2.putText(img, text_info, (
                xmin,
                ymin - 2,
            ), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, colors[cls_id], 2,
                              cv2.LINE_AA)
    return img[..., ::-1]
