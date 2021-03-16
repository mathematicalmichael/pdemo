# -*- coding: utf-8 -*-
"""This module facilitates preprocessing any images and also setting up
the objects needed for a training operation.
"""
###############################################################################
# Imports

import pathlib
import logging
import sys
import typing as T
import numpy as np
import pandas as pd
import PIL
from PIL import Image as I
import tensorflow as tf
import sklearn.preprocessing as pp

###############################################################################
# Get module context
C = sys.modules[__name__.split(".")[0]].context

###############################################################################
# Setup Logging
L = logging.getLogger(__name__)
###############################################################################
# Setup Reflexive link to module root
M = sys.modules[__name__.split(".")[0]]

###############################################################################
# Setup Locations
_ROOT = pathlib.Path(__file__).parent.absolute()

###############################################################################
# Dunders
__ALL__ = ["prep_an_image" "prep_images" "SetupTrainingParticulars"]


def prep_an_image(img: I.Image, size=(256, 256), preserve_aspect_ratio=True):
    # These should already be in RGB format as it is handled
    # earlier in the pipe
    img = tf.image.resize(
        images=np.array(img), size=size, preserve_aspect_ratio=preserve_aspect_ratio
    )
    img = tf.image.resize_with_pad(
        image=img, target_height=size[0], target_width=size[1]
    )
    # prep for batch concatenation
    img = tf.expand_dims(img, 0).numpy()
    return img


def prep_images(images: T.List[I.Image], size=(256, 256), preserve_aspect_ratio=True):
    with tf.device("/cpu:0"):
        imgs = [
            prep_an_image(img, size=size, preserve_aspect_ratio=preserve_aspect_ratio)
            for img in images
        ]
        imgs = np.concatenate(imgs, axis=0)
        return imgs
