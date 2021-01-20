"""SSD model builder
Utilities for building network layers are also provided
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.layers import ELU, MaxPooling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import numpy as np


def conv2d(inputs,
           filters=32,
           kernel_size=3,
           strides=1,
           name=None):

    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  kernel_initializer='he_normal',
                  name=name,
                  padding='same')

    return conv(inputs)


def conv_layer(inputs,
               filters=32,
               kernel_size=3,
               strides=1,
               use_maxpool=True,
               postfix=None,
               activation=None):

    x = conv2d(inputs,
               filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               name='conv'+postfix)
    x = BatchNormalization(name="bn"+postfix)(x)
    x = ELU(name='elu'+postfix)(x)
    if use_maxpool:
        x = MaxPooling2D(name='pool'+postfix)(x)
    return x

def build_ssd(input_shape,
              backbone,
              n_layers=4,
              n_classes=4,
              aspect_ratios=(1, 2, .5)):
    """Build SSD model given a backbone

    :param input_shape: input image shape
    :param backbone: keras backbone model
    :param n_layers: Number of layers of ssd head
    :param n_classes: number of classes
    :param aspect_ratios: anchor box aspect ratios

    :return n_anchors: number of anchor boxes per feature pt
    :return feature_shapes: SSD head feature maps
    :return model: keras model (SSD model)
    """
    # number of anchor boxes per feature pt
    n_anchors = len(aspect_ratios)+1

    inputs = Input(shape=input_shape)
    # no. of base outputs depends on n_layers
    base_outputs = backbone(inputs)

    outputs = []
    feature_shapes = []
    out_cls = []
    out_off = []

    for i in range(n_layers):
        # each conv layer from the backbone is used as a feature maps for class and offset predictions also known as
        # multi-scale predictions
        conv = base_outputs if n_layers == 1 else base_outputs[i]
        name = "cls" + str(i+1)
        classes = conv2d(conv,
                         n_anchors*n_classes,
                         kernel_size=3,
                         name=name)
        # offsets: (batch, height, width, n_anchors*4)
        name = "off" + str(i+1)
        offsets = conv2d(conv,
                         n_anchors*4,
                         kernel_size=4,
                         name=name)

        shape = np.array(K.int_shape(offsets))[1:]
        feature_shapes.append(shape)

        # reshape the class predictions, yielding 3D tensors of shape (batch, height * width * n_anchors, n_classes)
        # last axis to perform softmax on them
        name = "cls_res" + str(i + 1)
        classes = Reshape((-1, n_classes), name=name)(classes)

        # reshape the offset predictions, yielding 3D tensors of shape (batch, height * width * n_anchors, 4)
        # last axis to compute the (smooth) L1 or L2 loss
        name = "off_res" + str(i + 1)
        offsets = Reshape((-1, 4), name=name)(offsets)
        # concat for alignment with ground truth size made of ground truth offsets and mask of same dim
        # needed during loss computation
        offsets = [offsets, offsets]
        name = "off_cat" + str(i + 1)
        offsets = Concatenate(axis=-1, name=name)(offsets)

        # collect offset prediction per scale
        out_off.append(offsets)

        name = "cls_out" + str(i + 1)

        classes = Activation('softmax', name=name)(classes)

        # collect class prediction per scale
        out_cls.append(classes)

    if n_layers > 1:
        # concat all class and offset from each scale
        offsets = Concatenate(axis=1, name="offsets")(out_off)
        classes = Concatenate(axis=1, name="classes")(out_cls)
    else:
        offsets = out_off[0]
        classes = out_cls[0]

    outputs = [classes, offsets]
    model = Model(inputs=inputs,
                  outputs=outputs,
                  name='ssd_head')

    return n_anchors, feature_shapes, model
