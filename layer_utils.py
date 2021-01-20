def anchor_boxes(feature_shape,
                 image_shape,
                 index=0,
                 n_layers=4,
                 aspect_ratios=(1, 2, .5)):
    """

    :param feature_shape: (list) Feature map shape
    :param image_shape: (list) Image size shape
    :param index: (int) Indicates which ssd head layers are we referring to
    :param n_layers: (int) Number of ssd head layers
    :param aspect_ratios: (list) The aspect ratios of the anchor boxes
    :return: boxes (tensor) Anchor boxes per feature map
    """

    boxes = []
    return boxes
