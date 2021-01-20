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
    return n_anchors, feature_shapes, model