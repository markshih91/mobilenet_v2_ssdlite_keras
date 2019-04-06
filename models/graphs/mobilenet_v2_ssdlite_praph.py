import keras.layers as KL
import keras.backend as K


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def _inverted_res_block(inputs, expansion, stride, alpha, filters, stage=1, block_id=1, expand=True, output2=False):
    in_channels = K.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    name = 'bbn_stage{}_block{}'.format(stage, block_id)

    if expand:
        # Expand
        x = KL.Conv2D(expansion * in_channels, kernel_size=1,
                      padding='same', use_bias=False, activation=None, name=name + '_expand_conv')(x)
        x = KL.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999, name=name + '_expand_bn')(x)
        x = KL.ReLU(6., name=name + '_expand_relu')(x)

    out2 = x

    # Depthwise
    if stride == 2:
        x = KL.ZeroPadding2D(padding=correct_pad(K, x, 3),
                             name=name + '_dw_pad')(x)
    x = KL.DepthwiseConv2D(kernel_size=3, strides=stride,
                           activation=None, use_bias=False,
                           padding='same' if stride == 1 else 'valid',
                           name=name + '_dw_conv')(x)
    x = KL.BatchNormalization(epsilon=1e-3,
                              momentum=0.999, name=name + '_dw_bn')(x)

    x = KL.ReLU(6., name=name + '_dw_relu')(x)

    # Project
    x = KL.Conv2D(pointwise_filters, kernel_size=1,
                  padding='same', use_bias=False, activation=None, name=name + '_project_conv')(x)
    x = KL.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name=name + '_project_bn')(x)

    if in_channels == pointwise_filters and stride == 1:
        return KL.Add(name=name + '_add')([inputs, x])
    if output2:
        return x, out2
    return x


def _followed_down_sample_block(inputs, conv_out_channel, sep_out_channel, id):
    name = 'ssd_{}'.format(id)
    x = KL.Conv2D(conv_out_channel, kernel_size=1, padding='same', use_bias=False,
                  activation=None, name=name + '_conv')(inputs)
    x = KL.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_conv_bn')(x)
    x = KL.ReLU(6., name=name + '_conv_relu')(x)

    x = KL.ZeroPadding2D(padding=correct_pad(K, x, 3), name=name + '_dw_pad')(x)
    x = KL.DepthwiseConv2D(kernel_size=3, strides=2,
                           activation=None, use_bias=False, padding='valid', name=name + '_dw_conv')(x)
    x = KL.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_dw_bn')(x)
    x = KL.ReLU(6., name=name + '_dw_relu')(x)

    x = KL.Conv2D(sep_out_channel, kernel_size=1, padding='same', use_bias=False,
                  activation=None, name=name + '_conv2')(x)
    x = KL.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_conv2_bn')(x)
    x = KL.ReLU(6., name=name + '_conv2_relu')(x)
    return x


def mobilenet_v2_ssdlite(input_image):

    alpha = 1.0

    first_block_filters = _make_divisible(32 * alpha, 8)

    # stage1
    x = KL.ZeroPadding2D(padding=correct_pad(K, input_image, 3),
                         name='bbn_stage1_block1_pad')(input_image)
    x = KL.Conv2D(first_block_filters,
                  kernel_size=3, strides=(2, 2),
                  padding='valid', use_bias=False,
                  name='bbn_stage1_block1_conv')(x)
    x = KL.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='bbn_stage1_block1_bn')(x)
    x = KL.ReLU(6., name='bbn_stage1_block1_relu')(x)
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                                 expansion=1, stage=1, block_id=2, expand=False)

    # stage2
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, stage=2, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                 expansion=6, stage=2, block_id=2)

    # stage3
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, stage=3, block_id=1)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, stage=3, block_id=2)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                 expansion=6, stage=3, block_id=3)

    # stage4
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, stage=4, block_id=1)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, stage=4, block_id=2)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, stage=4, block_id=3)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, stage=4, block_id=4)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, stage=4, block_id=5)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, stage=4, block_id=6)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                                 expansion=6, stage=4, block_id=7)

    # stage5
    x, link1 = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                            expansion=6, stage=5, block_id=1, output2=True)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, stage=5, block_id=2)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, stage=5, block_id=3)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                                 expansion=6, stage=5, block_id=4)

    x = KL.Conv2D(1280, kernel_size=1, padding='same', use_bias=False, activation=None, name='ssd_2_conv')(x)
    x = KL.BatchNormalization(epsilon=1e-3, momentum=0.999, name='ssd_2_conv_bn')(x)
    link2 = x = KL.ReLU(6., name='ssd_2_conv_relu')(x)

    link3 = x = _followed_down_sample_block(x, 256, 512, 3)

    link4 = x = _followed_down_sample_block(x, 128, 256, 4)

    x = _followed_down_sample_block(x, 128, 256, 5)

    link5 = x = _followed_down_sample_block(x, 64, 128, 6)

    links = [link1, link2, link3, link4, link5]

    return links

