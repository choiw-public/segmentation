from functions.utils import get_shape
from tensorflow.keras import layers, regularizers
from tensorflow.keras.activations import elu, relu, swish, sigmoid, softmax
import tensorflow as tf


class Conv(layers.Layer):
    def __init__(self, out_depth, kernel_size, stride, l2_decay, do_bnorm, do_actv, gc_factor, scope):
        super(Conv, self).__init__()
        self.bnorm = do_bnorm
        self.actv = do_actv
        self.gc_factor = gc_factor

        self.conv = layers.Conv2D(filters=out_depth,
                                  kernel_size=kernel_size,
                                  strides=stride,
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(l2_decay),
                                  name=scope + '/conv')
        if self.bnorm:
            self.bnorm = layers.BatchNormalization(name=scope + '/bnorm')

        if self.gc_factor:
            self.gc_layer = GCBlock(out_depth, gc_factor, scope + '/gc')

    def call(self, tensor_in, training=False, **kwargs):
        x = self.conv(tensor_in, training=training)
        if self.bnorm:
            x = self.bnorm(x, training=training)
        if self.actv:
            x = swish(x)
        if self.gc_factor:
            x = self.gc_layer(x)
        return x


class GCBlock(layers.Layer):
    def __init__(self,
                 in_channel_num,
                 gc_factor,
                 scope):
        super(GCBlock, self).__init__()

        self.context_conv = layers.Conv2D(filters=1,
                                          kernel_size=1,
                                          strides=1,
                                          padding='same',
                                          name=scope + '/context/conv')
        self.transform_shrink_conv = layers.Conv2D(filters=in_channel_num // gc_factor,
                                                   kernel_size=1,
                                                   strides=1,
                                                   padding='same',
                                                   name=scope + '/transform/shrink/conv')
        self.transform_lnorm = layers.LayerNormalization(name=scope + '/transform/shrink/bnorm')
        self.transform_expand_conv = layers.Conv2D(filters=in_channel_num,
                                                   kernel_size=1,
                                                   strides=1,
                                                   padding='same',
                                                   name=scope + '/transform/expand/conv')

    def call(self, tensor_in, **kwargs):
        n, h, w, c = get_shape(tensor_in)
        flatten = tf.reshape(tensor_in, [n, h * w, c])
        context = self.context_conv(tensor_in)
        context = tf.reshape(context, [n, h * w, 1])
        context = softmax(context, axis=1)
        context = tf.matmul(flatten, context, transpose_a=True)
        context = tf.reshape(context, [n, 1, 1, c])

        transform = self.transform_shrink_conv(context)
        transform = self.transform_lnorm(transform)
        transform = relu(transform)

        transform = self.transform_expand_conv(transform)
        transform = sigmoid(transform)
        return tensor_in + transform


class DenseConcat(layers.Layer):
    def __init__(self, out_depth_list, conv_size_list, l2_decay, gc_factor, scope):
        super(DenseConcat, self).__init__()
        if not len(out_depth_list) == len(conv_size_list):
            raise ValueError('check the length of each input')

        self.gc_factor = gc_factor
        self.dense_layers = []
        for i, (out_depth, conv_size) in enumerate(zip(out_depth_list, conv_size_list)):
            self.dense_layers.append(Conv(out_depth, conv_size, 1, l2_decay, True, False, 0, scope=scope + '/concat%d-pw' % (i + 1)))
            self.dense_layers.append(Conv(out_depth, conv_size, 1, l2_decay, True, True, 0, scope=scope + '/concat%d-conv' % (i + 1)))
        if self.gc_factor:
            self.gc_layer = GCBlock(out_depth_list[-1], gc_factor, scope + '/gc')

    def call(self, tensor_in, training=False):
        x = tensor_in
        branches = []
        for i, l in enumerate(self.dense_layers):
            if i % 2 == 0:
                branches.append(x)
                x = tf.concat(branches, 3)
            x = l(x)
            x = swish(x)
        if self.gc_factor:
            x = self.gc_layer(x)
        return x
