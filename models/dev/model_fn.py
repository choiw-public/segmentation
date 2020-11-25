from tensorflow.keras import Input, Model
from functions import conv_blocks
from functions.utils import get_shape
import tensorflow as tf


class ModelFunction(Model):
    def __init__(self, config):
        super(ModelFunction, self).__init__()
        l2_decay = config.weight_decay
        training = config.training
        self.root = conv_blocks.Conv(8, 5, 1, l2_decay, True, False, 0, 'root')

        self.encoder1a = conv_blocks.DenseConcat([8, 8], [5, 3], l2_decay, 0, 'encoder1a')  # 4
        self.encoder1a_skip = conv_blocks.Conv(16, 1, 1, l2_decay, True, True, 0, 'encoder1a_skip')
        self.encoder1b = conv_blocks.Conv(16, 3, 2, l2_decay, True, True, 0, 'encoder1b')

        self.encoder2a = conv_blocks.DenseConcat([8, 8, 16], [7, 5, 3], l2_decay, 4, 'encoder2a')  # 4
        self.encoder2a_skip = conv_blocks.Conv(32, 1, 1, l2_decay, True, True, 0, 'encoder2a_skip')
        self.encoder2b = conv_blocks.Conv(32, 3, 2, l2_decay, True, True, 0, 'encoder2b')

        self.encoder3a = conv_blocks.DenseConcat([16, 16, 16, 32], [9, 7, 5, 3], l2_decay, 4, 'encoder3a')  # 4
        self.encoder3a_skip = conv_blocks.Conv(64, 1, 1, l2_decay, True, True, 0, 'encoder3a_skip')
        self.encoder3b = conv_blocks.Conv(64, 3, 2, l2_decay, True, True, 0, 'encoder3b')

        self.encoder4a = conv_blocks.DenseConcat([16, 24, 32, 40, 48], [11, 9, 7, 5, 3], l2_decay, 4, 'encoder4a')  # 4
        self.encoder4b = conv_blocks.Conv(192, 3, 1, l2_decay, True, True, 0, 'encoder4b')

        self.decoder1 = conv_blocks.Conv(64, 3, 1, l2_decay, True, True, 0, 'decoder1')  # 4
        self.decoder2 = conv_blocks.Conv(32, 3, 1, l2_decay, True, True, 0, 'decoder2')  # 4
        self.decoder3 = conv_blocks.Conv(16, 3, 1, l2_decay, True, True, 0, 'decoder3')  # 4

        self.logit = conv_blocks.Conv(2, 3, 1, l2_decay, False, False, 0, 'logit')

    def call(self, tensor_in, training=False, **kwargs):
        root = self.root(tensor_in, training=training)

        encoder1a = self.encoder1a(root, training=training)
        encoder1a_skip = self.encoder1a_skip(encoder1a, training=training)
        encoder1b = self.encoder1b(encoder1a, training=training)

        encoder2a = self.encoder2a(encoder1b, training=training)
        encoder2a_skip = self.encoder2a_skip(encoder2a, training=training)
        encoder2b = self.encoder2b(encoder2a, training=training)

        encoder3a = self.encoder3a(encoder2b, training=training)
        encoder3a_skip = self.encoder3a_skip(encoder3a, training=training)
        encoder3b = self.encoder3b(encoder3a, training=training)

        encoder4a = self.encoder4a(encoder3b, training=training)
        encoder4b = self.encoder4b(encoder4a, training=training)

        decoder1 = self.decoder1(encoder4b, training=training)
        _, h, w, _ = get_shape(encoder3a_skip)
        decoder1 = tf.image.resize(decoder1, [h, w]) + encoder3a_skip

        decoder2 = self.decoder2(decoder1, training=training)
        _, h, w, _ = get_shape(encoder2a_skip)
        decoder2 = tf.image.resize(decoder2, [h, w]) + encoder2a_skip

        decoder3 = self.decoder3(decoder2, training=training)
        _, h, w, _ = get_shape(encoder1a_skip)
        decoder3 = tf.image.resize(decoder3, [h, w]) + encoder1a_skip

        return self.logit(decoder3)
