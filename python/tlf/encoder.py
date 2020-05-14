import tensorflow as tf
import tensorflow.keras.layers as layers

from python.tlf.encoder_layer import EncoderLayer


class Encoder(layers.Layer):

    def __init__(self, num_layers, models, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()

        self.models = models
        self.num_layers = num_layers

        self.enc_layers = [EncoderLayer(models, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, **kwargs):
        return self.enc_loop(inputs)

    # @tf.function
    def enc_loop(self, x):
        for encoder in self.enc_layers:
            x = encoder(x)
        return x
