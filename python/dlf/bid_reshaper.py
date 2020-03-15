import tensorflow as tf
import tensorflow.keras.layers as layers


class BidReshape(layers.Layer):

    def __init__(self, bid_sequence):
        super(BidReshape, self).__init__()
        self.bid_sequence = bid_sequence
        self.reshape = None
        self.bids = tf.reshape(tf.cast(tf.range(bid_sequence), dtype=tf.float32), shape=(bid_sequence, 1))

    def build(self, input_shape):
        self.reshape = layers.Reshape((self.bid_sequence, input_shape[1]))
        self.built = True

    def call(self, inputs, **kwargs):
        x = tf.map_fn(lambda i: tf.tile(i, [self.bid_sequence]), inputs)
        x = self.reshape(x)
        x = tf.map_fn(lambda i: tf.concat([i, self.bids], axis=1), x)
        return x
