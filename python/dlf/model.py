import tensorflow as tf
import tensorflow.keras.models as models

from python.dlf.bid_embedding import BidEmbeddingLayer
from python.dlf.bid_reshaper import BidReshape
from python.dlf.bid_rnn import BidRNNLayer
from python.dlf.bid_prefix import BidPrefix


class DLF(models.Model):

    _EMBEDDING_DIM = 32
    _STATE_SIZE = 128
    _MAX_SEQ_LEN = 360

    def __init__(self, features_number=16):
        super(DLF, self).__init__()
        self.features_number = features_number
        self.embedding_layer = BidEmbeddingLayer(features_number, self._EMBEDDING_DIM)
        self.bid_reshape = BidReshape(self._MAX_SEQ_LEN)
        self.rnn = BidRNNLayer(self._STATE_SIZE)
        self.bid_info_size = None
        self.bid_prefix = None

    def build(self, input_shape):
        self.bid_info_size = input_shape[1] - self.features_number
        self.bid_prefix = BidPrefix(self._MAX_SEQ_LEN, self.bid_info_size)
        self.built = True

    def call(self, inputs, **kwargs):
        bid_info, features = tf.split(inputs, [self.bid_info_size, self.features_number], axis=1)
        x = self.embedding_layer(features)
        x = self.bid_reshape(x)
        x = self.rnn(x)
        x = tf.concat([x, tf.cast(bid_info, dtype=tf.float32)], axis=1)
        x = self.bid_prefix(x)
        return x
