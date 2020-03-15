import tensorflow.keras.models as models

from python.dlf.bid_embedding import BidEmbeddingLayer
from python.dlf.bid_reshaper import BidReshape
from python.dlf.bid_rnn import BidRNNLayer


class DLF(models.Model):

    _EMBEDDING_DIM = 32
    _STATE_SIZE = 128
    _MAX_SEQ_LEN = 360

    def __init__(self, features_number=16):
        super(DLF, self).__init__()
        self.embedding_layer = BidEmbeddingLayer(features_number, self._EMBEDDING_DIM)
        self.bid_reshape = BidReshape(self._MAX_SEQ_LEN)
        self.rnn = BidRNNLayer(self._STATE_SIZE)

    def call(self, inputs, **kwargs):
        x = self.embedding_layer(inputs)
        x = self.bid_reshape(x)
        x = self.rnn(x)
        return x
