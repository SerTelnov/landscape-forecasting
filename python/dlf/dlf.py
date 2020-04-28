#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tensorflow.keras.models as models

from python.dlf.bid_embedding import BidEmbeddingLayer
from python.dlf.bid_prefix import BidPrefix
from python.dlf.bid_reshaper import BidReshape
from python.dlf.bid_rnn import BidRNNLayer
from python.dlf.attention_rnn import BidAttentionRNNLayer
from python.util import LossMode


class DLF(models.Model):

    _EMBEDDING_DIM = 32
    _STATE_SIZE = 128
    _MAX_SEQ_LEN = 310

    def __init__(self, loss_mode=LossMode.ALL_LOSS, *args, **kwargs):
        super(DLF, self).__init__(*args, **kwargs)
        self.loss_mode = loss_mode
        self.embedding_layer = None
        self.bid_reshape = BidReshape(self._MAX_SEQ_LEN)
        # self.rnn = BidRNNLayer(self._STATE_SIZE)
        self.attention_rnn = BidAttentionRNNLayer(self._STATE_SIZE)
        self.bid_prefix = None

    def build(self, input_shape):
        features_number = input_shape[0][1]
        bid_info_number = input_shape[1][1]

        self.embedding_layer = BidEmbeddingLayer(features_number, self._EMBEDDING_DIM)
        self.bid_prefix = BidPrefix(self._MAX_SEQ_LEN, bid_info_number, self.loss_mode)
        self.built = True

    def call(self, inputs, **kwargs):
        features, bid_info = inputs
        x = self.embedding_layer(features)
        x = self.bid_reshape(x)
        # x = self.rnn(x)
        x = self.attention_rnn(x)
        x = tf.concat([x, tf.cast(bid_info, dtype=tf.float32)], axis=1)
        x = self.bid_prefix(x)
        return x
