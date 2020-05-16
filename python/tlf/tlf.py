#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers

from python.common.bid_prefix import BidPrefix
from python.tlf.embedding_layer import EmbeddingLayer
from python.tlf.encoder import Encoder
from python.tlf.final_layer import FinalLayer


class TransformerForecasting(models.Model):
    _MAX_SEQ_LEN = 310
    _NUM_LAYERS = 2
    _MODELS = 128
    _NUM_HEADS = 8
    _DFF = 2048

    def __init__(self, training_mode=True, *args, **kwargs):
        super(TransformerForecasting, self).__init__(*args, **kwargs)
        self.embedding_layer = None

        self.encoder = Encoder(
            num_layers=self._NUM_LAYERS,
            models=self._MODELS,
            num_heads=self._NUM_HEADS,
            dff=self._DFF
        )
        self.final = FinalLayer(self._MODELS)

        if training_mode:
            self.bid_prefix = BidPrefix(self._MAX_SEQ_LEN)
        else:
            self.bid_prefix = lambda x: x[1]

    def build(self, input_shape):
        features_number = input_shape[0][1]
        bid_info_number = input_shape[1][1]

        self.embedding_layer = EmbeddingLayer(self._MODELS, features_number, self._MAX_SEQ_LEN)
        self.built = True

    def call(self, inputs, **kwargs):
        features, bid_info = inputs

        x = self.embedding_layer(features)
        x = self.encoder(x)
        x = self.final(x)
        x = self.bid_prefix([bid_info, x])

        return x
