#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tensorflow.keras.layers as layers


class EmbeddingLayer(layers.Layer):
    _MAX_DEN = 580_000

    def __init__(self, models, features_number):
        super(EmbeddingLayer, self).__init__()
        self.models = models
        self.embedding_layer = layers.Embedding(
            input_length=features_number,
            input_dim=self._MAX_DEN,
            output_dim=self.models
        )
        self.middle_layer = layers.Dense(2)

    def call(self, input, **kwargs):
        x = self.embedding_layer(input)
        x *= tf.math.sqrt(tf.cast(self.models, tf.float32))
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.middle_layer(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        return x
