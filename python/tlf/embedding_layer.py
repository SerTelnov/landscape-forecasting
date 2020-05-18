#!/usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers


class EmbeddingLayer(layers.Layer):
    _MAX_DEN = 580_000
    _EMB_DIM = 32

    def __init__(self, models, features_number, seq_len):
        super(EmbeddingLayer, self).__init__()
        self.models = models
        self.seq_len = seq_len
        self.embedding_layer = layers.Embedding(
            input_length=features_number,
            input_dim=self._MAX_DEN,
            output_dim=self._EMB_DIM
        )

        self.reshape = layers.Reshape(target_shape=(self._EMB_DIM * features_number,))
        self.middle_layer = layers.Dense(self.models)

        self.bid_reshape = layers.Reshape((seq_len, self.models))
        # self.bids = tf.reshape(tf.range(seq_len, dtype=tf.float32), shape=(seq_len, 1))
        self.pos_encoding = self.positional_encoding(seq_len, models)

    def call(self, input, **kwargs):
        x = self.embedding_layer(input)
        x *= tf.math.sqrt(tf.cast(self.models, tf.float32))

        x = self.reshape(x)
        x = self.middle_layer(x)

        x = tf.map_fn(lambda i: tf.tile(i, [self.seq_len]), x)
        x = self.bid_reshape(x)
        # x = tf.map_fn(lambda i: tf.concat([i, self.bids], axis=1), x)

        x += self.pos_encoding[:, :self.seq_len, :]
        return x

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
