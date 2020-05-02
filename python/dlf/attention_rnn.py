#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tensorflow.keras.layers as layers


class BidAttentionRNNLayer(layers.Layer):

    def __init__(self, units):
        super(BidAttentionRNNLayer, self).__init__()
        self.rnn = layers.RNN(
            layers.LSTMCell(units),
            return_sequences=True,
            return_state=True
        )
        self.attention = layers.Attention()
        self.dense = layers.Dense(
            units=1,
            input_shape=(-1, units, 1)
        )
        self.reshape = None

    def build(self, input_shape):
        self.reshape = layers.Reshape((input_shape[1],))
        self.built = True

    def call(self, inputs, **kwargs):
        x, state_h, state_c = self.rnn(inputs)
        x = self.attention([x, state_h])
        x = self.dense(x)
        x = self.reshape(x)
        x = tf.keras.activations.sigmoid(x)
        return x
