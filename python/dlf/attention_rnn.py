#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tensorflow.keras.layers as layers

from python.util import BATCH_SIZE


class AttentionRNNLayer(layers.Layer):

    def __init__(self, units):
        super(AttentionRNNLayer, self).__init__()
        self.units = units
        self.rnn = layers.RNN(
            layers.LSTMCell(units),
            return_sequences=True
        )
        self.attention = layers.Attention()
        self.attention_denses = None
        self.dense = layers.Dense(
            units=1,
            input_shape=(-1, units, 1)
        )
        self.reshape = None
        self.bid_sequence = None

    def build(self, input_shape):
        self.bid_sequence = input_shape[1]
        self.reshape = layers.Reshape((self.bid_sequence,))
        self.attention_denses = [layers.Dense(units=1, input_shape=(-1, i)) for i in range(self.bid_sequence)]
        self.built = True

    def call(self, inputs, **kwargs):
        x = self.rnn(inputs)
        x = tf.map_fn(self.attention_loop, x)
        x = self.dense(x)
        x = self.reshape(x)
        x = tf.keras.activations.sigmoid(x)
        return x

    def attention_loop(self, x):
        att_out = [x[0]]
        for time_step in range(1, self.bid_sequence):
            query, key = x[:time_step], x[time_step]
            out = self.attention([query, [key]])
            out = tf.transpose(out)
            out = self.attention_denses[time_step](out)
            out = tf.transpose(out)[0]
            att_out.append(out)
        return tf.stack(att_out)

    def _zero_init_state(self):
        zeros = tf.zeros((BATCH_SIZE, self.units))
        return [zeros, zeros]
