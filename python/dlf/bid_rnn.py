#!/usr/bin/env python
# coding=utf-8
import tensorflow.keras.layers as layers


class BidRNNLayer(layers.Layer):

    def __init__(self, units):
        super(BidRNNLayer, self).__init__()
        self.rnn = layers.RNN(layers.LSTMCell(units), return_sequences=True)
        self.mapper = layers.Softmax()
        self.dense = layers.Dense(
            units=1,
            input_shape=(-1, units, 1)
        )
        self.reshape = None

    def build(self, input_shape):
        self.reshape = layers.Reshape((input_shape[1],))
        self.built = True

    def call(self, inputs, **kwargs):
        x = self.rnn(inputs)
        x = self.mapper(x)
        x = self.dense(x)
        x = self.reshape(x)
        return x
