#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tensorflow.keras.layers as layers

from python.util import LossMode


class BidPrefix(layers.Layer):

    def __init__(self, seq_len, loss_mode=LossMode.ALL_LOSS):
        super(BidPrefix, self).__init__()
        self.seq_len = seq_len

        if loss_mode == LossMode.ALL_LOSS:
            self.prod_fn = self._prod_prefix
            self.splitter = lambda x: tf.split(x, num_or_size_splits=2, axis=1)
        elif loss_mode == LossMode.CROSS_ENTROPY:
            self.prod_fn = self._cross_entropy_fn
            self.splitter = lambda x: tf.reshape(x, shape=(-1, 1))
        elif loss_mode == LossMode.ANLP:
            self.prod_fn = self._anlp_fn
            self.splitter = lambda x: tf.reshape(x, shape=(-1, 1))

    def call(self, inputs, **kwargs):
        bid_info, x = inputs
        x = tf.concat([x, tf.cast(bid_info, dtype=tf.float32)], axis=1)
        x = tf.map_fn(self.prod_fn, elems=x)
        x = self.splitter(x)
        return x

    @tf.function
    def _prod_prefix(self, x):
        market_price = tf.cast(x[self.seq_len], dtype=tf.int32)
        bid = tf.cast(x[self.seq_len + 1], dtype=tf.int32)

        survival_rate = tf.reduce_prod(x[0:bid])

        if market_price != 0:
            rate_last_one = tf.reduce_prod(x[0:market_price + 1])
            rate_last_two = tf.reduce_prod(x[0:market_price])
            rate_last = rate_last_two - rate_last_one
        else:
            rate_last = tf.keras.backend.epsilon()

        return tf.stack([survival_rate, rate_last])

    @tf.function
    def _cross_entropy_fn(self, x):
        bid = tf.cast(x[self.seq_len + 1], dtype=tf.int32)
        survival_rate = tf.reduce_prod(x[0:bid])
        return survival_rate

    @tf.function
    def _anlp_fn(self, x):
        market_price = tf.cast(x[self.seq_len], dtype=tf.int32)
        rate_last_one = tf.reduce_prod(x[0:market_price + 1])
        rate_last_two = tf.reduce_prod(x[0:market_price])
        return rate_last_two - rate_last_one
