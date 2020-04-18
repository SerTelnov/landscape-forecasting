#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tensorflow.keras.layers as layers


class BidPrefix(layers.Layer):

    def __init__(self, seq_len, bid_info_size):
        super(BidPrefix, self).__init__()
        self.seq_len = seq_len
        self.bid_info_size = bid_info_size

    def call(self, inputs, **kwargs):
        x = tf.map_fn(self._prod_prefix, elems=inputs)
        x = tf.split(x, num_or_size_splits=2, axis=1)
        return x

    @tf.function
    def _prod_prefix(self, x):
        market_price = tf.cast(x[self.seq_len], dtype=tf.int32)
        bid = tf.cast(x[self.seq_len + 1], dtype=tf.int32)

        survival_rate = tf.reduce_prod(x[0:bid])

        rate_last_one = tf.reduce_prod(x[0:market_price + 1])
        rate_last_two = tf.reduce_prod(x[0:market_price])
        rate_last = rate_last_two - rate_last_one

        return tf.stack([survival_rate, rate_last])
