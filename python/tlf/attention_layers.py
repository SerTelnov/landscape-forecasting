#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tensorflow.keras.layers as layers


class MultiHeadAttention(layers.Layer):

    def __init__(self, models, h, *args, **kwargs):
        super(MultiHeadAttention, self).__init__(*args, **kwargs)
        assert models % h == 0

        self.models = models
        self.num_heads = h
        self.depth = models // self.num_heads
        self.softmax = layers.Softmax()

        self.wq = layers.Dense(models)
        self.wk = layers.Dense(models)
        self.wv = layers.Dense(models)

        self.dense = layers.Dense(models)

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]

        query = self.wq(inputs)
        key = self.wk(inputs)
        value = self.wv(inputs)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        #### Self-Attention

        qk = tf.matmul(query, key, transpose_b=True)

        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = qk / tf.math.sqrt(dk)

        w_attn = self.softmax(scaled_attention_logits)
        scaled_attention = tf.matmul(w_attn, value)

        #### Concatenation

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.models))
        output = self.dense(concat_attention)

        return output

    @tf.function
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])


class SelfAttention(layers.Layer):

    def __init__(self, models):
        super(SelfAttention).__init__()
        self.models = models
        self.w_q, self.w_k, self.w_v = None, None, None
        self.softmax = layers.Softmax()
        self.dropout = layers.Dropout(0.1)

    def build(self, input_shape):
        self.w_q = self.add_weight(name="W_query", shape=(input_shape[-1],))
        self.w_k = self.add_weight(name="W_key", shape=(input_shape[-1],))
        self.w_v = self.add_weight(name="K_value", shape=(input_shape[-1],))

    def call(self, inputs, **kwargs):
        query = tf.matmul(inputs, self.w_q)
        key = tf.matmul(inputs, self.w_k)
        value = tf.matmul(inputs, self.w_v)

        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(self.models)
        w_attn = self.softmax(scores)
        w_attn = self.dropout(w_attn)

        return tf.matmul(w_attn, value)
