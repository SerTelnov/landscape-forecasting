import tensorflow as tf
import tensorflow.keras.layers as layers


class FeedForward(layers.Layer):

    def __init__(self, models, dff, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(models)

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x
