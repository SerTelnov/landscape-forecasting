import tensorflow as tf
import tensorflow.keras.layers as layers


class BidEmbeddingLayer(layers.Layer):
    _MAX_DEN = 580_000
    _MIDDLE_FEATURE_SIZE = 30

    def __init__(self, features_number, embedding_dim):
        super(BidEmbeddingLayer, self).__init__()
        self.embedding_layer = layers.Embedding(
            input_length=features_number,
            input_dim=self._MAX_DEN,
            output_dim=embedding_dim
        )
        self.reshape = layers.Reshape(target_shape=(embedding_dim * features_number,))
        self.middle_layer = layers.Dense(self._MIDDLE_FEATURE_SIZE, activation=tf.nn.relu)

    def call(self, input, **kwargs):
        x = self.embedding_layer(input)
        x = self.reshape(x)
        x = self.middle_layer(x)
        return x
