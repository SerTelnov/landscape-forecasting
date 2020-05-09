import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.

    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.

    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self, bias=True, **kwargs):
        super(AttentionWithContext, self).__init__(**kwargs)
        self.bias = bias
        self.W = None
        self.b = None
        self.u = None

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(
            name='{}_W'.format(self.name),
            shape=(input_shape[-1], input_shape[-1],),
            trainable=True
        )
        if self.bias:
            self.b = self.add_weight(
                name='{}_b'.format(self.name),
                shape=(input_shape[-1],),
                initializer='zero',
                trainable=True
            )

        self.u = self.add_weight(
            name='{}_u'.format(self.name),
            shape=(input_shape[-1],),
            trainable=True
        )

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = self.dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = self.dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return weighted_input

    def compute_output_shape(self, input_shape):
        return input_shape

    @staticmethod
    def dot_product(x, kernel):
        """
        Wrapper for dot product operation
        Args:
            x (): input
            kernel (): weights
        """
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
