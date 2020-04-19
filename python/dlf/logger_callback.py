import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as callbacks

from python.dataset.stat_holder import StatHolder


class LoggerCallback(callbacks.Callback):

    def __init__(self, logger):
        super(LoggerCallback, self).__init__()

        self.stat_holder_train = StatHolder("TRAIN", logger)
        self.stat_holder_test = StatHolder("TEST", logger)

    def on_train_batch_end(self, batch, logs=None):
        self.stat_holder_train.hold(batch + 1, logs['loss'], logs['auc'])

    def on_test_batch_end(self, batch, logs=None):
        self.stat_holder_test.hold(batch + 1, logs['loss'], logs['auc'])
