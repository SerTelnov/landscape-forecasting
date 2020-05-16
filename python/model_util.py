import tensorflow as tf
import os

from python.dlf.dlf import DLF
from python.tlf.tlf import TransformerForecasting
from python.util import ModelMode


def read_checkpoint(model_name):
    checkpoint_path = '../output/checkpoint/aws/' + model_name + '/cp-{epoch:02d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    return tf.train.latest_checkpoint(checkpoint_dir)


def make_model(model_mode, checkpoint_model=None, training_mode=True):
    if model_mode in [ModelMode.DLF, ModelMode.DLF_ATTENTION]:
        model = DLF(model_mode, training_mode=training_mode)
    elif model_mode == ModelMode.TRANSFORMER:
        model = TransformerForecasting(training_mode)
    else:
        raise Exception("invalid model %ds" % model_mode)

    model.build(input_shape=([-1, 16], [-1, 2]))

    if checkpoint_model is not None:
        latest = read_checkpoint(checkpoint_model)
        model.load_weights(latest)

    return model
