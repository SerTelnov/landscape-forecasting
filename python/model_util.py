import tensorflow as tf
import os

from python.dlf.dlf import DLF
from python.tlf.transformer import TransformerForecasting
from python.util import ModelMode


def read_checkpoint(model, model_name):
    checkpoint_path = 'output/checkpoint/' + model_name

    print(checkpoint_path)

    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Restore model %s' % model_name)

    return model


def make_model(model_mode, checkpoint_model=None, training_mode=True):
    if model_mode in [ModelMode.DLF, ModelMode.DLF_ATTENTION]:
        model = DLF(model_mode, training_mode=training_mode)
    elif model_mode == ModelMode.TRANSFORMER:
        model = TransformerForecasting(training_mode)
    else:
        raise Exception("invalid model %ds" % model_mode)

    model.build(input_shape=([-1, 16], [-1, 2]))

    if checkpoint_model is not None:
        model = read_checkpoint(model, checkpoint_model)

    return model
