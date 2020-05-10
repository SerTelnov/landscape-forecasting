import tensorflow as tf
import os

from python.dlf.dlf import DLF


def read_checkpoint(model_name):
    checkpoint_path = '../output/checkpoint/aws/' + model_name + '/cp-{epoch:02d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    return tf.train.latest_checkpoint(checkpoint_dir)


def make_model(model_mode, checkpoint_model=None):
    dlf = DLF(model_mode)
    dlf.build(input_shape=([-1, 16], [-1, 2]))

    if checkpoint_model is not None:
        latest = read_checkpoint(checkpoint_model)
        dlf.load_weights(latest)

    return dlf
