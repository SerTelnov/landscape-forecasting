#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tensorflow.keras.losses as losses

from python.dlf.dlf import DLF

_SMALL_VALUE = 1e-20
_L2_NORM = 0.001
_GRAD_CLIP = 5.0
ALPHA = 0.25
BETA = 0.2


def cross_entropy(y_true, y_pred):
    final_survival_rate = y_pred
    w = tf.cast(y_true, dtype=tf.float32)

    final_dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), final_survival_rate)
    w_ = tf.subtract(tf.constant(1.0, dtype=tf.float32), w)

    final_survival_rate = _clip_values(final_survival_rate)
    final_dead_rate = _clip_values(final_dead_rate)

    return -tf.reduce_mean(w * tf.math.log(final_survival_rate) + w_ * tf.math.log(final_dead_rate))


def _clip_values(x):
    return tf.clip_by_value(x, 1e-10, 1.0)


def loss1(y_true, y_pred):
    return -tf.reduce_mean(tf.math.log(tf.add(y_pred, _SMALL_VALUE)))

# @tf.function
# def loss1(target, prediction):
#     rate_last_one = prediction[1]
#     rate_last_two = prediction[2]
#
#     return -tf.reduce_mean(
#         tf.math.log(
#             tf.add(
#                 rate_last_two - rate_last_one,
#                 _SMALL_VALUE
#             )
#         )
#     )
    # return -tf.reduce_mean(
    #     tf.subtract(
    #         tf.math.log(rate_last_two),
    #         tf.math.log(rate_last_one)
    #     )
    # )


def common_loss(l1, l2):
    return l1 * BETA + l2 * ALPHA


def loss_grad(tape, tvar, target, pred, loss_function):
    loss_value = loss_function(target, pred)
    return _grad_(tape, loss_value, tvar)


def grad_common_loss(tape: tf.GradientTape, tvar, loss1_value, loss2_value) -> object:
    loss_value = common_loss(loss1_value, loss2_value)
    return _grad_(tape, loss_value, tvar)


def _grad_(tape, loss_value, train_vars):
    grads = tape.gradient(loss_value, train_vars)
    grads, _ = tf.clip_by_global_norm(grads, _GRAD_CLIP)
    return loss_value, grads
