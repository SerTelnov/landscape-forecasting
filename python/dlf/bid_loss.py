import tensorflow as tf

from python.dlf.model import DLF

_SMALL_VALUE = 1e-20
_L2_NORM = 0.001
_GRAD_CLIP = 5.0
ALPHA = 0.25
BETA = 0.2


@tf.function
def cross_entropy(target, prediction):
    target = tf.cast(target, dtype=tf.float32)

    final_survival_rate = prediction[0]
    final_dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), final_survival_rate)

    predict = tf.transpose(tf.stack([final_survival_rate, final_dead_rate]))
    return -tf.reduce_mean(target * tf.math.log(tf.clip_by_value(predict, 1e-10, 1.0)))


# @tf.function
def loss1(target, prediction):
    rate_last_one = prediction[1]
    rate_last_two = prediction[2]

    return -tf.reduce_mean(
        tf.subtract(
            tf.math.log(
                tf.clip_by_value(rate_last_one, _SMALL_VALUE, 1)
            ),
            tf.math.log(
                tf.clip_by_value(rate_last_two, _SMALL_VALUE, 1)
            )
        )
    )


@tf.function
def common_loss(l1, l2):
    return l1 * BETA + l2 * ALPHA


def grad_(tape: tf.GradientTape, model: DLF, prediction: object, targets: object, loss_function) -> object:
    loss_value = loss_function(targets, prediction)
    return _grad_(tape, loss_value, model.trainable_variables)


def grad_common_loss(tape: tf.GradientTape, model: DLF, cross_entropy_value, loss1_value) -> object:
    loss_value = common_loss(loss1_value, cross_entropy_value)
    return _grad_(tape, loss_value, model.trainable_variables)


def _grad_(tape, loss_value, train_vars):
    grads = tape.gradient(loss_value, train_vars)
    grads, _ = tf.clip_by_global_norm(grads, _GRAD_CLIP)
    return loss_value, grads

# @tf.function
# def grad_cross_entropy(model, prediction, targets):
#     with tf.GradientTape() as tape:
#         tape.watch(model.trainable_variables)
#         lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables]) * _L2_NORM
#         loss_value = cross_entropy(targets, prediction)
#         cost = tf.add(loss_value, lossL2)
#
#         # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, model.trainable_variables), _GRAD_CLIP)
#         grads = tape.gradient(cost, model.trainable_variables)
#         return loss_value, grads
