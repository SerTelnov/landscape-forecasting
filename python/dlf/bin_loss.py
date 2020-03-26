import tensorflow as tf

_SMALL_VALUE = 1e-20
_L2_NORM = 0.001
_GRAD_CLIP = 5.0
_ALPHA = 0.25
_BETA = 0.2


@tf.function
def cross_entropy(target, prediction):
    target = tf.cast(target, dtype=tf.float32)
    final_survival_rate, _, _ = tf.split(prediction, num_or_size_splits=3, axis=1)
    final_survival_rate = tf.clip_by_value(final_survival_rate, 1e-10, 1.0)
    final_dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), final_survival_rate)
    loss_cost = tf.subtract(tf.constant(1.0, dtype=tf.float32), target)

    return -tf.reduce_sum(target * tf.math.log(final_survival_rate) + loss_cost * final_dead_rate)


@tf.function
def loss1(target, prediction):
    _, rate_last_one, rate_last_two = tf.split(prediction, num_or_size_splits=3, axis=1)
    return -tf.reduce_sum(tf.math.log(tf.add(rate_last_two - rate_last_one, _SMALL_VALUE)))


def common_loss(target, prediction):
    return cross_entropy(target, prediction) * _ALPHA + \
           loss1(target, prediction) * _BETA


@tf.function
def grad(model, prediction, targets, loss_function):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        loss_value = loss_function(targets, prediction)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


def grad_(tape, model, prediction, targets, loss_function):
    loss_value = loss_function(targets, prediction)
    grads = tape.gradient(loss_value, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, _GRAD_CLIP)
    return loss_value, grads
    # return loss_value, grads

# @tf.function
# def grad_loss(model, loss_value):
#     with tf.GradientTape() as tape:
#         tape.watch(model.trainable_variables)
#         return tape.gradient(loss_value, model.trainable_weights)


@tf.function
def grad_cross_entropy(model, prediction, targets):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables]) * _L2_NORM
        loss_value = cross_entropy(targets, prediction)
        cost = tf.add(loss_value, lossL2)

        # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, model.trainable_variables), _GRAD_CLIP)
        grads = tape.gradient(cost, model.trainable_variables)
        return loss_value, grads


# @tf.function
# def grad_loss1(model, prediction, targets):
#     # print(model.trainable_variables)
#     with tf.GradientTape() as tape:
#         tape.watch(model.trainable_variables)
#         loss_value = loss1(targets, prediction)
#         grads = tape.gradient(loss_value, model.trainable_variables)
#         return loss_value, grads
