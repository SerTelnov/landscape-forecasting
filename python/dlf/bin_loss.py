import tensorflow as tf

_INFINITE = 1e-20


def cross_entropy(win_flags, output):
    win_flags = tf.cast(win_flags, dtype=tf.float32)
    final_survival_rate, _, _ = tf.split(output, num_or_size_splits=3, axis=1)
    output = tf.clip_by_value(output, 1e-10, 1)
    final_dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), output)
    loss_cost = tf.subtract(tf.constant(1.0, dtype=tf.float32), win_flags)

    return -(win_flags * tf.math.log(output) + loss_cost * final_dead_rate)


def loss1(target, output):
    _, rate_last_one, rate_last_two = tf.split(output, num_or_size_splits=3, axis=1)
    return -tf.math.log(tf.add(rate_last_two - rate_last_one, _INFINITE))
