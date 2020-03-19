import tensorflow as tf


def cross_entropy(win_flags, output):
    win_flags = tf.cast(win_flags, dtype=tf.float32)
    output = tf.clip_by_value(output, 1e-10, 1)
    final_dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), output)
    loss_cost = tf.subtract(tf.constant(1.0, dtype=tf.float32), win_flags)

    return -tf.reduce_sum(win_flags * tf.math.log(output) + loss_cost * final_dead_rate)
