import tensorflow as tf


def main():
    # x = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # y0 = tf.constant([[4.0], [8.0], [12.0]])
    #
    # w = tf.Variable([[1.0], [1.0]])
    #
    # with tf.GradientTape() as tape:
    #     y = tf.matmul(x, w)
    #     print("y : ", y.numpy())
    #     loss = tf.reduce_sum(y - y0)
    #     print("loss : ", loss.numpy())
    #
    # grad = tape.gradient(loss, w)  # gradient calculation is correct
    # print("gradient : ", grad.numpy())
    #
    # mu = 0.01
    # w = w - mu * grad
    #
    # with tf.GradientTape() as tape:
    #     tape.watch(w)
    #     y = tf.matmul(x, w)
    #     print("y : ", y.numpy())
    #     loss = tf.reduce_sum(y - y0)
    #     print("loss : ", loss.numpy())
    #
    # grad = tape.gradient(loss, w)  # gradient value go to 'None'
    # print("gradient : ", grad)
    with tf.GradientTape() as tape:
        print(tape.gradient(tf.constant(5), tf.Variable(0)))


if __name__ == '__main__':
    main()