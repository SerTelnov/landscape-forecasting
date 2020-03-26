import tensorflow as tf

from python.dataset.data import get_dataset
from python.dlf.model import DLF
from python.dlf.bin_loss import (
    cross_entropy, loss1, grad, grad_cross_entropy, common_loss, grad_
)

_TRAIN_STEP = 3
_LEARNING_RATE = 1e-3


def main():
    model = DLF()

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LEARNING_RATE, beta_2=0.99)
    loss_avg = tf.keras.metrics.Mean()

    input, y = get_dataset("../data/toy_dataset_train.tsv")
    input = input[:128]
    y = y[:128]

    for step in range(_TRAIN_STEP):
        with tf.GradientTape(persistent=True) as tape:
            prediction = model(input)

            loss1_value, grads1 = grad_(tape, model, prediction, y, loss1)
            cross_entropy_value, grads2 = grad_(tape, model, prediction, y, cross_entropy)
            loss_common, grads3 = grad_(tape, model, prediction, y, common_loss)

            optimizer.apply_gradients(zip(grads1, model.trainable_variables))
            optimizer.apply_gradients(zip(grads2, model.trainable_variables))
            optimizer.apply_gradients(zip(grads3, model.trainable_variables))

            print("Step: {}, Loss: {}".format(step, loss1_value.numpy()))
            print("Step: {}, Cross entropy: {}".format(step, cross_entropy_value.numpy()))
            print("Step: {}, Common loss: {}".format(step, loss_common.numpy()))

    # model.fit(input, y, batch_size=128)
    x, y = get_dataset("../data/toy_dataset_test.tsv")
    # results = model.evaluate(x, y, batch_size=128)
    # print('test loss, test acc:', results)

    print(model.summary())


if __name__ == '__main__':
    main()
