import tensorflow as tf
import numpy as np

from python.dataset.data import get_dataset
from python.dlf.model import DLF
from python.dlf.bin_loss import (
    cross_entropy, loss1, grad, grad_cross_entropy, common_loss, grad_
)

_BATCH_SIZE = 128

_TRAIN_STEP = 10
_LEARNING_RATE = 1e-3


def main():
    model = DLF()

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LEARNING_RATE, beta_2=0.99)
    loss_avg = tf.keras.metrics.Mean()

    inputs, y = get_dataset("../data/toy_dataset_train.tsv")
    # chunks_number = len(inputs) // _BATCH_SIZE + (1 if len(inputs) % _BATCH_SIZE != 0 else 0)
    #
    # inputs = np.array_split(inputs, chunks_number)
    # yy = np.array_split(y, chunks_number)
    #
    # model.build(input_shape=(_BATCH_SIZE, 18))
    #
    # for step in range(chunks_number):
    #     with tf.GradientTape(persistent=True) as tape:
    #         tape.watch(model.trainable_variables)
    #         prediction = model(inputs[step])
    #         target = yy[step]
    #
    #         loss1_value, grads1 = grad_(tape, model, prediction, target, loss1)
    #         cross_entropy_value, grads2 = grad_(tape, model, prediction, target, cross_entropy)
    #         loss_common, grads3 = grad_(tape, model, prediction, target, common_loss)
    #
    #         optimizer.apply_gradients(zip(grads1, model.trainable_variables))
    #         optimizer.apply_gradients(zip(grads2, model.trainable_variables))
    #         optimizer.apply_gradients(zip(grads3, model.trainable_variables))
    #
    #         print("Step: {}, Loss: {}".format(step, loss1_value.numpy()))
    #         print("Step: {}, Cross entropy: {}".format(step, cross_entropy_value.numpy()))
    #         print("Step: {}, Common loss: {}".format(step, loss_common.numpy()))

    model.compile(optimizer, loss=cross_entropy)
    model.fit(inputs, y, batch_size=128, epochs=1)
    x, y = get_dataset("../data/toy_dataset_test.tsv")
    # results = model.evaluate(x, y, batch_size=128)
    # print('test loss, test acc:', results)

    print(model.summary())


if __name__ == '__main__':
    main()
