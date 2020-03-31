import tensorflow as tf

from python.dataset.data_reader import BiSparseData
from python.dlf.bin_loss import (
    cross_entropy, loss1, common_loss, grad_
)
from python.dlf.model import DLF
from python.dataset.stat_holder import StatHolder
from python.dataset.logger import Logger


_TRAIN_STEP = 21000
_BATCH_SIZE = 128

_LEARNING_RATE = 1e-3
_BETA_2 = 0.99


def run_test(model, step, logger):
    dataset = BiSparseData("../data/3476/test_all.tsv", _BATCH_SIZE)
    stat_holder = StatHolder('TEST', logger, is_train=False)

    while dataset.has_next():
        features, targets, is_win = dataset.next()
        prediction = model(features)
        cross_entropy_value = cross_entropy(targets, prediction)
        loss1_value = loss1(targets, prediction) if is_win else None
        stat_holder.hold(step, cross_entropy_value, targets, prediction, loss1_value)
    stat_holder.flush(step)


def main():
    model = DLF()
    logger = Logger(3476)
    stat_holder = StatHolder('TRAIN', logger)

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LEARNING_RATE, beta_2=_BETA_2)

    dataset = BiSparseData("../data/3476/train_all.tsv", _BATCH_SIZE)
    model.build(input_shape=(_BATCH_SIZE, 18))

    for step in range(_TRAIN_STEP):
        current_features, current_target, is_win = dataset.next()
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(model.trainable_variables)
            prediction = model(current_features)

            cross_entropy_value, grads1 = grad_(tape, model, prediction, current_target, cross_entropy)
            optimizer.apply_gradients(zip(grads1, model.trainable_variables))

            if is_win:
                loss1_value, grads2 = grad_(tape, model, prediction, current_target, loss1)
                optimizer.apply_gradients(zip(grads2, model.trainable_variables))

                loss_common, grads3 = grad_(tape, model, prediction, current_target, common_loss)
                optimizer.apply_gradients(zip(grads3, model.trainable_variables))

            loss1_value = loss1_value if is_win else None
            stat_holder.hold(step, cross_entropy_value, current_target, prediction, loss1_value)

        if 300 < step < 2000:
            if step % 100 == 0:
                run_test(model, step, logger)
        elif 2000 < step < 10000:
            if step % 500 == 0:
                run_test(model, step, logger)
        elif 10000 < step < 21000:
            if step % 2000 == 0:
                run_test(model, step, logger)
        elif 21000 < step:
            if step % 3000 == 0:
                run_test(model, step, logger)


if __name__ == '__main__':
    main()
