#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

from python.dataset.data_reader import BiSparseData
from python.dataset.logger import Logger
from python.dataset.stat_holder import StatHolder
from python.dlf.bid_loss import (
    cross_entropy, loss1, grad_, grad_common_loss
)
from python.dlf.model import DLF

_TRAIN_STEP = 21000
_BATCH_SIZE = 128

_LEARNING_RATE = 1e-3
_BETA_2 = 0.99


def run_test_win(model, step, dataset, stat_holder):
    print("Win data TEST")
    for i in range(dataset.win_chunks_number() // 2):
        if i > 0 and i % 100 == 0:
            print("Iter number #" + str(i))

        features, targets = dataset.next_win()
        prediction = model(features)
        cross_entropy_value = cross_entropy(targets, prediction)
        loss1_value = loss1(targets, prediction)
        stat_holder.hold(step, cross_entropy_value, targets, prediction, loss1_value)


def run_test_all(model, step, dataset, stat_holder):
    print("Loss data TEST")
    for i in range(dataset.loss_chunks_number() // 2):
        if i > 0 and i % 500 == 0:
            print("Iter number #" + str(i))
        features, targets = dataset.next_loss()
        prediction = model(features)
        cross_entropy_value = cross_entropy(targets, prediction)
        stat_holder.hold(step, cross_entropy_value, targets, prediction, None)


def run_test(model, step, dataset, stat_holder):
    print('Test started...')
    run_test_win(model, step, dataset, stat_holder)
    run_test_all(model, step, dataset, stat_holder)

    stat_holder.flush(step)
    dataset.reshuffle()


def main():
    model = DLF()
    logger = Logger(3476)

    stat_holder_train = StatHolder('TRAIN', logger)
    stat_holder_test = StatHolder('TEST', logger, is_train=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LEARNING_RATE, beta_2=_BETA_2)

    # test_dataset = BiSparseData('../data/toy_datasets/3476_all.tsv', _BATCH_SIZE, is_train=False)
    # train_dataset = BiSparseData('../data/toy_datasets/3476_all.tsv', _BATCH_SIZE)

    test_dataset = BiSparseData('data/3476/test_all.tsv', _BATCH_SIZE, is_train=False)
    train_dataset = BiSparseData('data/3476/train_all.tsv', _BATCH_SIZE)

    model.build(input_shape=(_BATCH_SIZE, 18))

    for step in range(_TRAIN_STEP):
        current_features, current_target, is_win = train_dataset.next()
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(model.trainable_variables)
            prediction = model(current_features)

            cross_entropy_value, grads1 = grad_(tape, model, prediction, current_target, cross_entropy)
            optimizer.apply_gradients(zip(grads1, model.trainable_variables))

            if is_win:
                loss1_value, grads2 = grad_(tape, model, prediction, current_target, loss1)
                optimizer.apply_gradients(zip(grads2, model.trainable_variables))

                loss_common, grads3 = grad_common_loss(tape, model, cross_entropy_value, loss1_value)
                optimizer.apply_gradients(zip(grads3, model.trainable_variables))

            loss1_value = loss1_value if is_win else None
            stat_holder_train.hold(step, cross_entropy_value, current_target, prediction, loss1_value)

        if 300 <= step < 500:
            if step % 100 == 0:
                run_test(model, step, test_dataset, stat_holder_test)
        elif 500 <= step < 2000:
            if step % 500 == 0:
                run_test(model, step, test_dataset, stat_holder_test)
        elif 2000 <= step < 10000:
            if step % 2000 == 0:
                run_test(model, step, test_dataset, stat_holder_test)
        elif 21000 < step:
            if step % 3000 == 0:
                run_test(model, step, test_dataset, stat_holder_test)

    run_test(model, _TRAIN_STEP, test_dataset, stat_holder_test)


if __name__ == '__main__':
    main()
