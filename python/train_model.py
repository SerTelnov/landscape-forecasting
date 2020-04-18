#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import time

from python.dataset.data_reader import BiSparseData
from python.dataset.logger import Logger
from python.dataset.stat_holder import StatHolder
from python.dlf.dlf import DLF
from python.util import LossMode, DataMode

from python.dlf.losses import (
    cross_entropy, loss1, grad_common_loss, loss_grad
)

_TRAIN_STEP = 21000
_TEST_STEP = 310
_BATCH_SIZE = 128

_LEARNING_RATE = 1e-3
_BETA_2 = 0.99

ALPHA = 0.25
BETA = 0.2


def run_test_win(model, step, dataset, stat_holder):
    print("Win data TEST")
    steps = min(_TEST_STEP, dataset.win_chunks_number())
    for i in range(steps):
        if i > 0 and i % 100 == 0:
            print("Iter number #%s" % i)

        features, bids, targets = dataset.next_win()
        survival_rate, rate_last = model.predict_on_batch([features, bids])
        cross_entropy_value = cross_entropy(targets, survival_rate)
        loss1_value = loss1(targets, rate_last)
        stat_holder.hold(step, cross_entropy_value, targets, [survival_rate, rate_last], loss1_value)


def run_test_loss(model, step, dataset, stat_holder):
    print("Loss data TEST")
    steps = min(_TEST_STEP, dataset.loss_chunks_number())
    for i in range(steps):
        if i > 0 and i % 100 == 0:
            print("Iter number #%s" % i)
        features, bids, targets = dataset.next_loss()
        survival_rate, rate_last = model.predict_on_batch([features, bids])
        cross_entropy_value = cross_entropy(targets, survival_rate)
        stat_holder.hold(step, cross_entropy_value, targets, [survival_rate, rate_last], None)


def run_test(model, step, dataset, stat_holder):
    print('Test started...')
    run_test_win(model, step, dataset, stat_holder)
    run_test_loss(model, step, dataset, stat_holder)

    stat_holder.flush(step)
    dataset.reshuffle()


def train_cross_entropy_only(campaign):
    logger = Logger(campaign, DataMode.ALL_DATA, 'cross_entropy')

    model = DLF(LossMode.CROSS_ENTROPY)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=_LEARNING_RATE, beta_2=_BETA_2),
        loss=cross_entropy
    )

    test_dataset = BiSparseData('../data/toy_datasets/%s_all.tsv' % campaign, _BATCH_SIZE, is_train=False)
    train_dataset = BiSparseData('../data/toy_datasets/%s_all.tsv' % campaign, _BATCH_SIZE)

    for step in range(101):
        current_features, current_bids, current_target, is_win = train_dataset.next()
        start_time = time.time()
        loss_out = model.train_on_batch([current_features, current_bids], y=[current_target])
        print("Prev step %s worked %s sec" % (step, '{:.4f}'.format(time.time() - start_time)))
        print(loss_out)


def train_all(campaign):
    logger = Logger(campaign, DataMode.ALL_DATA)

    stat_holder_train = StatHolder('TRAIN', logger)
    stat_holder_test = StatHolder('TEST', logger, is_train=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LEARNING_RATE, beta_2=_BETA_2)

    test_dataset = BiSparseData('../data/toy_datasets/%s_all.tsv' % campaign, _BATCH_SIZE, is_train=False)
    train_dataset = BiSparseData('../data/toy_datasets/%s_all.tsv' % campaign, _BATCH_SIZE)

    # test_dataset = BiSparseData('data/3476/test_all.tsv', _BATCH_SIZE, is_train=False)
    # train_dataset = BiSparseData('data/3476/train_all.tsv', _BATCH_SIZE)

    model = DLF()
    model.build(input_shape=([_BATCH_SIZE, 16], [_BATCH_SIZE, 2]))

    for step in range(_TRAIN_STEP):
        current_features, current_bids, current_target, is_win = train_dataset.next()
        start_time = time.time()

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            survival_rate, rate_last = model.predict_on_batch([current_features, current_bids])

            if is_win:
                loss1_value = loss1(current_target, survival_rate)
                cross_entropy_value = cross_entropy(current_target, rate_last)

                loss_common, grads = grad_common_loss(tape, model.trainable_variables, loss1_value, cross_entropy_value)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            else:
                cross_entropy_value, grads = loss_grad(tape, model.trainable_variables, current_target, survival_rate, cross_entropy)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            loss1_value = loss1_value if is_win else None
            stat_holder_train.hold(step, cross_entropy_value, current_target, [survival_rate, rate_last], loss1_value)

        print("Prev step %s worked %s sec" % (step, '{:.4f}'.format(time.time() - start_time)))
        if step > 0 and step % 10 == 0:
            stat_holder_train.flush(step)

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


def main():
    train_all(3476)
    # train_cross_entropy_only(3476)


if __name__ == '__main__':
    main()
