#!/usr/bin/env python
# coding=utf-8

import time

import tensorflow as tf

from python.dataset.data_reader import BiSparseData
from python.dataset.logger import Logger
from python.dataset.stat_holder import StatHolder
from python.dlf.dlf import DLF
from python.dlf.losses import (
    cross_entropy, loss1, grad_common_loss, loss_grad
)
from python.util import LossMode, DataMode

_TRAIN_STEP = 21010
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


def _get_dataset(path, campaign, data_mode=DataMode.ALL_DATA, is_train=True):
    mode = {
        DataMode.ALL_DATA: "all",
        DataMode.WIN_ONLY: "win",
        DataMode.LOSS_ONLY: "lose"
    }[data_mode]
    dataset_option = 'train' if is_train else 'test'
    return BiSparseData('%s/%s/%s_%s.tsv' % (path, campaign, dataset_option, mode), _BATCH_SIZE, is_train=is_train)


def train_model(campaign, loss_mode=LossMode.ALL_LOSS, data_mode=DataMode.ALL_DATA):
    logger = Logger(campaign, data_mode)

    stat_holder_train = StatHolder('TRAIN', logger)
    stat_holder_test = StatHolder('TEST', logger, is_train=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LEARNING_RATE, beta_2=_BETA_2)

    train_dataset = _get_dataset('../data', 'toy_datasets', data_mode)
    test_dataset = _get_dataset('../data', 'toy_datasets', data_mode, is_train=False)

    # train_dataset = _get_dataset('../data', str(campaign), data_mode)
    # test_dataset = _get_dataset('../data', str(campaign), data_mode, is_train=False)

    model = DLF()
    model.build(input_shape=([_BATCH_SIZE, 16], [_BATCH_SIZE, 2]))

    steps = min(_TRAIN_STEP, train_dataset.epoch_steps(2))
    for step in range(steps):
        current_features, current_bids, current_target, is_win = train_dataset.next()
        start_time = time.time()

        model.predict_on_batch([current_features, current_bids])
        with tf.GradientTape() as tape:
            print(model.trainable_variables)
            tape.watch(model.trainable_variables)
            survival_rate, rate_last = model.predict_on_batch([current_features, current_bids])

            if is_win and loss_mode == LossMode.ALL_LOSS:
                loss1_value = loss1(current_target, survival_rate)
                cross_entropy_value = cross_entropy(current_target, rate_last)

                loss_common, grads = grad_common_loss(tape, model.trainable_variables, loss1_value, cross_entropy_value)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            elif not is_win or loss_mode == LossMode.CROSS_ENTROPY:
                cross_entropy_value, grads = loss_grad(tape, model.trainable_variables, current_target, survival_rate, cross_entropy)
                loss1_value = None
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            else:
                cross_entropy_value = None
                loss1_value,  grads = loss_grad(tape, model.trainable_variables, current_target, survival_rate, cross_entropy_value)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            stat_holder_train.hold(
                step,
                cross_entropy=cross_entropy_value,
                targets=current_target,
                prediction=[survival_rate, rate_last],
                anlp_loss=loss1_value
            )

        print("Prev step %s worked %s sec" % (step, '{:.4f}'.format(time.time() - start_time)))
        if step > 0 and step % 10 == 0:
            stat_holder_train.flush(step)

        if 100 <= step < 500:
            if step % 100 == 0:
                run_test(model, step, test_dataset, stat_holder_test)
        elif 500 <= step < 2000:
            if step % 500 == 0:
                run_test(model, step, test_dataset, stat_holder_test)
        elif 2000 <= step < 10000:
            if step % 2000 == 0:
                run_test(model, step, test_dataset, stat_holder_test)
        elif 10000 <= step < 21000:
            if step % 3000 == 0:
                run_test(model, step, test_dataset, stat_holder_test)

    run_test(model, steps, test_dataset, stat_holder_test)


def main():
    train_model(2997)


if __name__ == '__main__':
    main()
