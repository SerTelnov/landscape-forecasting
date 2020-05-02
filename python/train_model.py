#!/usr/bin/env python
# coding=utf-8

import time

import tensorflow as tf

from python.dataset.data_reader import read_dataset
from python.dataset.logger import Logger
from python.dataset.stat_holder import StatHolder
from python.dlf.dlf import DLF
from python.dlf.losses import (
    cross_entropy, loss1, grad_common_loss, loss_grad
)
from python.util import (
    LossMode, DataMode, ModelMode, LEARNING_RATE, BATCH_SIZE, NUMBER_OF_EPOCH
)

_TRAIN_STEP = 21010
_TEST_STEP = 310

_BETA_2 = 0.99


def run_test_win(model, step, dataset, stat_holder, test_all_data):
    print("Win data TEST")
    epoch_steps = dataset.win_epoch_steps()
    steps = epoch_steps if test_all_data else min(epoch_steps, _TEST_STEP)

    for i in range(steps):
        if i > 0 and i % 100 == 0:
            print("Iter number %d/%d" % (i, steps))

        features, bids, targets = dataset.next_win()
        survival_rate, rate_last = model.predict_on_batch([features, bids])
        cross_entropy_value = cross_entropy(targets, survival_rate)
        loss1_value = loss1(targets, rate_last)
        stat_holder.hold(
            step,
            cross_entropy=cross_entropy_value,
            targets=targets,
            prediction=[survival_rate, rate_last],
            anlp_loss=loss1_value
        )


def run_test_loss(model, step, dataset, stat_holder, test_all_data):
    print("Loss data TEST")
    epoch_steps = dataset.loss_epoch_steps()
    steps = epoch_steps if test_all_data else min(epoch_steps, _TEST_STEP)

    for i in range(steps):
        if i > 0 and i % 100 == 0:
            print("Iter number %d/%d" % (i, steps))
        features, bids, targets = dataset.next_loss()
        survival_rate, rate_last = model.predict_on_batch([features, bids])
        cross_entropy_value = cross_entropy(targets, survival_rate)
        stat_holder.hold(
            step,
            cross_entropy=cross_entropy_value,
            targets=targets,
            prediction=[survival_rate, rate_last],
            anlp_loss=None
        )


def run_test(model, step, dataset, stat_holder, test_all_data=False):
    print('Test started...')
    run_test_win(model, step, dataset, stat_holder, test_all_data)
    run_test_loss(model, step, dataset, stat_holder, test_all_data)

    stat_holder.flush(step)
    dataset.reshuffle()


def train_model(campaign, model_mode, loss_mode=LossMode.ALL_LOSS, data_mode=DataMode.ALL_DATA):
    logger = Logger(
        campaign=campaign,
        model_mode=model_mode,
        data_mode=data_mode,
        loss_mode=loss_mode
    )

    stat_holder_train = StatHolder('TRAIN', logger)
    stat_holder_test = StatHolder('TEST', logger, is_train=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_2=_BETA_2)

    train_dataset = read_dataset('../data', 'toy_datasets', data_mode)
    test_dataset = read_dataset('../data', 'toy_datasets', data_mode, is_train=False)

    # train_dataset = read_dataset('../data', str(campaign), data_mode)
    # test_dataset = read_dataset('../data', str(campaign), data_mode, is_train=False)

    model = DLF(model_mode)
    model.build(input_shape=([BATCH_SIZE, 16], [BATCH_SIZE, 2]))
    # model.run_eagerly = True

    # steps = min(_TRAIN_STEP, train_dataset.epoch_steps(NUMBER_OF_EPOCH))
    steps = train_dataset.epoch_steps(NUMBER_OF_EPOCH)

    for step in range(steps):
        current_features, current_bids, current_target, is_win = train_dataset.next()
        start_time = time.time()

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            survival_rate, rate_last = model.predict_on_batch([current_features, current_bids])

            cross_entropy_value = None
            if is_win and loss_mode == LossMode.ALL_LOSS:
                cross_entropy_value = cross_entropy(current_target, survival_rate)
                loss1_value = loss1(current_target, rate_last)

                loss_common, grads = grad_common_loss(tape, model.trainable_variables, loss1_value, cross_entropy_value)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            elif is_win and loss_mode == LossMode.ANLP:
                loss1_value,  grads = loss_grad(tape, model.trainable_variables, current_target, rate_last, loss1)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            elif loss_mode != LossMode.ANLP:
                cross_entropy_value, grads = loss_grad(tape, model.trainable_variables, current_target, survival_rate, cross_entropy)
                loss1_value = None
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

    run_test(model, steps, test_dataset, stat_holder_test, test_all_data=True)


def main():
    train_model(2997, model_mode=ModelMode.DLF)


if __name__ == '__main__':
    main()
