#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

from python.common.early_stopping import EarlyStopping
from python.common.losses import (
    cross_entropy, loss1, grad_common_loss, loss_grad
)
from python.common_test import run_test
from python.dataset.data_reader import read_dataset
from python.dataset.logger import Logger
from python.dataset.stat_holder import StatHolder
from python.model_util import make_model
from python.util import (
    LossMode, DataMode, ModelMode, LEARNING_RATE, NUMBER_OF_EPOCH
)

_TRAIN_STEP = 21010

_BETA_2 = 0.99


def train_model(campaign, model_mode, loss_mode=LossMode.ALL_LOSS, data_mode=DataMode.ALL_DATA, data_path='../'):
    logger = Logger(
        campaign=campaign,
        model_mode=model_mode,
        data_mode=data_mode,
        loss_mode=loss_mode
    )

    stat_holder_train = StatHolder('TRAIN', logger)
    stat_holder_test = StatHolder('TEST', logger, is_train=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_2=_BETA_2)

    # train_dataset = read_dataset('../data', 'toy_datasets', data_mode)
    # test_dataset = read_dataset('../data', 'toy_datasets', data_mode, is_train=False)

    train_dataset = read_dataset(data_path + 'data', str(campaign), data_mode)
    test_dataset = read_dataset(data_path + 'data', str(campaign), data_mode, is_train=False)

    model = make_model(model_mode)
    early_stopping = EarlyStopping(model, optimizer, logger.model_name)

    steps = train_dataset.epoch_steps()
    train_finished = False
    step_number = 0

    for epoch in range(NUMBER_OF_EPOCH):
        if train_finished:
            break

        for step in range(steps):
            current_features, current_bids, current_target, is_win = train_dataset.next()
            # start_time = time.time()

            with tf.GradientTape() as tape:
                tape.watch(model.trainable_variables)
                survival_rate, rate_last = model.predict_on_batch([current_features, current_bids])

                cross_entropy_value = None
                loss1_value = None
                if is_win and loss_mode == LossMode.ALL_LOSS:
                    cross_entropy_value = cross_entropy(current_target, survival_rate)
                    loss1_value = loss1(current_target, rate_last)

                    loss_common, grads = grad_common_loss(
                        tape,
                        tvar=model.trainable_variables,
                        loss1_value=loss1_value,
                        loss2_value=cross_entropy_value
                    )
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                elif is_win and loss_mode == LossMode.ANLP:
                    loss1_value, grads = loss_grad(
                        tape,
                        tvar=model.trainable_variables,
                        target=current_target,
                        prediction=rate_last,
                        loss_function=loss1
                    )
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                elif loss_mode != LossMode.ANLP:
                    cross_entropy_value, grads = loss_grad(
                        tape,
                        tvar=model.trainable_variables,
                        target=current_target,
                        prediction=survival_rate,
                        loss_function=cross_entropy
                    )
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

            step_number = epoch * steps + step
            stat_holder_train.hold(
                step_number,
                cross_entropy=cross_entropy_value,
                anlp_loss=loss1_value,
                targets=current_target,
                prediction=[survival_rate, rate_last]
            )

            # print("Prev step %s worked %s sec" % (step_number, '{:.4f}'.format(time.time() - start_time)))
            # if step_number > 0 and step_number % 10 == 0:
            #     stat_holder_train.flush(step_number)

            if step_number > 0 and step_number % 500 == 0:
                anlp, auc = run_test(model, step, test_dataset, stat_holder_test, DataMode.WIN_ONLY, test_all_data=True)
                train_finished = early_stopping.check(step_number, anlp)

                if train_finished:
                    print('Early stopping!!')
                    break

            if 100 <= step_number < 500:
                if step_number % 100 == 0:
                    run_test(model, step_number, test_dataset, stat_holder_test)
            elif 500 <= step_number < 2000:
                if step_number % 500 == 0:
                    run_test(model, step_number, test_dataset, stat_holder_test)
            elif 2000 <= step_number < 10000:
                if step_number % 2000 == 0:
                    run_test(model, step_number, test_dataset, stat_holder_test)
            elif 10000 <= step_number < 21000:
                if step_number % 3000 == 0:
                    run_test(model, step_number, test_dataset, stat_holder_test)
            elif 21000 < step_number:
                if step_number % 5000 == 0:
                    run_test(model, step_number, test_dataset, stat_holder_test)

        if not train_finished:
            print('epoch #%d came to the end' % (epoch + 1))
            run_test(model, step_number, test_dataset, stat_holder_test, test_all_data=True)

    early_stopping.times_up()
    run_test(model, step_number, test_dataset, stat_holder_test, test_all_data=True)
