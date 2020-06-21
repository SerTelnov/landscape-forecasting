from python.common.losses import (
    cross_entropy, loss1
)
from python.util import DataMode

_TEST_STEP = 300


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
    print("Iter number %d/%d" % (steps, steps))


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

    print("Iter number %d/%d" % (steps, steps))


def run_test(model, step, dataset, stat_holder, data_mode=DataMode.ALL_DATA, test_all_data=False):
    print('Test started...')
    if data_mode in [DataMode.ALL_DATA, DataMode.WIN_ONLY]:
        run_test_win(model, step, dataset, stat_holder, test_all_data)
    if data_mode in [DataMode.ALL_DATA, DataMode.LOSS_ONLY]:
        run_test_loss(model, step, dataset, stat_holder, test_all_data)

    dataset.reshuffle()
    return stat_holder.flush(step, data_mode)
