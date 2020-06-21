from python.dataset.logger import Logger
from python.dataset.stat_holder import StatHolder
from python.dataset.data_reader import read_dataset
from python.util import ModelMode, DataMode, LossMode, model2str
from python.model_util import make_model
from python.common_test import run_test


def _get_stat_holder(campaign, model_mode, logger_filename):
    logger = Logger(
        campaign=campaign,
        model_mode=model_mode,
        data_mode=DataMode.ALL_DATA,
        loss_mode=LossMode.ALL_LOSS,
        log_filename='%s_%s' % (model2str(model_mode), logger_filename)
    )
    return StatHolder('TEST', logger, is_train=False)


def test_models(campaign, models_path, models_mode):
    dataset = read_dataset('data', str(campaign), is_train=False)

    for i, model_path in enumerate(models_path):
        stat_holder = _get_stat_holder(campaign, models_mode, 'model%s' % (i + 1))
        model = make_model(models_mode, model_path)

        run_test(model, 0, dataset, stat_holder, data_mode=DataMode.ALL_DATA, test_all_data=True)


def main():
    models = ['../aws-results/interval/checkpoints/tlf_3476_all__0.25_0.75_0.0001_20200527_1412',
              '../aws-results/interval/checkpoints/tlf_3476_all__0.25_0.75_0.0001_20200527_1416',
              '../aws-results/interval/checkpoints/tlf_3476_all__0.25_0.75_0.0001_20200527_1713']
    test_models(3476, models, ModelMode.TRANSFORMER)


if __name__ == '__main__':
    main()
