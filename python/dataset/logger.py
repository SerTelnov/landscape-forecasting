#!/usr/bin/env python
# coding=utf-8

from datetime import datetime

from python.util import BETA, ALPHA, LEARNING_RATE
from python.util import (
    LossMode, ModelMode, SEPARATOR,
    loss2str, data2str, model2str
)


class Logger:
    _PATH_PREFIX = '../output/'
    _LABELS = ['campaign', 'category', 'step', 'cross_entropy', 'mean_anlp', 'common_loss', 'mean_auc']

    def __init__(self, campaign, data_mode, loss_mode=LossMode.ALL_LOSS, model_mode=ModelMode.DLF):
        self.campaign = campaign
        self.log_file = self._get_log_name(campaign, model_mode, data_mode, loss_mode)
        self._force_write(SEPARATOR.join(Logger._LABELS))

    def log(self, category, step, cross_entropy, mean_anlp, mean_auc):
        stat = self._stat_str(category, step, cross_entropy, mean_anlp, mean_auc)
        self._force_write(stat)
        print(stat)

    def _stat_str(self, category, step, mean_cross_entropy, mean_anlp, mean_auc):
        common_loss = ALPHA * mean_anlp + BETA * mean_cross_entropy if mean_cross_entropy and mean_anlp else None
        log = [str(self.campaign), category, str(step),
               Logger._to_str('{:.6f}', mean_cross_entropy),
               Logger._to_str('{:.4f}', mean_anlp),
               Logger._to_str('{:.4f}', common_loss),
               Logger._to_str('{:.4f}', mean_auc)]
        return SEPARATOR.join(log)

    def _force_write(self, info):
        with open(self.log_file, 'a') as logfile:
            logfile.write(info + '\n')

    @staticmethod
    def _to_str(x_format, x):
        return x_format.format(x) if x is not None else 'NULL'

    @staticmethod
    def _get_log_name(campaign, model_mode, data_mode, loss_mode):
        timestamp_str = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        data_name = data2str(data_mode)
        loss_name = loss2str(loss_mode)
        model_name = model2str(model_mode)

        hyper_params = '%s_%s_%s' % (ALPHA, BETA, LEARNING_RATE)
        return Logger._PATH_PREFIX + '%s_%s_%s_%s_%s_%s.tsv' % \
            (model_name, campaign, data_name, loss_name, hyper_params, timestamp_str)
