#!/usr/bin/env python
# coding=utf-8

from datetime import datetime

from python.dataset.dataset_builder import SEPARATOR
from python.dlf.losses import ALPHA, BETA
from python.dataset.dataset_builder import DataMode


class Logger:
    _PATH_PREFIX = '../output/'
    _LABELS = ['campaign', 'category', 'step', 'cross_entropy', 'anlp', 'common_loss', 'mean_auc']

    def __init__(self, campaign, data_mode, loss_name=''):
        self.campaign = campaign
        self.log_file = self._get_log_name(campaign, data_mode, loss_name)
        self._force_write(SEPARATOR.join(Logger._LABELS))

    def log(self, category, step, cross_entropy, mean_anlp, mean_auc):
        stat = self._stat_str(category, step, cross_entropy, mean_anlp, mean_auc)
        self._force_write(stat)
        print(stat)

    def _stat_str(self, category, step, mean_cross_entropy, mean_anlp, mean_auc):
        common_loss = ALPHA * mean_cross_entropy + BETA * mean_anlp if mean_anlp else None
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
    def _get_log_name(campaign, data_mode, loss_name):
        timestamp_str = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        data_mode_str = {
            DataMode.ALL_DATA: 'all',
            DataMode.WIN_ONLY: 'win',
            DataMode.LOSS_ONLY: 'loss'
        }[data_mode]
        return Logger._PATH_PREFIX + 'dlf_%s_%s_%s_%s.tsv' % (campaign, data_mode_str, loss_name, timestamp_str)
