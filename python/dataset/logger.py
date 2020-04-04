from datetime import datetime

from python.dataset.dataset_builder import SEPARATOR
from python.dlf.bid_loss import ALPHA, BETA


class Logger:
    _PATH_PREFIX = '../output/'
    _LABELS = ['campaign', 'category', 'step', 'mean_loss', 'mean_anlp', 'mean_common', 'mean_auc']

    def __init__(self, campaign):
        self.campaign = campaign
        timestamp_str = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        self.log_file = Logger._PATH_PREFIX + 'dlf_%s_%s.tsv' % (campaign, timestamp_str)
        self._force_write(SEPARATOR.join(Logger._LABELS))

    def log(self, category, step, mean_loss, mean_anlp):
        stat = self._stat_str(category, step, mean_loss, mean_anlp)
        self._force_write(stat)
        print(stat)

    def _stat_str(self, category, step, mean_loss, mean_anlp):
        common_loss = ALPHA * mean_loss + BETA * mean_anlp if mean_anlp is not None else None
        log = [str(self.campaign), category, str(step),
               Logger._to_str('{:.6f}', mean_loss),
               Logger._to_str('{:.4f}', mean_anlp),
               Logger._to_str('{:.4f}', common_loss)]
        return SEPARATOR.join(log)

    def _force_write(self, info):
        with open(self.log_file, 'a') as logfile:
            logfile.write(info + '\n')

    @staticmethod
    def _to_str(x_format, x):
        return x_format.format(x) if x is not None else 'NULL'
