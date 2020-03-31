from datetime import datetime

from python.dataset.dataset_builder import SEPARATOR
from python.dlf.bin_loss import ALPHA, BETA


class Logger:
    _PATH_PREFIX = '../output/'
    _LABELS = ['campaign', 'category', 'step', 'mean_loss', 'mean_anlp', 'mean_common', 'mean_auc']

    def __init__(self, campaign):
        self.campaign = campaign
        timestamp_str = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        self.log_file = Logger._PATH_PREFIX + 'dlf_%s_%s.tsv' % (campaign, timestamp_str)
        self._force_write(SEPARATOR.join(Logger._LABELS))

    def log(self, category, step, mean_loss, mean_anlp, mean_auc):
        stat = self._stat_str(category, step, mean_loss, mean_anlp, mean_auc)
        self._force_write(stat)
        print(stat)

    def _stat_str(self, category, step, mean_loss, mean_anlp, mean_auc):
        return str(self.campaign) + "\t" + category + "\t" + str(step) + "\t" \
               "{:.6f}".format(mean_loss) + "\t" + \
               "{:.4f}".format(mean_anlp) + "\t" + \
               "{:.4f}".format(ALPHA * mean_loss + BETA * mean_anlp) + "\t" + \
               "{:.4f}".format(mean_auc)

    def _force_write(self, info):
        with open(self.log_file, 'a') as logfile:
            logfile.write(info)
