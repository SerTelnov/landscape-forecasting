#!/usr/bin/env python
# coding=utf-8

import numpy as np
from sklearn.metrics import roc_auc_score


class StatHolder:

    def __init__(self, category, logger, is_train=True):
        self.anlp_loss = []
        self.auc = []
        self.cross_entropy = []
        self.auc_label = []
        self.auc_prob = []
        self.logger = logger
        self.category = category
        self.is_train = is_train

    def hold(self, step, cross_entropy, targets=None, prediction=None, anlp_loss=None):
        if cross_entropy is not None:
            self.cross_entropy.append(cross_entropy)

        self.auc_label.append(targets)
        self.auc_prob.append(prediction[0].numpy())

        if anlp_loss is not None:
            self.anlp_loss.append(anlp_loss)
        self._log(step)

    def _log(self, step):
        if (step != 0 and step % 100 == 0) and self.is_train:
            self.flush(step)

    def flush(self, step):
        mean_anlp = StatHolder._mean_value(self.anlp_loss)
        mean_loss = StatHolder._mean_value(self.cross_entropy)
        mean_auc = roc_auc_score(self.auc_label, self.auc_prob)

        self.logger.log(self.category, step, mean_loss, mean_anlp, mean_auc)

        self.anlp_loss = []
        self.cross_entropy = []
        self.auc_label = []
        self.auc_prob = []

    @staticmethod
    def calc_roc_auc_score(Y_true, Y_pred):
        return StatHolder._roc_score(
            np.reshape(Y_true, [1, -1])[0],
            np.reshape(Y_pred, [1, -1])[0]
        )

    @staticmethod
    def _mean_value(values):
        return np.array(values).mean() if len(values) != 0 else None

    @staticmethod
    def _roc_score(y_true, y_pred):
        if len(np.unique(y_true)) > 1:
            return roc_auc_score(y_true, y_pred)
        return 0.0001
