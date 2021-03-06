#!/usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf


class EarlyStopping:

    def __init__(self, model, optimizer, model_name, model2load=None, min_delta=0.0001, comp=np.greater):
        self.model = model
        self.optimizer = optimizer
        self.model_name = model_name
        self.min_delta = abs(min_delta)
        self.best = np.Inf
        self.previous = np.Inf
        self.comp = comp

        checkpoint_path = 'output/checkpoint/' + model_name
        if model2load is not None:
            checkpoint_path = 'output/checkpoint/' + model2load

        self.ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=3)

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Restore model %s' % model2load)

    def times_up(self):
        self._restore_best()

    def check(self, step, loss_score):
        if abs(self.previous - loss_score) < self.min_delta:
            print('Training is over on step %s' % step)
            self._save(step, loss_score)
            return True
        self.previous = loss_score

        if loss_score / self.best > 1.5:
            print('Overfitting!!')
            self._restore_best()
            return True

        if self.comp(self.best, loss_score):
            self.best = loss_score
            self._save(step, loss_score)
        return False

    def _save(self, step, score):
        self.ckpt_manager.save()
        print('Saving checkpoint in step %s with ANLP %s' % (step, score))

    def _restore_best(self):
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        print('Restore best result')
