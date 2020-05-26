#!/usr/bin/env python
# coding=utf-8

import math
import random
import time
from itertools import islice

import numpy as np

from python.util import (
    DataMode, SEPARATOR, BATCH_SIZE, data2str
)


class SparseData:

    def __init__(self, file_path, data_mode):
        self.features = []
        self.labels = []
        self.bids = []

        with open(file_path, 'r') as fi:
            for line in fi:
                sample = SparseData.parse_line(line, data_mode)
                if sample is not None:
                    features, label = sample
                    self.features.append(features[2:])
                    self.bids.append(features[:2])
                    self.labels.append(label)

        self.size = len(self.features)
        print("data size ", self.size, "\n")
        self.features = np.array(self.features)
        self.bids = np.array(self.bids)
        self.labels = np.array(self.labels)
        self.indices = np.arange(self.size)
        self.shuffle_indices()
        self.batch_pointer = 0

    @staticmethod
    def parse_line(line, data_mode):
        features = list(map(int, line.split(SEPARATOR)))
        market_price = features[0]
        bid_price = features[1]
        label = None

        if bid_price < market_price or market_price == 0:
            if data_mode != DataMode.WIN_ONLY:
                label = 1.
        elif data_mode != DataMode.LOSS_ONLY:
            label = 0.

        if label is None:
            return None
        return features, label

    def reshuffle(self):
        self.batch_pointer = 0
        self.shuffle_indices()

    def shuffle_indices(self):
        np.random.shuffle(self.indices)

    def number_of_chunks(self, batch_size):
        return math.floor(len(self.features) / batch_size)

    def has_next(self, batch_size):
        return self.batch_pointer + batch_size <= self.size

    def next(self, batch_size, is_train):
        if self.batch_pointer + batch_size > self.size and is_train:
            self.shuffle_indices()
            self.batch_pointer = 0

        indices = self.indices[self.batch_pointer:self.batch_pointer + batch_size]
        self.batch_pointer += batch_size

        batch_features = self.features[indices]
        batch_bids = self.bids[indices]
        batch_labels = self.labels[indices]
        return np.array(batch_features), np.array(batch_bids), np.array(batch_labels)


class BiSparseData:

    def __init__(self, file_path, batch_size, data_mode=DataMode.ALL_DATA, is_train=True):
        random.seed(time.time())
        self.batch_size = batch_size
        self.is_train = is_train
        self.data_mode = data_mode

        if data_mode == DataMode.ALL_DATA:
            self.winData = SparseData(file_path, DataMode.WIN_ONLY)
            self.loseData = SparseData(file_path, DataMode.LOSS_ONLY)
            self.size = self.winData.size + self.loseData.size
        elif data_mode == DataMode.WIN_ONLY:
            self.winData = SparseData(file_path, DataMode.WIN_ONLY)
            self.loseData = None
            self.size = self.winData.size
        elif data_mode == DataMode.LOSS_ONLY:
            self.winData = None
            self.loseData = SparseData(file_path, DataMode.LOSS_ONLY)
            self.size = self.loseData.size

    def epoch_steps(self, epoch_number=1):
        return (self.size // self.batch_size) * epoch_number

    def win_epoch_steps(self, epoch_number=1):
        return self.winData.size // self.batch_size * epoch_number

    def loss_epoch_steps(self, epoch_number=1):
        return self.loseData.size // self.batch_size * epoch_number

    def next(self):
        win = int(random.random() * 100) % 11 <= 5

        if not self.is_train:
            has_loss = self.loseData.has_next(self.batch_size)
            has_win = self.winData.has_next(self.batch_size)

            if not has_loss and not has_win:
                raise Exception("No data")
            if not has_win:
                win = False
            elif not has_loss:
                win = True

        if self.data_mode != DataMode.ALL_DATA:
            win = self.data_mode == DataMode.WIN_ONLY

        # win = True
        # win = False
        current_data_type = self.winData if win else self.loseData
        features, bids, targets = current_data_type.next(self.batch_size, self.is_train)
        return features, bids, targets, win

    def next_win(self):
        return self.winData.next(self.batch_size, self.is_train)

    def next_loss(self):
        return self.loseData.next(self.batch_size, self.is_train)

    def get_all_data(self):
        return np.concatenate([self.winData.features, self.loseData.features]), \
                np.concatenate([self.winData.bids, self.loseData.bids]), \
                np.concatenate([self.winData.labels, self.loseData.labels])

    def chunks_number(self):
        return self.winData.number_of_chunks(self.batch_size) + \
               self.loseData.number_of_chunks(self.batch_size)

    def win_chunks_number(self):
        return self.winData.number_of_chunks(self.batch_size)

    def loss_chunks_number(self):
        return self.loseData.number_of_chunks(self.batch_size)

    def reshuffle(self):
        if self.winData:
            self.winData.reshuffle()
        if self.loseData:
            self.loseData.reshuffle()


def _get_dataset_name(path, campaign, is_train):
    data_name = 'all'
    dataset_option = 'train' if is_train else 'test'
    return '%s/%s/%s_%s.tsv' % (path, campaign, dataset_option, data_name)


def read_dataset(path, campaign, data_mode=DataMode.ALL_DATA, is_train=True):
    dataset_path = _get_dataset_name(path, campaign, is_train)
    return BiSparseData(dataset_path, BATCH_SIZE, data_mode, is_train)


def _lazy_next(path, campaign, data_mode, is_train):
    dataset_path = _get_dataset_name(path, campaign, is_train)
    with open(dataset_path, 'r') as input:
        while True:
            batch_features, batch_bids, batch_labels = [], [], []
            while len(batch_labels) < BATCH_SIZE:
                line = input.readline()
                sample = SparseData.parse_line(line, data_mode)
                if sample is not None:
                    features, label = sample
                    batch_features.append(features[2:])
                    batch_bids.append(features[:2])
                    batch_labels.append(label)

            a, b, c = list(map(np.array, [batch_features, batch_bids, batch_labels]))
            yield [a, b, c, label == 0.]


def _take(n, iterable):
    return list(islice(iterable, n))


def read_first_n(n, path, campaign, data_mode=DataMode.ALL_DATA, is_train=False):
    return _take(n, _lazy_next(path, campaign, data_mode, is_train))


def balance_read_n(n, path, campaign, is_train=False):
    k, m = n // 2, n - (n // 2)

    win_samples = _take(k, _lazy_next(path, campaign, DataMode.WIN_ONLY, is_train))
    loss_samples = _take(m, _lazy_next(path, campaign, DataMode.LOSS_ONLY, is_train))

    samples = []

    for x in win_samples:
        samples.append(x)
    for x in loss_samples:
        samples.append(x)

    return samples


def read_first(path, campaign, data_mode=DataMode.ALL_DATA, is_train=False):
    return _take(1, _lazy_next(path, campaign, data_mode, is_train))[0]
