#!/usr/bin/env python
# coding=utf-8

import numpy as np
import random
import time
import math

from python.dataset.dataset_builder import DataMode, SEPARATOR


class SparseData:

    def __init__(self, file_path, data_mode):
        self.features = []
        self.labels = []
        self.bids = []

        with open(file_path, 'r') as fi:
            for line in fi:
                features = list(map(int, line.split(SEPARATOR)))
                market_price = features[0]
                bid_price = features[1]
                label = None

                if bid_price <= market_price:
                    if data_mode != DataMode.WIN_ONLY:
                        label = 0.
                elif data_mode != DataMode.LOSS_ONLY:
                    label = 1.

                if label is not None:
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

    def __init__(self, file_path, batch_size, is_train=True):
        random.seed(time.time())
        self.batch_size = batch_size
        self.winData = SparseData(file_path, DataMode.WIN_ONLY)
        self.loseData = SparseData(file_path, DataMode.LOSS_ONLY)
        self.size = self.winData.size + self.loseData.size
        self.is_train = is_train

    def epoch_steps(self, epoch_number):
        lose_batches = self.loseData.size // self.batch_size
        win_batches = self.winData.size // self.batch_size
        return (lose_batches + win_batches) * epoch_number

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

        # win = True
        # win = False
        current_data_type = self.winData if win else self.loseData
        features, bids, targets = current_data_type.next(self.batch_size, self.is_train)
        return features, bids, targets, win

    def get_all_data(self):
        return np.concatenate([self.winData.features, self.loseData.features]), \
                np.concatenate([self.winData.bids, self.loseData.bids]), \
                np.concatenate([self.winData.labels, self.loseData.labels])

    def next_win(self):
        return self.winData.next(self.batch_size, self.is_train)

    def next_loss(self):
        return self.loseData.next(self.batch_size, self.is_train)

    def chunks_number(self):
        return self.winData.number_of_chunks(self.batch_size) + \
               self.loseData.number_of_chunks(self.batch_size)

    def win_chunks_number(self):
        return self.winData.number_of_chunks(self.batch_size)

    def loss_chunks_number(self):
        return self.loseData.number_of_chunks(self.batch_size)

    def reshuffle(self):
        self.winData.reshuffle()
        self.loseData.reshuffle()
