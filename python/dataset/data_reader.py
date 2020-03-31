import numpy as np
import random
import time

from python.dataset.dataset_builder import DataMode, SEPARATOR


class SparseData:

    def shuffle_indices(self):
        np.random.shuffle(self.indices)

    def __init__(self, file_path, data_mode):
        self.data = []
        self.labels = []
        with open(file_path, 'r') as fi:
            for line in fi:
                features = list(map(int, line.split(SEPARATOR)))
                market_price = int(features[0])
                bid_price = int(features[1])
                if bid_price <= market_price:
                    if data_mode != DataMode.WIN_ONLY:
                        self.data.append(features)
                        self.labels.append(0.)
                elif data_mode != DataMode.LOSS_ONLY:
                    self.data.append(features)
                    self.labels.append(1.)

        self.size = len(self.data)
        print("data size ", self.size, "\n")
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.indices = np.arange(self.size)
        self.shuffle_indices()
        self.batch_pointer = 0

    def has_next(self, batch_size):
        return self.batch_pointer + batch_size <= self.size

    def next(self, batch_size):
        if self.batch_pointer + batch_size > self.size:
            self.shuffle_indices()
            self.batch_pointer = 0
        indices = self.indices[self.batch_pointer:self.batch_pointer + batch_size]
        batch_data = self.data[indices]
        batch_labels = self.labels[indices]
        self.batch_pointer += batch_size
        return np.array(batch_data), np.array(batch_labels)


class BiSparseData:

    def __init__(self, file_path, batch_size):
        random.seed(time.time())
        self.batch_size = batch_size
        self.winData = SparseData(file_path, DataMode.WIN_ONLY)
        self.loseData = SparseData(file_path, DataMode.LOSS_ONLY)
        self.size = self.winData.size + self.loseData.size

    def next(self):
        win = int(random.random() * 100) % 11 <= 5
        # win = True
        current_data_type = self.winData if win else self.loseData
        features, targets = current_data_type.next(self.batch_size)
        return features, targets, win

    def next_win(self):
        return self.winData.next(self.batch_size)

    def has_next(self):
        return self.winData.has_next(self.batch_size)
