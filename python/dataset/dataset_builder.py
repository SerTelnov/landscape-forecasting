#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
import random
from python.util import (
    DataMode, SEPARATOR,
    data2str, loss2str
)

_LABELS = ["market_price", "bid", "weekday", "hour", "IP", "region", "city",
           "adexchange", "domain", "slotid", "slotwidth", "slotheight",
           "slotvisibility", "slotformat", "creative", "advertiser", "useragent", "slotprice"]


def build_toy_dataset(dataset_path, dataset_name, size, data_mode=DataMode.ALL_DATA):
    output_path = '../../data/toy_datasets/' + dataset_name
    with open(dataset_path, 'r') as input, \
            open(output_path, 'w') as output:
        count = 0
        while count < size:
            line = input.readline()
            sample = line.split(SEPARATOR)

            if data_mode == DataMode.ALL_DATA:
                output.write(line)
                count += 1
            else:
                market_prize = int(sample[0])
                bid = int(sample[1])

                if market_prize >= bid:
                    if data_mode == DataMode.LOSS_ONLY:
                        output.write(line)
                        count += 1
                elif data_mode == DataMode.WIN_ONLY:
                    output.write(line)
                    count += 1


def decrease_dataset(dataset_path, dataset_name, decrease_value=100):
    in_file_path = dataset_path + dataset_name
    out_file_path = dataset_path + 'low_' + dataset_name

    number_of_lines = sum(1 for _ in open(in_file_path, 'r'))
    new_number_of_lines = number_of_lines // decrease_value

    with open(in_file_path, 'r') as input, \
            open(out_file_path, 'w') as output:
        indices = np.arange(number_of_lines)
        random.shuffle(indices)
        indices = indices[:new_number_of_lines]

        line_number = 0
        for line in input:
            if line_number in indices:
                output.write(line)
            line_number += 1


def rebuild_dataset(dataset_path, out_dir, out_name_prefix, add_title=False, rebuild_mode=DataMode.ALL_DATA):
    out_file_path = _make_out_path(out_dir, out_name_prefix, rebuild_mode)

    with open(dataset_path, 'r') as input, \
            open(out_file_path, 'w') as output:
        if add_title:
            output.write(SEPARATOR.join(_LABELS) + '\n')

        for line in input:
            sample = line.split(' ')
            market_price = int(sample[1])
            bid = int(sample[2])

            if rebuild_mode == DataMode.WIN_ONLY and market_price >= bid:
                continue

            new_sample = [str(market_price), str(bid)] + \
                         list(map(lambda x: x.split(':')[0], sample[3:]))

            output.write(SEPARATOR.join(new_sample) + '\n')


def _make_out_path(out_dir, out_name_prefix, rebuild_mode):
    suffix = data2str(rebuild_mode)
    return out_dir + out_name_prefix + '_' + suffix + '.tsv'


class SocialNetDatasetRebuilder:

    def __init__(self, dataset_path, dataset_name):
        self._global_label = None
        self._dataset_path = dataset_path
        self._dataset_name = dataset_name

        self._global_counter = 0
        self._global_feature_map = {}

    def rebuild(self, new_filename):
        features_index_path = self._dataset_path + 'featindex.tsv'
        new_dataset_path = self._dataset_path + new_filename + '.tsv'
        path = self._dataset_path + self._dataset_name

        df = pd.read_csv(path).drop(['time', 'label'], axis=1)
        df = df[df.bid.between(10, 300)]

        labels = list(df)
        labels[0] = 'market_price'
        labels[1] = 'bid'

        df = df[labels]

        for label in labels[2:]:
            self._global_feature_map[label] = {}
            self._global_label = label
            df[label] = df[label].map(self._count_feature)

        with open(features_index_path, 'w') as feat_out:
            feat_out.write(SEPARATOR.join(['feature', 'value', 'number']) + '\n')

            for label in labels[2:]:
                for value, num in self._global_feature_map[label].items():
                    line = SEPARATOR.join(map(str, [label, value, num]))
                    feat_out.write(line + '\n')

        df.to_csv(new_dataset_path, sep=SEPARATOR, header=False, index=False)

    def _count_feature(self, x):
        feature_map = self._global_feature_map[self._global_label]
        x = str(x)

        if x not in feature_map:
            self._global_counter += 1
            feature_map[x] = self._global_counter
        return feature_map[x]

# decrease_dataset(
#     dataset_path='../../data/3476/',
#     dataset_name='test.yzbx.txt',
#     decrease_value=10
# )


# rebuild_dataset(
#     dataset_path='../../data/2997/test.yzbx.txt',
#     out_name_prefix='test',
#     out_dir='../../data/2997/'
# )


# build_toy_dataset(
#     dataset_path='../../data/3476/test_all.tsv',
#     dataset_name='train_all.tsv',
#     size=2048,
#     data_mode=DataMode.ALL_DATA
# )

def main():
    SocialNetDatasetRebuilder(
        dataset_path='../../data/vk1/',
        dataset_name='ad_1_test.csv'
    ).rebuild('test_all')


if __name__ == '__main__':
    main()
