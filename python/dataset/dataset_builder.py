#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle
from python.util import (
    DataMode, SEPARATOR,
    data2str, loss2str
)

_LABELS = ["market_price", "bid", "weekday", "hour", "IP", "region", "city",
           "adexchange", "domain", "slotid", "slotwidth", "slotheight",
           "slotvisibility", "slotformat", "creative", "advertiser", "useragent", "slotprice"]

_LABELS_vk = ['market_price', 'bid', 'domain_hash', 'fb_index',
              'user_age', 'user_sex', 'device_model', 'os', 'user_ip', 'country_id',
              'city_id', 'weekday', 'hour', 'fb_section_id', 'ads_section_id', 'ads_platform_id']


def build_toy_dataset(dataset_path, dataset_name, size, data_mode=DataMode.ALL_DATA):
    input_path = dataset_path + dataset_name
    output_path = dataset_path + 'toy_datasets_' + dataset_name

    df = pd.read_csv(
        input_path,
        sep='\t',
        names=['market_price', 'bid', 'domain_hash', 'fb_index', 'user_age', 'user_sex', 'device_model', 'os',
               'user_ip', 'country_id', 'city_id', 'weekday', 'hour', 'fb_section_id', 'ads_section_id',
               'ads_platform_id']
    )

    if data_mode == DataMode.WIN_ONLY:
        df = df[(df.bid >= df.market_price)]
    elif data_mode == DataMode.LOSS_ONLY:
        df = df[(df.bid < df.market_price) | (df.market_price == 0)]

    df['bid'] = df['bid'].map(lambda x: 1 + int(random.random() * 100) % 5)

    df = df.sample(n=size, random_state=1)
    df.to_csv(output_path, sep=SEPARATOR, header=False, index=False)


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
    _LOSS_SAMPLES_COUNT = 24_000

    def __init__(self, dataset_path, dataset_train, dataset_test):
        self._global_label = None
        self._dataset_path = dataset_path
        self._dataset_train = dataset_train
        self._dataset_test = dataset_test

        self._global_counter = 0
        self._global_feature_map = {}
        self._number_counter = {}

    def rebuild(self):
        features_index_path = self._dataset_path + 'featindex.tsv'
        train_path = self._dataset_path + self._dataset_train
        test_path = self._dataset_path + self._dataset_test

        train_df = self._read_dataset(train_path)
        test_df = self._read_dataset(test_path)

        loss_df = test_df[(test_df.bid < test_df.market_price)] \
            .sample(self._LOSS_SAMPLES_COUNT)

        loss_df['bid'] = loss_df['bid'].map(lambda x: int(random.random() * 100) % 6)
        loss_df['market_price'] = 0

        train_df = shuffle(pd.concat([train_df, loss_df]))

        labels = list(train_df)[2:]

        for label in labels:
            self._global_feature_map[label] = {}
            self._global_label = label
            features = pd.concat([train_df[label], test_df[label]])
            features.sort_values() \
                .apply(self._count_feature)

            feature_map = self._global_feature_map[self._global_label]
            train_df[label] = train_df[label].map(lambda x: feature_map[str(x)])
            test_df[label] = test_df[label].map(lambda x: feature_map[str(x)])

        with open(features_index_path, 'w') as feat_out:
            feat_out.write(SEPARATOR.join(['feature', 'value', 'number', 'count']) + '\n')

            for label in labels:
                for value, num in self._global_feature_map[label].items():
                    count = self._number_counter[num]
                    line = SEPARATOR.join(map(str, [label, value, num, count]))
                    feat_out.write(line + '\n')

        train_df.to_csv(self._dataset_path + 'train_all.tsv', sep=SEPARATOR, header=False, index=False)
        test_df.to_csv(self._dataset_path + 'test_all.tsv', sep=SEPARATOR, header=False, index=False)

    def _read_dataset(self, path):
        df = pd.read_csv(path).drop(['label', 'time'], axis=1)
        df = df[(df['bid'].between(1, 305)) & (df['market_price'].between(1, 305)) & (df['user_sex'].between(1, 2)) & (
                    df.user_age < 90)]

        # df['device_model'] = df['device_model'].apply(lambda x: str(x).split(',')[0])
        df['user_ip'] = df['user_ip'].apply(lambda x: '.'.join(x.split('.')[:3]) + '.*')

        labels = list(df)
        labels[0] = 'market_price'
        labels[1] = 'bid'

        return df[labels]

    def _count_feature(self, x):
        feature_map = self._global_feature_map[self._global_label]
        x = str(x)

        if x not in feature_map:
            self._global_counter += 1
            feature_map[x] = self._global_counter
            self._number_counter[self._global_counter] = 0

        num = feature_map[x]
        self._number_counter[num] += 1
        return num


def dataset_info(path, dirs):
    for i, curr_dir in enumerate(dirs):
        stat = []
        for mode in ['train', 'test']:
            current_path = path + curr_dir + '/' + mode + '_all.tsv'
            df = pd.read_csv(current_path, sep=SEPARATOR, names=_LABELS_vk)
            win_count = df[(df['bid'] >= df['market_price'])].shape[0]
            loss_count = df[(df.bid < df.market_price) | (df.market_price == 0)].shape[0]

            stat.append(win_count)
            stat.append(loss_count)
        stat_str = ' & '.join(map(str, stat))
        line = '%s & %s \\\\' % (i + 1, stat_str)
        print(line)


def main():
    dirs = ['vk%s' % (i + 1) for i in range(0, 12)]
    dataset_info('../../data/', dirs)

    # SocialNetDatasetRebuilder(
    #     dataset_path='../../data/vk12/',
    #     dataset_train='ad_12_train.csv',
    #     dataset_test='ad_12_test.csv'
    # ).rebuild()


if __name__ == '__main__':
    main()
