#!/usr/bin/env python
# coding=utf-8

import numpy as np
import random
from enum import Enum


class DataMode(Enum):
    ALL_DATA = 1
    WIN_ONLY = 2
    LOSS_ONLY = 3


_LABELS = ["market_price", "bid", "weekday", "hour", "IP", "region", "city",
           "adexchange", "domain", "slotid", "slotwidth", "slotheight",
           "slotvisibility", "slotformat", "creative", "advertiser", "useragent", "slotprice"]

SEPARATOR = '\t'


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
    suffix = {
        DataMode.ALL_DATA: "all",
        DataMode.WIN_ONLY: "win"
    }[rebuild_mode]
    return out_dir + out_name_prefix + '_' + suffix + '.tsv'


decrease_dataset(
    dataset_path='../../data/3476/',
    dataset_name='train_all.tsv'
)

# rebuild_dataset(
#     dataset_path='../../data/3476/test.yzbx.txt',
#     out_name_prefix='test',
#     out_dir='../../data/3476/'
# )


# build_toy_dataset(
#     dataset_path='../../data/3476/test_all.tsv',
#     dataset_name='3476_all.tsv',
#     size=2048,
#     data_mode=DataMode.ALL_DATA
# )
