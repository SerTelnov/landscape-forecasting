from enum import Enum
import pandas as pd


class DataMode(Enum):
    ALL_DATA = 1
    WIN_ONLY = 2


_LABELS = ["market_price", "bid", "weekday", "hour", "IP", "region", "city",
          "adexchange", "domain", "slotid", "slotwidth", "slotheight",
          "slotvisibility", "slotformat", "creative", "advertiser", "useragent", "slotprice"]

_SEPARATOR = '\t'


def get_dataset(dataset_path):
    df = pd.read_csv(dataset_path, sep=_SEPARATOR)
    X = df[["market_price", "bid"]].to_numpy()
    Y = df.drop(["market_price", "bid"], axis=1).to_numpy()
    return X, Y


def rebuild_dataset(dataset_path, out_dir, out_name_prefix, rebuild_mode=DataMode.ALL_DATA):
    out_file_path = _make_out_path(out_dir, out_name_prefix, rebuild_mode)

    with open(dataset_path, 'r') as input, \
            open(out_file_path, 'w') as output:
        output.write(_SEPARATOR.join(_LABELS) + '\n')

        for line in input:
            sample = line.split(' ')
            market_price = int(sample[1])
            bid = int(sample[2])

            if rebuild_mode == DataMode.WIN_ONLY and market_price >= bid:
                continue

            new_sample = [str(market_price), str(bid)] + \
                         list(map(lambda x: x.split(':')[0], sample[3:]))

            output.write(_SEPARATOR.join(new_sample) + '\n')


def _make_out_path(out_dir, out_name_prefix, rebuild_mode):
    suffix = {
        DataMode.ALL_DATA: "all",
        DataMode.WIN_ONLY: "win"
    }[rebuild_mode]
    return out_dir + out_name_prefix + '_' + suffix + '.tsv'


# rebuild_dataset(
#     dataset_path='../../data/1458/train.yzbx.txt',
#     out_name_prefix='train',
#     out_dir='../../data/1458/',
#     rebuild_mode=RebuildMode.WIN_ONLY
# )
