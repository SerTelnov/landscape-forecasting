from enum import Enum


class DataMode(Enum):
    ALL_DATA = 1
    WIN_ONLY = 2
    LOSS_ONLY = 3


_LABELS = ["market_price", "bid", "weekday", "hour", "IP", "region", "city",
           "adexchange", "domain", "slotid", "slotwidth", "slotheight",
           "slotvisibility", "slotformat", "creative", "advertiser", "useragent", "slotprice"]

SEPARATOR = '\t'


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


# rebuild_dataset(
#     dataset_path='../../data/3476/test.yzbx.txt',
#     out_name_prefix='test',
#     out_dir='../../data/3476/'
# )
