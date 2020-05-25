import pandas as pd
import math

from sklearn.metrics import roc_auc_score

from python.util import SEPARATOR

_LABELS = ["market_price", "bid", "weekday", "hour", "IP", "region", "city",
           "adexchange", "domain", "slotid", "slotwidth", "slotheight",
           "slotvisibility", "slotformat", "creative", "advertiser", "useragent", "slotprice"]

_LABELS_vk = ['market_price', 'bid', 'domain_hash', 'fb_index',
              'user_age', 'user_sex', 'device_model', 'os', 'user_ip', 'country_id',
              'city_id', 'weekday', 'hour', 'fb_section_id', 'ads_section_id', 'ads_platform_id']


class KM:

    def __init__(self):
        self.winning_prob = None
        self.max_bid = 0

    def fit(self, bid_log):
        bid_info = [{'wins': 0, 'loses': 0} for _ in range(0, 310)]

        for b, z in bid_log:
            mode = 'wins' if self._is_win(b, z) else 'loses'
            bid_info[b][mode] += 1

        counter = len(bid_log)

        prefix = 1.0
        self.winning_prob = {}
        for bid, info in enumerate(bid_info):
            total = info['wins'] + info['loses']
            if total > 0:
                d = info['wins']
                n = counter
                counter -= total

                if n == 0:
                    prefix = 0
                else:
                    prefix *= (n - d) / n

                prob_win = max(1 - prefix, 0)
                self.winning_prob[bid] = prob_win
                self.max_bid = max(self.max_bid, bid)

    def metric(self, bid_log):
        win_prediction = []
        labels = []

        anlp = 0
        for b, z in bid_log:
            x = 1 if self._is_win(b, z) else 0
            labels.append(x)
            b_pred = self._win_prob(b)
            z_pred = self._win_prob(z)
            z_next_pred = self._win_prob(z + 1)

            anlp -= self._save_log(abs(z_next_pred - z_pred))
            win_prediction.append(b_pred)

        anlp /= len(bid_log)
        auc = roc_auc_score(labels, win_prediction)
        return anlp, auc

    @staticmethod
    def _save_log(x):
        if x <= 0:
            x = 1e-10
        return math.log(x)

    def _win_prob(self, bid):
        while bid not in self.winning_prob and bid <= self.max_bid:
            bid += 1
        if bid > self.max_bid:
            return 1
        return self.winning_prob[bid]

    @staticmethod
    def _is_win(b, z):
        return not (b < z or z == 0)


def get_bid_log(mode, campaign):
    path = '../../data/'

    dataset_path = '%s%s/%s_all.tsv' % (path, campaign, mode)
    df = pd.read_csv(dataset_path, sep=SEPARATOR, names=_LABELS_vk)
    df = df[['bid', 'market_price']]

    return df.values.tolist()


def main():
    for i in range(0, 12):
        campaign = 'vk%s' % (i + 1)
        bid_log = get_bid_log('train', campaign)

        km = KM()
        km.fit(bid_log)

        bid_log = get_bid_log('test', campaign)

        anlp, auc = km.metric(bid_log)
        print('%s: %s %s' % (campaign, anlp, auc))


if __name__ == '__main__':
    main()
