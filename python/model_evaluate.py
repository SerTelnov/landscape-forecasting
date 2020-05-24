import os

import tensorflow as tf
import numpy as np

from python.dlf.dlf import DLF
from python.util import BATCH_SIZE, ModelMode, DataMode
from python.dataset.data_reader import read_dataset, read_first
from python.model_util import make_model


def get_probabilities(h):
    distribution = []
    W = []

    prefix = 1
    for i in bid_range:
        prob = (1 - h[i]) * prefix
        distribution.append(prob)

        prefix = prefix * h[i]
        W.append(1 - prefix)

    return {'Distribution': distribution, 'Winning': W}


model = make_model(ModelMode.TRANSFORMER, 'tlf_vk1_all__0.25_0.75_0.0001_20200522_0825', training_mode=False)

path = '../data'
campaign = 'vk1'
bid_range = np.arange(0, 309)

features, bids, _, _ = read_first(path, campaign, DataMode.LOSS_ONLY)
pred = model.predict_on_batch([features, bids])

for idx in range(0, len(bids), 10):
    z, b = bids[idx]
    if z == 144 and b == 29:
        get_probabilities(pred[idx])
