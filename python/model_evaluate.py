import os

import tensorflow as tf

from python.dlf.dlf import DLF
from python.util import BATCH_SIZE
from python.dataset.data_reader import read_dataset
from python.dlf.losses import cross_entropy


checkpoint_path = '../output/checkpoint/aws/dlf_2997_all__0.25_0.75_0.0001_20200502_1437/cp-{epoch:02d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)

model = DLF()
model.build(input_shape=([BATCH_SIZE, 16], [BATCH_SIZE, 2]))

dataset = read_dataset('../data', 'toy_datasets')

idx = 1
features, bids, target = dataset.next_loss()
# bids[idx][1] = 5

model.load_weights(latest)

h = model.predict_on_batch([features, bids])
h = h.numpy()[idx]
print(h)
