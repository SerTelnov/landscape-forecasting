import os

import tensorflow as tf

from python.dlf.dlf import DLF
from python.util import BATCH_SIZE
from python.dataset.data_reader import read_dataset
from python.dlf.losses import cross_entropy


checkpoint_path = '../output/checkpoint/dlf_2997_all__0.25_0.75_0.0001_20200502_1648/cp-{epoch:02d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)

model = DLF()
model.build(input_shape=([BATCH_SIZE, 16], [BATCH_SIZE, 2]))

test_dataset = read_dataset('../data', 'toy_datasets', is_train=False)

features, bids, target, _ = test_dataset.next()
survival_rate, _ = model.predict_on_batch([features, bids])
print(cross_entropy(target, survival_rate))

tvars = model.trainable_variables[:]
model.load_weights(latest)

survival_rate, _ = model.predict_on_batch([features, bids])
print(cross_entropy(target, survival_rate))


print(tvars == model.trainable_variables)