from enum import Enum


class LossMode(Enum):
    ALL_LOSS = 1
    CROSS_ENTROPY = 2
    ANLP = 3


class DataMode(Enum):
    ALL_DATA = 1
    WIN_ONLY = 2
    LOSS_ONLY = 3


SMALL_VALUE = 1e-20
ALPHA = 0.25
BETA = 0.2
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
