from enum import Enum


class LossMode(Enum):
    ALL_LOSS = 1
    CROSS_ENTROPY = 2
    ANLP = 3


class DataMode(Enum):
    ALL_DATA = 1
    WIN_ONLY = 2
    LOSS_ONLY = 3

