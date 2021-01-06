from enum import Enum


class LossMode(Enum):
    ALL_LOSS = 'all'
    CROSS_ENTROPY = 'cross_entropy'
    ANLP = 'anlp'


class DataMode(Enum):
    ALL_DATA = 'all_data'
    WIN_ONLY = 'win'
    LOSS_ONLY = 'loss'


class ModelMode(Enum):
    DLF = 'dlf',
    DLF_ATTENTION = 'dlf_attention',
    TRANSFORMER = 'tlf'


SEPARATOR = '\t'

ALPHA = 0.25  # loss1
BETA = 0.75  # cross entropy
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
NUMBER_OF_EPOCH = 2


def loss2str(loss_mode: LossMode):
    return {
        LossMode.ALL_LOSS: '',
        LossMode.ANLP: 'anlp',
        LossMode.CROSS_ENTROPY: 'cross_entropy'
    }[loss_mode]


def data2str(data_mode: DataMode):
    return {
        DataMode.ALL_DATA: 'all',
        DataMode.WIN_ONLY: 'win',
        DataMode.LOSS_ONLY: 'loss'
    }[data_mode]


def model2str(model_mode: ModelMode):
    return {
        ModelMode.DLF: 'dlf',
        ModelMode.DLF_ATTENTION: 'dlf_attention',
        ModelMode.TRANSFORMER: 'tlf'
    }[model_mode]
