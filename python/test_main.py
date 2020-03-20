import tensorflow as tf

from python.dataset.data import get_dataset
from python.dlf.model import DLF
from python.dlf.bin_loss import (
    cross_entropy, loss1
)


def main():
    model = DLF()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer, loss=loss1)

    input, y = get_dataset("../data/toy_dataset_train.tsv")

    model.fit(input, y, batch_size=128, epochs=3)

    x, y = get_dataset("../data/toy_dataset_test.tsv")
    results = model.evaluate(x, y, batch_size=128)
    print('test loss, test acc:', results)

    print(model.summary())


if __name__ == '__main__':
    main()
