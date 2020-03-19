import tensorflow as tf

from python.dataset.data import get_dataset
from python.dlf.model import DLF
from python.dlf.bin_loss import cross_entropy


def main():
    model = DLF()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer, loss=cross_entropy)

    input, y = get_dataset("../data/1458/train_all.tsv")

    model.fit(input, y, batch_size=128, epochs=2)
    print(model.summary())


if __name__ == '__main__':
    main()
