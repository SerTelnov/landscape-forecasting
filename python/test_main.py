import tensorflow as tf
from python.dlf.model import DLF
from python.dataset.data import get_dataset


def main():
    model = DLF()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

    X, Y = get_dataset("../data/1458/train_all.tsv")

    model.fit(X, Y, batch_size=128)
    print(model.summary())


if __name__ == '__main__':
    main()
