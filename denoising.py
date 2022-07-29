import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from configuration import *

class ArtificialDenoisingAutoencoder(keras.Model):
    def __init__(self):
        super(ArtificialDenoisingAutoencoder, self).__init__()
        self.encoder = keras.Sequential([
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS)),
            layers.Conv2D(64, 4, strides=2, padding="same", activation="relu"),
            layers.Conv2D(32, 4, strides=2, padding="same", activation="relu"),
        ])

        self.decoder = keras.Sequential([
            layers.Conv2DTranspose(32, 4, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu"),
            layers.Conv2D(OUTPUT_CHANNELS, 4, padding="same", activation="tanh")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DenoisingPostProcess(keras.Model):
    def __init__(self, generator, train_ds, test_ds):
        super(DenoisingPostProcess, self).__init__()
        self.encoder = keras.Sequential([
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS)),
            layers.Conv2D(128, 4, strides=2, padding="same", activation="relu"),
            layers.Dropout(0.1),
            layers.Conv2D(64, 4, strides=2, padding="same", activation="relu"),
        ])

        self.decoder = keras.Sequential([
            layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu"),
            layers.Dropout(0.1),
            layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu"),
            layers.Conv2D(OUTPUT_CHANNELS, 4, padding="same", activation="tanh")
        ])
        # self.train_ds = train_ds
        # self.test_ds = test_ds
        # self.generator = generator
        # lrelu = layers.LeakyReLU()
        # self.encoder = keras.Sequential([
        #     layers.Input(shape=(IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS)),
        #     layers.Conv2D(64, 4, strides=2, padding="same"),
        #     lrelu,
        #     layers.Dropout(0.5),
        #     layers.Conv2D(64, 4, strides=2, padding="same"),
        #     lrelu,
        #     layers.Dropout(0.5),
        #     layers.Conv2D(16, 4, strides=2, padding="same"),
        #     lrelu,
        #     layers.Flatten(),
        #     layers.Dense(1024),
        #     lrelu,
        #     # layers.Dropout(0.1),
        #     # layers.Dense(512),
        #     # lrelu
        # ])
        #
        # self.decoder = keras.Sequential([
        #     # layers.Dense(512),
        #     # lrelu,
        #     layers.Dense(1024),
        #     lrelu,
        #     layers.Dropout(0.1),
        #     layers.Reshape((8, 8, 16)),
        #     layers.Conv2DTranspose(16, 4, strides=2, padding="same"),
        #     lrelu,
        #     layers.Dropout(0.5),
        #     layers.Conv2DTranspose(64, 4, strides=2, padding="same"),
        #     lrelu,
        #     layers.Dropout(0.5),
        #     layers.Conv2DTranspose(64, 4, strides=2, padding="same"),
        #     lrelu,
        #     layers.Conv2D(OUTPUT_CHANNELS, 4, padding="same", activation="tanh")
        # ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
