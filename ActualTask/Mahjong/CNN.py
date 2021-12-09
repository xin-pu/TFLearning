import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.layers.normalization import *


class CNN(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            kernel_regularizer=l2(5e-4))
        self.batch1 = BatchNormalization()
        self.leakyRelu1 = LeakyReLU(alpha=0.1)

        self.conv2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=[3, 3],
            padding='same',
            kernel_regularizer=l2(5e-4))
        self.batch2 = BatchNormalization()
        self.leakyRelu2 = LeakyReLU(alpha=0.1)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu, kernel_regularizer=l2(5e-4))
        self.dense2 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_regularizer=l2(5e-4))
        self.dense3 = tf.keras.layers.Dense(units=7)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.batch1(x)
        x = self.leakyRelu1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.leakyRelu2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        output = tf.nn.softmax(x)
        return output

    def get_config(self):
        return dict()


if __name__ == "__main__":
    test_x = tf.ones([1, 64, 64, 1])
    model = CNN()
    test_y = model(test_x)
    print(test_y)
