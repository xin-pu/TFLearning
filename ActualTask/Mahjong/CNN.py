import tensorflow as tf


class CNN(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],
            strides=3)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],
            strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=7)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        output = tf.nn.softmax(x)
        return output

    def get_config(self):
        return dict()
