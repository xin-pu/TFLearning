import tensorflow as tf
from tensorflow.keras.layers import *

# 下采样
from tensorflow.python.keras.regularizers import l2


def dense_block(units):
    return Dense(units=units,
                 kernel_initializer=tf.keras.initializers.truncated_normal(0.0, 0.01),
                 # kernel_regularizer=l2(5e-4),
                 activation=LeakyReLU(0.1),
                 trainable=True)


def CBL(filters):
    blk = tf.keras.Sequential()
    blk.add(Conv2D(filters=filters, kernel_size=3, kernel_regularizer=l2(5e-4), padding='same'))
    blk.add(BatchNormalization())
    blk.add(LeakyReLU(alpha=0.1))
    blk.add(MaxPool2D())
    return blk


class GraveModel(tf.keras.Model):
    drop_rate = 0.5

    def get_config(self):
        return dict()

    def __init__(self):
        super().__init__()
        self.clb1 = CBL(32)
        self.clb2 = CBL(16)
        self.clb3 = CBL(6)
        self.clb4 = CBL(4)
        self.clb5 = CBL(2)
        # self.clb6 = CBL(2)

        self.flatten = Flatten()
        # self.dropout1 = Dropout(rate=self.drop_rate)
        self.dense1 = Dense(units=32, activation=tf.nn.relu, kernel_regularizer=l2(5e-4))
        # self.dropout2 = Dropout(rate=self.drop_rate)
        self.dense2 = Dense(units=2)

    def call(self, inputs, **kwargs):
        x = self.clb1(inputs)
        x = self.clb2(x)
        x = self.clb3(x)
        x = self.clb4(x)
        x = self.clb5(x)
        # x = self.clb6(x)

        x = self.flatten(x)
        # x = self.dropout1(x)
        x = self.dense1(x)
        # x = self.dropout2(x)
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    test_x = tf.ones([1, 416, 416, 1])
    model = GraveModel()
    test_y = model(test_x)
    model.summary()
    print(test_y.shape)
