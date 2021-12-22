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


def CBLM(filters):
    blk = tf.keras.Sequential()
    blk.add(Conv2D(filters=filters,
                   kernel_size=3,
                   kernel_regularizer=l2(5e-4),
                   padding='same'))
    blk.add(BatchNormalization())
    blk.add(LeakyReLU(alpha=0.1))
    blk.add(MaxPool2D())
    return blk


def CBL(filters):
    blk = tf.keras.Sequential()
    blk.add(Conv2D(filters=filters,
                   kernel_size=3,
                   kernel_regularizer=l2(5e-4),
                   padding='same'))
    blk.add(BatchNormalization())
    blk.add(LeakyReLU(alpha=0.1))
    return blk


class GraveModel(tf.keras.Model):
    drop_rate = 0.5

    def get_config(self):
        return dict()

    def __init__(self):
        super().__init__()

        self.cblm_1 = CBLM(32)

        self.cbl_2 = CBL(16)
        self.cblm_2 = CBLM(16)

        self.cbl_3 = CBL(8)
        self.cblm_3 = CBLM(8)

        self.cbl_4 = CBL(4)
        self.cblm_4 = CBLM(4)

        self.cblm_5 = CBLM(2)
        self.cblm_6 = CBLM(2)

        self.flatten = Flatten()

        self.dense1 = Dense(units=32,
                            activation=tf.nn.leaky_relu,
                            kernel_regularizer=l2(5e-4))

        self.dense2 = Dense(units=8,
                            activation=tf.nn.leaky_relu,
                            kernel_regularizer=l2(5e-4))

        self.dense_end = Dense(units=2)

    def call(self, inputs, **kwargs):
        x = self.cblm_1(inputs)

        x = self.cbl_2(x)
        x = self.cblm_2(x)

        x = self.cbl_3(x)
        x = self.cblm_3(x)

        x = self.cbl_4(x)
        x = self.cblm_4(x)

        x = self.cblm_5(x)
        x = self.cblm_6(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense_end(x)
        return x


if __name__ == "__main__":
    test_x = tf.ones([1, 416, 416, 1])
    model = GraveModel()
    test_y = model(test_x)
    model.summary()
    print(test_y.shape)
