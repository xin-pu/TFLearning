import tensorflow as tf
from tensorflow.keras.layers import *

# 下采样
from tensorflow.python.keras.regularizers import l2


def Conv(*args, **kwargs):
    new_kwargs = {'kernel_regularizer': l2(5e-4),
                  'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    new_kwargs.update(kwargs)

    return Conv2D(*args, **new_kwargs)


def CBL(*args, **kwargs):
    new_kwargs = {'use_bias': False}
    new_kwargs.update(kwargs)
    blk = tf.keras.Sequential()
    blk.add(Conv(*args, **new_kwargs))
    blk.add(BatchNormalization())
    blk.add(LeakyReLU(alpha=0.1))
    return blk


def PCBL(num_filters):
    blk = tf.keras.Sequential()
    blk.add(ZeroPadding2D(((1, 0), (1, 0))))
    blk.add(CBL(num_filters, 3, strides=(2, 2)))
    return blk


class GraveModel(tf.keras.Model):
    drop_rate = 0.5

    def get_config(self):
        return dict()

    def __init__(self):
        super().__init__()

        self.cbl_10 = CBL(32 * 2, 3)
        self.cbl_11 = CBL(32, 1)
        self.cbl_12 = CBL(32 * 2, 3)
        self.add_1 = Add()
        self.poo_1 = MaxPooling2D(pool_size=4)

        self.cbl_20 = CBL(8 * 2, 3)
        self.cbl_21 = CBL(8, 1)
        self.cbl_22 = CBL(8 * 2, 3)
        self.add_2 = Add()
        self.poo_2 = MaxPooling2D(pool_size=4)

        self.cbl_30 = CBL(4 * 2, 3)
        self.cbl_31 = CBL(4, 1)
        self.cbl_32 = CBL(4 * 2, 3)
        self.add_3 = Add()
        self.poo_3 = MaxPooling2D(pool_size=4)

        self.cbl_final = CBL(2, 3)

        self.flatten = Flatten()

        self.dense_end = Dense(units=2)

    def call(self, inputs, **kwargs):
        x = self.cbl_10(inputs)
        y = self.cbl_11(x)
        y = self.cbl_12(y)
        x = self.add_1([x, y])
        x = self.poo_1(x)

        x = self.cbl_20(x)
        y = self.cbl_21(x)
        y = self.cbl_22(y)
        x = self.add_2([x, y])
        x = self.poo_2(x)

        x = self.cbl_30(x)
        y = self.cbl_31(x)
        y = self.cbl_32(y)
        x = self.add_3([x, y])
        x = self.poo_3(x)

        x = self.cbl_final(x)

        x = self.flatten(x)
        x = self.dense_end(x)

        return x


if __name__ == "__main__":
    test_x = tf.ones([1, 416, 416, 1])
    model = GraveModel()
    test_y = model(test_x)
    model.summary()
    print(test_y.shape)
    print(test_y)
