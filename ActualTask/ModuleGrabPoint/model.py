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
    blk.add(Conv2D(1, kernel_size=1, padding='valid'))
    blk.add(CBL(num_filters, 3, strides=(2, 2)))
    return blk


class GraveModel(tf.keras.Model):
    drop_rate = 0.5

    def get_config(self):
        return dict()

    def __init__(self):
        super().__init__()
        self.pcbl1 = PCBL(32)
        self.pcbl2 = PCBL(16)
        self.pcbl3 = PCBL(4)
        self.pcbl4 = PCBL(2)
        self.flatten = Flatten()
        self.dense1 = Dense(units=32, activation=tf.nn.relu, kernel_regularizer=l2(5e-4))
        self.dense_end = Dense(units=2)

    def call(self, inputs, **kwargs):
        x = self.pcbl1(inputs)
        x = self.pcbl2(x)
        x = self.pcbl3(x)
        x = self.pcbl4(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense_end(x)

        return x


if __name__ == "__main__":
    test_x = tf.ones([1, 416, 416, 1])
    model = GraveModel()
    test_y = model(test_x)
    model.summary()
    print(test_y.shape)
    print(test_y)

    checkpoint = tf.train.Checkpoint(myModel=model)
    checkpoint.restore(tf.train.latest_checkpoint(r'D:\DataSets\GrabModule\ResModel2'))
    test_y = model(test_x)
    print(test_y)
