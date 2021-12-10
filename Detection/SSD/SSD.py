import tensorflow as tf
from tensorflow.python.keras.layers import *


# 勒边预测层


def class_predictor(num_anchors, num_class):
    return Conv2D(num_anchors * (num_class + 1), kernel_size=[3, 3], padding='same')


# 边界框预测层
def box_predictor(num_anchors):
    return Conv2D(num_anchors * 4, kernel_size=[3, 3], strides=1, padding='same')


def down_sample(num_filer):
    blk = tf.keras.Sequential()
    for i in range(2):
        blk.add(Conv2D(num_filer, kernel_size=3, padding='same'))
        blk.add(BatchNormalization())
        blk.add(LeakyReLU(alpha=0.1))
    blk.add(MaxPool2D())
    return blk


def baseNet(number_filters=(16, 32, 64)):
    blk = tf.keras.Sequential()
    for num_filter in number_filters:
        blk.add(down_sample(num_filter))
    return blk


class TinySSD(tf.keras.Model):
    def __init__(self, classes=20):
        super().__init__()
        self.classes = classes

        self.base_net = baseNet()
        self.class_predictor_32 = class_predictor(32, self.classes)
        self.box_predictor_32 = box_predictor(32)

        self.ds_16 = down_sample(128)
        self.class_predictor_16 = class_predictor(16, self.classes)
        self.box_predictor_16 = box_predictor(16)

        self.ds_8 = down_sample(128)
        self.class_predictor_8 = class_predictor(8, self.classes)
        self.box_predictor_8 = box_predictor(8)

        self.ds_4 = down_sample(128)
        self.class_predictor_4 = class_predictor(4, self.classes)
        self.box_predictor_4 = box_predictor(4)

        self.ds_1 = MaxPool2D(pool_size=4)
        self.class_predictor_1 = class_predictor(1, self.classes)
        self.box_predictor_1 = box_predictor(1)
        super().__init__()

    def call(self, inputs, training=None, mask=None):
        x = self.base_net(inputs)

        anchors = []
        box = []
        y_class_32 = self.class_predictor_32(x)
        y_box_32 = self.box_predictor_32(x)
        anchors.append(y_class_32)
        box.append(y_box_32)

        x = self.ds_16(x)
        y_class_16 = self.class_predictor_16(x)
        y_box_16 = self.box_predictor_16(x)
        anchors.append(y_class_16)
        box.append(y_box_16)

        x = self.ds_8(x)
        y_class_8 = self.class_predictor_8(x)
        y_box_8 = self.box_predictor_8(x)
        anchors.append(y_class_8)
        box.append(y_box_8)

        x = self.ds_4(x)
        y_class_4 = self.class_predictor_4(x)
        y_box_4 = self.box_predictor_4(x)
        anchors.append(y_class_4)
        box.append(y_box_4)

        x = self.ds_1(x)
        y_class_1 = self.class_predictor_1(x)
        y_box_1 = self.box_predictor_1(x)
        anchors.append(y_class_1)
        box.append(y_box_1)

        return anchors, box

    def get_config(self):
        return dict()


if __name__ == "__main__":
    a = tf.ones([1, 256, 256, 3], dtype=float)
    anchors, box = TinySSD()(a)
    for a in anchors:
        print(a.shape)
