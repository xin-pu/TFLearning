import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import *


anchors = np.array([[10, 13], [16, 30], [33, 23],
                    [30, 61], [62, 45], [59, 119],
                    [116, 90], [156, 198], [373, 326]])

# 勒边预测层


def class_predictor(num_anchors, num_class):
    return Conv2D(num_anchors * (num_class + 1), kernel_size=[3, 3], strides=1, padding='same')


# 边界框预测层
def box_predictor(num_anchors):
    return Conv2D(num_anchors * 4, kernel_size=[3, 3], strides=1, padding='same')


# 下采样
def down_sample(num_filer, stride=1):
    blk = tf.keras.Sequential()
    for i in range(2):
        blk.add(Conv2D(num_filer, kernel_size=3, strides=stride, padding='same'))
        blk.add(BatchNormalization())
        blk.add(LeakyReLU(alpha=0.1))
    blk.add(MaxPool2D())
    return blk


# 主干
def baseNet(number_filters=(16, 32, 64)):
    blk = tf.keras.Sequential()
    for num_filter in number_filters:
        blk.add(down_sample(num_filter))
    return blk


class TinySSD(tf.keras.Model):
    def __init__(self, classes=20, num_anchors=4):
        super().__init__()
        self.classes = classes
        self.anchors = num_anchors

        self.base_net = baseNet()
        self.class_predictor_32 = class_predictor(self.anchors, self.classes)
        self.box_predictor_32 = box_predictor(self.anchors)

        self.ds_16 = down_sample(128)
        self.class_predictor_16 = class_predictor(self.anchors, self.classes)
        self.box_predictor_16 = box_predictor(self.anchors)

        self.ds_8 = down_sample(128)
        self.class_predictor_8 = class_predictor(self.anchors, self.classes)
        self.box_predictor_8 = box_predictor(self.anchors)

        self.ds_4 = down_sample(128)
        self.class_predictor_4 = class_predictor(self.anchors, self.classes)
        self.box_predictor_4 = box_predictor(self.anchors)

        self.ds_2 =down_sample(128)
        self.class_predictor_1 = class_predictor(self.anchors, self.classes)
        self.box_predictor_1 = box_predictor(self.anchors)

    def call(self, inputs, training=None, mask=None):
        X = []
        x = self.base_net(inputs)
        X.append(x)
        y_class_32 = self.class_predictor_32(x)
        y_box_32 = self.box_predictor_32(x)

        x = self.ds_16(x)
        X.append(x)
        y_class_16 = self.class_predictor_16(x)
        y_box_16 = self.box_predictor_16(x)

        x = self.ds_8(x)
        X.append(x)
        y_class_8 = self.class_predictor_8(x)
        y_box_8 = self.box_predictor_8(x)

        x = self.ds_4(x)
        X.append(x)
        y_class_4 = self.class_predictor_4(x)
        y_box_4 = self.box_predictor_4(x)

        x = self.ds_2(x)
        X.append(x)
        y_class_1 = self.class_predictor_1(x)
        y_box_1 = self.box_predictor_1(x)

        all_class_preds = (y_class_32, y_class_16, y_class_8, y_class_4, y_class_1)
        all_box_preds = (y_box_32, y_box_16, y_box_8, y_box_4, y_box_1)

        return X, self.concat_class_preds(all_class_preds, self.classes), self.concat_box_preds(all_box_preds)

    def get_config(self):
        return dict()

    @staticmethod
    def concat_class_preds(all_class_preds, classes):
        d = []
        for c in all_class_preds:
            d.append(tf.reshape(c, shape=(c.shape[0], -1, classes + 1)))
        return tf.concat(d, axis=1)

    @staticmethod
    def concat_box_preds(all_box_preds):
        d = []
        for c in all_box_preds:
            d.append(tf.reshape(c, shape=(c.shape[0], -1)))
        return tf.concat(d, axis=1)


class SSDLoss(tf.keras.losses.Loss):
    bbox_mask = None

    def __init__(self, bbox_mask):
        super().__init__()
        self.bbox_mask = bbox_mask

    def call(self, y_true, y_pred):
        class_true, box_true = y_true
        class_pred, box_pred = y_pred
        cls = self.class_eval(class_pred, class_true)
        bbox = self.bbox_eval(box_pred, box_true, self.bbox_mask)
        return cls + bbox

    @staticmethod
    def class_eval(cls_pred, cls_true):
        return 1

    @staticmethod
    def bbox_eval(cls_pred, cls_true, bbox_mask):
        return 1


if __name__ == "__main__":
    # r = down_sample(10)(tf.ones([2, 20, 20, 3]))
    # print(r.shape)
    #
    # r = baseNet()(tf.ones([2, 256, 256, 3]))
    # print(r.shape)

    a = tf.ones([1, 416, 416, 3], dtype=float)
    xx, class_preds, box_preds = TinySSD(classes=1)(a)
    for x in xx:
        print(x.shape)
    print(class_preds.shape)
    print(box_preds.shape)

    y_true_test = [tf.ones([32, 5444, 2], dtype=float), tf.ones([32, 21776])]
    d = SSDLoss().call(y_true_test, [class_preds, box_preds])
    print(d)

    print(tf.argmax(class_preds, axis=-1) == 1)
