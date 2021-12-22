import tensorflow as tf
from tensorflow.keras import backend as kb


class Loss(tf.keras.losses.Loss):
    def __init__(self, name=None, reduction=None):
        super().__init__(name=name)

    @tf.autograph.experimental.do_not_convert
    def call(self, y_trues, y_pres):
        return kb.sum(kb.abs(y_trues - y_pres))
