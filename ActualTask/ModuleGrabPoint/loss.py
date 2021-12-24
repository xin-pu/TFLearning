import tensorflow as tf
from tensorflow.keras import backend as kb


class Loss(tf.keras.losses.Loss):
    def __init__(self, name=None, reduction=None):
        super().__init__(name=name)

    @tf.autograph.experimental.do_not_convert
    def call(self, y_trues, y_pres):
        return kb.sum(kb.abs(y_trues - y_pres))


class Metric(tf.keras.metrics.Metric):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.max_error = 0

    def update_state(self, y_trues, y_pres, sample_weight=None):
        self.max_error = kb.max(kb.abs(y_trues - y_pres))

    def result(self):
        return self.max_error
