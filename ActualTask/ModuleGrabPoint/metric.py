import tensorflow as tf
from tensorflow.keras import backend as kb


class Metric(tf.keras.metrics.Metric):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.max_error = 0

    def update_state(self, y_trues, y_pres, sample_weight=None):
        self.max_error = kb.max(kb.abs(y_trues - y_pres))

    def result(self):
        return self.max_error
