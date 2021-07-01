import tensorflow as tf


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten 层将除第一维以为的维度展平
        self.flatten = tf.keras.layers.Flatten()

        # 创建100个神经元的线性层
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        # 创建10个神经元的输出层
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = tf.nn.softmax(x)
        return output