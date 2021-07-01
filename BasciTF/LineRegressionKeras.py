import matplotlib.pyplot as plt
import tensorflow as tf


class Linear(tf.keras.Model):

    def get_config(self):
        return [self.W, self.B]

    def call(self, x, training=None, mask=None):
        return self.W * x + self.b

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W = tf.Variable(tf.random.uniform([1]))  # 随机初始化参数
        self.b = tf.Variable(tf.random.uniform([1]))


TRUE_W = 3.0
TRUE_b = 2.0
NUM_SAMPLES = 100

# 初始化随机数据
X = tf.random.normal(shape=[NUM_SAMPLES]).numpy()
noise = tf.random.normal(shape=[NUM_SAMPLES]).numpy()
y = X * TRUE_W + TRUE_b + noise * 0.5  # 添加噪声

model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(1000):
    with tf.GradientTape() as tape:
        y_estimate = model(X)
        loss = tf.reduce_mean(tf.square(y_estimate - y))

    grads = tape.gradient(loss, model.variables)

    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

plt.scatter(X, y, c='g')
plt.plot(X, model(X), c='r')
plt.show()
