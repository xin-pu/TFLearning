import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

TRUE_W = 3.0
TRUE_b = 2.0
NUM_SAMPLES = 100

# 初始化随机数据
X = tf.random.normal(shape=[NUM_SAMPLES]).numpy()
noise = tf.random.normal(shape=[NUM_SAMPLES]).numpy()
y = X * TRUE_W + TRUE_b + noise * 0.5  # 添加噪声

plt.scatter(X, y)

# 初始化模型参数
a = tf.Variable(initial_value=0., dtype=tf.float32)
b = tf.Variable(initial_value=0., dtype=tf.float32)

variables = [a, b]

num_epoch = 100
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

for e in range(num_epoch):
    with tf.GradientTape() as tape:
        _y = a * X + b
        loss = tf.reduce_mean(tf.square(_y - y))

    gards = tape.gradient(loss, variables)
    optimizer.apply_gradients(grads_and_vars=zip(gards, variables))

print(a.numpy(), b.numpy())
Y_p = a * X + b
plt.scatter(X, Y_p)
plt.show()
