import tensorflow as tf

# 定义2X2 常量矩阵
A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)

# 定义随机数
B = tf.random.uniform(shape=(), dtype=tf.float32)

# 定义两个元素的零向量
C = tf.zeros(shape=2)
D = tf.zeros(shape=(2, 2))

print(A.numpy())
print(B.numpy())
print(B)
print(C)
print(D)

A1 = tf.add(A, D)
A2 = tf.matmul(A, D)

print(A1)
print(A2)
