import tensorflow as tf

from MLP import MLP
from TrainDatas.MNISTLoader import MINSTLoader

num_epochs = 5
batch_size = 50
learning_rate = 1E-3

# 定义训练模型
model = MLP()
# 加载数据集
dataLoader = MINSTLoader()
# 设计优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

num_batches = int(dataLoader.num_train_data // batch_size * num_epochs)

for batch_index in range(num_batches):
    X, Y = dataLoader.train_data, dataLoader.train_label
    with tf.GradientTape() as tape:
        Y_estimate = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(Y, Y_estimate)
        loss = tf.reduce_mean(loss)
        print("batch {}: loss {:2%}".format(batch_index, loss.numpy()))

    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

SparseCategoricalAccuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(dataLoader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_estimate = model.predict(dataLoader.test_data[start_index:end_index])
    SparseCategoricalAccuracy.update_state(y_true=dataLoader.test_label[start_index:end_index], y_pred=y_estimate)

print(SparseCategoricalAccuracy.result())
