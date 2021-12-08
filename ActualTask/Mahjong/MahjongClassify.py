import tensorflow as tf
from tensorflow.python.keras.callbacks import *

from ActualTask.Mahjong.CNN import CNN
from ActualTask.Mahjong.MahjongDataset import MahjongDataSet

log_dir = '../Output/logs/'
model_dir = r'D:\DataSets\Mahjong\Model'

# Step 1, Prepare Data
dataset = MahjongDataSet()

num_train = len(dataset.train_dataset)
num_test = len(dataset.test_dataset)
print("Train num:{0}\tVal num:{1}".format(num_train, num_test))

# Step 2, Prepare Model
# model = tf.keras.applications.ResNet50(weights=None, classes=7, input_shape=(64, 64, 1))
model = CNN()

logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(model_dir,
                             monitor='val_sparse_categorical_accuracy',
                             save_weights_only=False,
                             save_best_only=False,
                             period=5)

reduce_lr = ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy',
                              factor=0.8,
                              patience=5,
                              verbose=1)

early_stopping = EarlyStopping(monitor='val_sparse_categorical_accuracy',
                               min_delta=0,
                               patience=6,
                               verbose=1)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=tf.keras.metrics.sparse_categorical_accuracy)

model.fit(
    dataset.train_dataset,
    validation_data=dataset.test_dataset,
    epochs=20,
    initial_epoch=0)

model.save(model_dir, overwrite=True, include_optimizer=True)
