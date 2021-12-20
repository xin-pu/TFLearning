import tensorflow as tf
from tensorflow.python.keras.callbacks import *

from ActualTask.Mahjong.CNN import CNN
from ActualTask.Mahjong.MahjongDataset import MahjongDataSet

init_learning_rate = 0.001
size_pattern = 64
batch_size = 32
log_dir = '../Output/logs/'
model_dir = r'D:\DataSets\Mahjong\Model'

# Step 1, Prepare Data
train_dataset = MahjongDataSet(resize_shape=(size_pattern, size_pattern), batch_size=batch_size, val_per=0,
                               train_folder=r"D:\DataSets\Mahjong\train")
val_dataset = MahjongDataSet(resize_shape=(size_pattern, size_pattern), batch_size=batch_size, val_per=0,
                             train_folder=r"D:\DataSets\Mahjong\val")

num_train = len(train_dataset.train_dataset)
num_test = len(val_dataset.train_dataset)
print("Train num:{0}\tVal num:{1}".format(num_train, num_test))

# Step 2, Prepare Model
# model = tf.keras.applications.ResNet50(
#     weights=None,
#     classes=7,
#     input_shape=(size_pattern, size_pattern, 1))
model = CNN()

logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(model_dir,

                             monitor='val_sparse_categorical_accuracy',
                             save_weights_only=False,
                             save_best_only=False,
                             period=5)

reduce_lr = ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy',
                              factor=0.5,
                              patience=2,
                              verbose=1)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=2,
                               verbose=1)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init_learning_rate, amsgrad=True),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

model.fit(
    train_dataset.train_dataset,
    validation_data=val_dataset.train_dataset,
    epochs=30,
    initial_epoch=0,
    callbacks=[reduce_lr, checkpoint, early_stopping])

model.save(model_dir, overwrite=True, include_optimizer=False)
