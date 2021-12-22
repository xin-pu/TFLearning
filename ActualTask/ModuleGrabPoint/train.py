import tensorflow as tf
from tensorflow.python.keras.callbacks import *

from ActualTask.ModuleGrabPoint.dataset import GrabDataSet
from ActualTask.ModuleGrabPoint.loss import Loss
from ActualTask.ModuleGrabPoint.model import GraveModel

init_learning_rate = 0.01
size_pattern = 416
batch_size = 2
model_dir = r'D:\DataSets\GrabModule\ResModel'

# Step 1, Prepare Data
train_dataset = GrabDataSet(batch_size=batch_size,
                            val_per=0.2)

num_train = len(train_dataset.train_dataset)
num_test = len(train_dataset.test_dataset)
print("Train num:{0}\tVal num:{1}".format(num_train, num_test))

# Step 2, Prepare Model
model = GraveModel()

checkpoint = ModelCheckpoint(model_dir,
                             monitor='val_loss',
                             save_weights_only=False,
                             save_best_only=False,
                             period=5)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=2,
                              verbose=1)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=2,
                               verbose=1)

loss = Loss()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init_learning_rate, amsgrad=True),
              loss=tf.keras.losses.mean_squared_error,
              metrics=[tf.keras.metrics.mean_absolute_error])

model.fit(
    train_dataset.train_dataset,
    validation_data=train_dataset.test_dataset,
    epochs=20,
    initial_epoch=0,
    callbacks=[reduce_lr, checkpoint])

model.save(model_dir, overwrite=True, include_optimizer=False)
