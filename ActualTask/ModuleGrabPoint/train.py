import tensorflow as tf
from tensorflow.python.keras.callbacks import *

from ActualTask.ModuleGrabPoint.dataset import GrabDataSet
from ActualTask.ModuleGrabPoint.loss import Loss, Metric
from ActualTask.ModuleGrabPoint.model import GraveModel

init_learning_rate = 1E-3
size_pattern = 416
batch_size = 2
model_dir = r'D:\DataSets\GrabModule\ResModel2'

# Step 1, Prepare Data
train_dataset = GrabDataSet(batch_size=batch_size,
                            val_per=0.4)

num_train = len(train_dataset.train_dataset)
num_test = len(train_dataset.test_dataset)
print("Train num:{0}\tVal num:{1}".format(num_train, num_test))
print("Start learning Rate:{0}".format(init_learning_rate))

# Step 2, Prepare Model
model = GraveModel()

checkpoint = ModelCheckpoint(model_dir,
                             monitor='val_metric',
                             save_weights_only=True,
                             save_best_only=True,
                             save_freq='epoch',
                             verbose=1,
                             mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_metric',
                              factor=0.5,
                              patience=5,
                              verbose=1,
                              mode='min')

early_stopping = EarlyStopping(monitor='val_metric',
                               patience=20,
                               verbose=1,
                               mode='min')

# model.load_weights(model_dir)
# test_x = tf.ones([1, 416, 416, 1])
# model(test_x)
# model.save(model_dir)

loss = Loss()
metric = Metric()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init_learning_rate, amsgrad=True),
              loss=loss,
              metrics=[tf.keras.metrics.mean_absolute_error,
                       metric])

model.fit(
    train_dataset.train_dataset,
    validation_data=train_dataset.test_dataset,
    epochs=100,
    initial_epoch=0,
    callbacks=[reduce_lr, checkpoint, early_stopping])

model.save(model_dir, overwrite=True, include_optimizer=False)
