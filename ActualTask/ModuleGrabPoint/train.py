import tensorflow as tf
from tensorflow.python.keras.callbacks import *

from ActualTask.ModuleGrabPoint.dataset import GrabDataSet
from ActualTask.ModuleGrabPoint.loss import Loss
from ActualTask.ModuleGrabPoint.metric import Metric
from ActualTask.ModuleGrabPoint.model import GraveModel

init_learning_rate = 1E-4
size_pattern = 416
batch_size = 2
model_dir = r'D:\DataSets\GrabModule\Model'

# Step 1, Prepare Data


train_data = GrabDataSet(r'D:\DataSets\train\info.csv',
                         batch_size=4,
                         val_per=0).train_dataset

val_data = GrabDataSet(r'D:\DataSets\val\info.csv',
                       batch_size=4,
                       val_per=0).train_dataset

num_train = len(train_data)
num_test = len(val_data)
print("Train num:{0}\tVal num:{1}".format(num_train, num_test))
print("Start learning Rate:{0}".format(init_learning_rate))

# Step 2, Prepare Model
model = GraveModel()
model.load_weights(model_dir)
checkpoint = ModelCheckpoint(model_dir,
                             monitor='val_metric',
                             save_weights_only=True,
                             save_best_only=True,
                             save_freq='epoch',
                             verbose=1,
                             mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_metric',
                              factor=0.5,
                              patience=8,
                              verbose=1,
                              mode='min')


def learning_rate_scheduler(epoch):
    epoch_by50 = epoch // 100
    learning_rate = 10 ** (-3 - epoch_by50)
    print("Current Learning Rate:", learning_rate)
    return learning_rate


model_learning_rate_callback = LearningRateScheduler(learning_rate_scheduler)

loss = Loss()
metric = Metric()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init_learning_rate, amsgrad=True),
              loss=loss,
              metrics=[tf.keras.metrics.mean_absolute_error,
                       metric])

model.fit(
    train_data,
    validation_data=val_data,
    epochs=500,
    initial_epoch=0,
    callbacks=[reduce_lr, checkpoint])

model.save(model_dir, overwrite=True, include_optimizer=False)
