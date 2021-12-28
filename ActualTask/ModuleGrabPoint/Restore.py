import tensorflow as tf

from ActualTask.ModuleGrabPoint.model import GraveModel

model_dir = r'D:\DataSets\GrabModule\Model'

model = GraveModel()

model.load_weights(model_dir)
test_x = tf.ones([1, 416, 416, 1])
model(test_x)
model.save(model_dir)
