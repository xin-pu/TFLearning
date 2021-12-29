import tensorflow as tf

from ActualTask.ModuleGrabPoint.Nets.model import GraveModel
from ActualTask.ModuleGrabPoint.Nets.resnet import get_net

model_dir = r'D:\DataSets\GrabModule\Model'

model = get_net((416, 416), 2)

model.load_weights(model_dir)
test_x = tf.ones([1, 416, 416, 1])
model(test_x)
model.save(model_dir)
