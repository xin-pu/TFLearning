import os

import numpy as np
import tensorflow as tf

from ActualTask.Mahjong.CNN import CNN


def get_dataset_list(folder):
    filenames, labels = [], []
    sub_dirs = [r[0] for r in os.walk(folder)]
    sub_dirs.remove(sub_dirs[0])
    for dire in sub_dirs:
        label = int(dire.split("\\")[-1])
        files = [r[2] for r in os.walk(dire)][0]
        for file in files:
            file_fullname = os.path.join(dire, file)
            filenames.append(file_fullname)
            labels.append(label)
    return filenames, labels


size_pattern = 64

model_dir = r'D:\DataSets\Mahjong\Model'
model = CNN()

model.load_weights(model_dir)

err = 0
index = 0
img_files, lab = get_dataset_list(r"D:\DataSets\Mahjong\val")
for img_file in img_files:
    img = tf.io.read_file(img_file, 'r')
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.cast(tf.image.resize(img, (size_pattern, size_pattern)), dtype=tf.float32)
    img /= 255.
    img = tf.expand_dims(img, 0)
    res = model.predict(img)
    pre = np.argmax(res[0]) + 1
    if pre != lab[index]:
        err += 1
        print(img_file, res[0], np.argmax(res[0]) + 1)
    index += 1


print("error", err)
