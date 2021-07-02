import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

X = tf.constant([2013, 2014, 2015, 2016, 2017])
Y = tf.constant([12000, 14000, 15000, 16500, 17500])

dataset = tf.data.Dataset.from_tensor_slices((X, Y))

for x, y in dataset:
    print(x.numpy(), y.numpy())

(trainData, trainLabel), (_, _) = tf.keras.datasets.mnist.load_data()

trainData = np.expand_dims(trainData.astype(np.float32) / 255.0, axis=-1)
mnist_dataset = tf.data.Dataset.from_tensor_slices((trainData, trainLabel))

for image, label in mnist_dataset:
    plt.title(label.numpy())
    plt.imshow(image.numpy()[:, :, 0])
    plt.show()
    break


# Map
def rot90(ima, lab):
    ima = tf.image.rot90(ima)
    return ima, lab


mnist_dataset = mnist_dataset.map(rot90)

# Shuffle

# Batch
mnist_dataset = mnist_dataset.batch(4)
for images, labels in mnist_dataset:
    fig, axs = plt.subplots(1, 4)
    for i in range(4):
        axs[i].set_title(labels.numpy()[i])
        axs[i].imshow(images.numpy()[i, :, :, 0])
    plt.show()
    break;

# Repeat

# Reduce

# Take
