import csv

import tensorflow as tf


def get_dataset_list():
    filenames, res = [], []
    f = open(r'D:\DataSets\GrabModule\info.csv', 'r')
    with f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            i = i + 1
            if i == 1:
                continue
            filenames.append(row[0])
            res.append([float(row[1]), float(row[2])])
    return filenames, res


def get_center(filename):
    pass


class GrabDataSet:
    batch_size = 24

    def __init__(self,
                 batch_size=24,
                 val_per=0.3):
        self.batch_size = batch_size
        train_dataset = self.get_data()

        data_count = len(train_dataset)
        print(data_count)
        split_size = int(data_count * val_per)
        self.train_dataset = train_dataset.skip(split_size)
        self.test_dataset = train_dataset.take(split_size)

    def get_data(self):
        png_files, labels = get_dataset_list()
        print(labels)
        train_dataset = tf.data.Dataset.from_tensor_slices((png_files, labels))
        train_dataset = train_dataset.map(
            map_func=self.decode_and_resize,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(buffer_size=1024)
        train_dataset = train_dataset.batch(batch_size=self.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset

    # Singe Image Read as Tensor to Resize to 64*64
    @staticmethod
    def decode_and_resize(filename, label, enhance=False):
        img = tf.io.read_file(filename, 'r')
        img = tf.io.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, [416, 416])
        img = img / 255.0
        return img, label[0]


if __name__ == "__main__":
    data = GrabDataSet(batch_size=12,
                       val_per=0.2)
    num_train = len(data.train_dataset)
    num_test = len(data.test_dataset)
    print("Train num:{0}\tVal num:{1}".format(num_train, num_test))
