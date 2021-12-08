import os

import tensorflow as tf


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


class MahjongDataSet:
    batch_size = 24
    resize_shape = (64, 64)
    train_folder = r"D:\DataSets\Mahjong\train-all"

    def __init__(self,
                 batch_size=24,
                 resize_shape=(64, 64),
                 val_per=0.2):
        self.batch_size = batch_size
        self.resize_shape = resize_shape

        train_dataset = self.get_data(self.train_folder)
        data_count = len(train_dataset)
        split_size = int(data_count * val_per)
        self.train_dataset = train_dataset.skip(split_size)
        self.test_dataset = train_dataset.take(split_size)

    def get_data(self, data_folder):
        png_files, labels = get_dataset_list(data_folder)

        train_dataset = tf.data.Dataset.from_tensor_slices((png_files, labels))
        train_dataset = train_dataset.map(
            map_func=self.decode_and_resize,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(buffer_size=1024)
        train_dataset = train_dataset.batch(batch_size=self.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset

    # Singe Image Read as Tensor to Resize to 64*64
    def decode_and_resize(self, filename, label, enhance=True):
        img = tf.io.read_file(filename, 'r')
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(tf.image.resize(img, self.resize_shape), dtype=tf.float32)
        if enhance:
            img = tf.image.random_brightness(img, max_delta=50.)
            img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
            img = tf.image.random_hue(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        img = tf.clip_by_value(img, 0, 255)
        img /= 255
        return img, label


if __name__ == "__main__":
    dataset = MahjongDataSet()
    print(dataset.test_dataset)
    print(len(dataset.train_dataset))
