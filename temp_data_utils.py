import os
import struct
import numpy as np


dataset_dir = './dataset'
train_image_file = os.path.join(dataset_dir, 'train-images.idx3-ubyte')
train_label_file = os.path.join(dataset_dir, 'train-labels.idx1-ubyte')


nimages = 60000
nclasses = 10
h, w = 28, 28


def onehot(n, d):
    c = np.zeros(d)
    c[n] = 1
    return c


def get_train_data():
    data = np.zeros((nimages, 1, 28, 28))
    with open(train_image_file, 'rb') as f:
        f.seek(16)
        for i in range(nimages):
            for j in range(h):
                data[i, 0, j, :] = np.array(
                struct.unpack('<' + 28 * 'B', f.read(28))
                )
    return data


def get_train_label():
    labels = np.zeros((nimages, 1, nclasses))
    with open(train_label_file, 'rb') as f:
        f.seek(8)
        for i in range(nimages):
            labels[i] = onehot(struct.unpack('<B', f.read(1)), nclasses)
    return labels


def shuffle(n):
    r = np.arange(n)
    np.random.shuffle(r)
    return r


class Minist(object):

    def __init__(self):
        self.datas = get_train_data()
        self.labels = get_train_label()
        self.remains = shuffle(nimages)

    def next_batch(self, n):
        print("remains: ", self.remains.shape[0])
        if self.remains.shape[0] < n:
            self.remains = shuffle(nimages)
        indexes = self.remains[:n]
        self.remains = np.delete(self.remains, range(n))
        return self.datas[indexes], self.labels[indexes]


if __name__ == '__main__':
    pass