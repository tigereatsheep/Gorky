# coding: utf-8
import numpy as np


class SoftmaxEntorpy(object):

    def __init__(self):
        pass

    def feed(self, x, y):
        x -= np.expand_dims(np.max(x, axis=2), axis=-1)
        expx = np.exp(x)
        normlized_exps = expx / np.expand_dims(np.sum(expx, axis=2), axis=-1)
        loss = -np.sum(y * np.log(normlized_exps), axis=2)
        error = normlized_exps - y
        print (error[0])
        return loss, error


class MeanSquareError(object):

    def __init__(self):
        pass

    def feed(self, x, y):
        loss = np.linalg.norm(x - y, axis=2)
        error = 2 * (x - y)
        return loss, error