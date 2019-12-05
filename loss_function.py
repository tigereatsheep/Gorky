# coding: utf-8
import numpy as np


class SoftmaxEntorpy(object):

    def __init__(self):
        pass

    def feed(self, x, y):
        x -= np.max(x)
        expx = np.exp(x)
        normlized_exps = expx / np.sum(expx)
        loss = -np.sum(y * np.log(normlized_exps))
        error = normlized_exps - y
        return loss, error
