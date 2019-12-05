# coding: utf-8


class BaseLayer(object):

    def __init__(self, trainable=True):
        self.trainable = trainable
        if self.trainable:
            self.weights = None
            self.gradients = []

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
