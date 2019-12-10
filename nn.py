# coding: utf-8
import numpy as np


class Module(object):

    def __init__(self, seq=None, loss_function=None):
        if seq:
            self.seq = seq
            self.rseq = list(reversed(self.seq))
        if loss_function:
            self.loss_function = loss_function

    def forward(self, input):
        for layer in self.seq:
            input = layer.forward(input)
        return input
    
    def backward(self, gradient):
        for layer in self.rseq:
            gradient = layer.backward(gradient)
    
    def train(self, xs, ys, learning_rate):
        # Shape: batch_size x ...
        f = self.forward(xs)    
        loss, gradient = self.loss_function.feed(f, ys)
        self.backward(gradient)
        for layer in self.seq:
            if layer.trainable:
                print(layer.weights[0, 0])
                layer.weights = layer.weights - learning_rate * np.sum(layer.gradients, axis=0)
        print("loss: \n", np.sum(loss))

