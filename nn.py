# coding: utf-8
import numpy as np


class Module(object):

    def __init__(self, seq=None, loss_function=None):
        if seq:
            self.seq = seq
            self.rseq = reversed(seq)
        if loss_function:
            self.loss_function = loss_function

    def forward(self, input):
        for layer in self.seq:
            input = layer.forward(input)
        return input
    
    def backward(self, gradient):
        for layer in self.rseq:
            gradient = layer.backward(gradient)
        return gradient
    
    def feed(self, xs, ys):
        # Shape: batch_size x ...
        total_loss = 0.0
        gradients = []
        for i in range(xs.shape[0]):
            f = self.forward(xs[i])
            loss, gradient = self.loss_function.feed(f, ys[i])
            total_loss += loss
            gradients.append(gradient)
        return total_loss, gradients
    
    def train(self, xs, ys, learning_rate):
        # Shape: batch_size x ...
        total_loss, gradients = self.feed(xs, ys)
        for gradient in gradients:
            for layer in self.rseq:
                gradient = layer.backward(gradient)
        for layer in self.seq:
            if layer.trainable:
                gradients = np.array(layer.gradients)
                layer.gradients = []
                layer.weights -= learning_rate * np.sum(gradients, axis=0)
        return total_loss
        