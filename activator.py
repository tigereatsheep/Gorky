# coding: utf-8
import numpy as np
from base_layer import BaseLayer


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ReLU(BaseLayer):

    def __init__(self):
        super(ReLU, self).__init__(trainable=False)

    def forward(self, input):
        self.input = input
        input[input < 0.0] = 0.0
        return input

    def backward(self, gradient):
        mask = (self.input > 0.0).astype(np.float)
        return gradient * mask
    

class Sigmoid(BaseLayer):

    def __init__(self):
        super(Sigmoid, self).__init__(trainable=False)
    
    def forward(self, input):
        self.input = input
        return sigmoid(input)

    def backward(self, gradient):
        return sigmoid(self.input) * \
            (1 - sigmoid(self.input)) * gradient