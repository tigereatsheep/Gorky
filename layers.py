# coding: utf-8
import numpy as np
from base_layer import BaseLayer
from operators import (naive_convolutional, naive_convolutional_deriviation,
                        conv_initializer, naive_initializer)


class Convolution(BaseLayer):

    def __init__(self, kernel_shape, pad=0, stride=1,
                    initializer=conv_initializer):
        super(Convolution, self).__init__()
        self.weights = initializer(kernel_shape)
        self.pad, self.stride = pad, stride
    
    def forward(self, input):
        self.input = input
        output, self.reshaped_input, self.reshaped_output = \
            naive_convolutional(input, self.weights, self.pad, self.stride)
        return output
    
    def backward(self, gradient):
        kernel_gradient, input_gradient = \
             naive_convolutional_deriviation(gradient, self.input, self.weights,
                                             self.reshaped_input, self.reshaped_output,
                                             self.pad, self.stride)
        if self.trainable:
            self.gradients = kernel_gradient
        return input_gradient


class FullyConnect(BaseLayer):

    def __init__(self, weights_shape, initializer=naive_initializer):
        super(FullyConnect, self).__init__()
        self.weights = initializer(weights_shape)

    def forward(self, input):
        self.input = input
        batch_size = input.shape[0]
        output = np.zeros((batch_size, 1, self.weights.shape[1]))
        for b in range(batch_size):
            output[b] = input[b].dot(self.weights)
        return output

    def backward(self, gradient):
        batch_size = gradient.shape[0]
        in_dim, out_dim = self.weights.shape
        self.gradients = np.zeros((batch_size, in_dim, out_dim))
        input_gradient = np.zeros((batch_size, 1, in_dim))
        for b in range(batch_size):
            self.gradients[b] = self.input[b].T.dot(gradient[b])
            input_gradient[b] = gradient[b].dot(self.weights[b].T)
        return input_gradient


class Flatten(BaseLayer):

    def __init__(self):
        super(Flatten, self).__init__(trainable=False)
        self.src_shape = None
    
    def forward(self, input):
        self.src_shape = input.shape
        input = input.reshape((input.shape[0], 1, -1))
        return input
    
    def backward(self, gradient):
        gradient = gradient.reshape(self.src_shape)
        return gradient