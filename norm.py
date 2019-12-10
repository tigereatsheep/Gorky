# coding: utf-8
import numpy as np
from base_layer import BaseLayer


class BatchNorm(BaseLayer):

    def __init__(self, out_channels=None, batch_size=8):
        super(BatchNorm, self).__init__()
        if out_channels:
            # Alpha and Beta
            self.weights = np.ones((2, out_channels))
        self.batch_size = batch_size
        self.mean = None
        self.var = None
        self.inputs = []
        self.count = 0

    def forward(self, input):
        self.inputs.append(input)
        output = input * np.broadcast_to(
            self.weights[0, :], input.shape
            ) + np.broadcast_to(
            self.weights[1, :], input.shape
            )
        return output
    
    def backward(self, gradient):
        if self.count == self.batch_size:
            pass
        else:
            pass
