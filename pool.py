# coding: utf-8
from base_layer import BaseLayer
from operators import maxpool, maxpool_deriviation

class MaxPool(BaseLayer):

    def __init__(self, stride=2):
        super(MaxPool, self).__init__(trainable=False)
        self.stride = stride
    
    def forward(self, x):
        self.mask, output = maxpool(x, stride=self.stride)
        return output
    
    def backward(self, gradient):
        return maxpool_deriviation(gradient, self.mask, self.stride)


