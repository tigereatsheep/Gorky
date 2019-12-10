# coding: utf-8
from layers import (Convolution, FullyConnect, Flatten)
from activator import ReLU, Sigmoid
from pool import MaxPool
from nn import Module
from temp_data_utils import Minist
from loss_function import SoftmaxEntorpy, MeanSquareError


batch_size = 16
learning_rate = 1e-3
turns = 1000


l = [Convolution((16, 1, 5, 5)), MaxPool(), ReLU(),
     Convolution((32, 16, 3, 3)), MaxPool(), ReLU(),
     Convolution((32, 32, 3, 3)), ReLU(), Flatten(), FullyConnect((288, 10))]


minist = Minist()
module = Module(l, SoftmaxEntorpy())


for i in range(turns):
    datas, labels = minist.next_batch(batch_size)
    datas = datas / 255.0
    module.train(datas, labels, learning_rate)