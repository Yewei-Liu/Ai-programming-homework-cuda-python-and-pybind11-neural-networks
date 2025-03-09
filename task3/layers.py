import sys
sys.path.append('build')
import mytensor as mt
import numpy as np
import torch
from torch import nn

class MyLayer():
    def __init__(self, device="gpu"):
        self.device = device

    def forward(self):
        pass

    def backward(self):
        pass

    def parameter(self):
        return []
    
    def trained_parameter(self):
        return []

class MyRelu(MyLayer):
    def __init__(self, input: mt.Tensor, output: mt.Tensor, device="gpu", name=None):
        super().__init__(device)
        if name is None:
            self.name = 'relu'
        else:
            self.name = name
        self.input = input
        self.output = output
    
    def forward(self):
        return mt.relu_forward(self.input, self.output)
    
    def backward(self):
        return mt.relu_backward(self.input, self.output)
    
    def parameter(self):
        return [self.input]


class MySigmoid(MyLayer):
    def __init__(self, input: mt.Tensor, output: mt.Tensor, device="gpu", name=None):
        super().__init__(device)
        if name is None:
            self.name = 'sigmoid'
        else:
            self.name = name
        self.input = input
        self.output = output
    
    def forward(self):
        return mt.sigmoid_forward(self.input, self.output)
    
    def backward(self):
        return mt.sigmoid_backward(self.input, self.output)

    def parameter(self):
        return [self.input]


class MyLinear(MyLayer):
    def __init__(self, input: mt.Tensor, output: mt.Tensor, device="gpu", name=None):
        super().__init__(device)
        if name is None:
            self.name = 'linear'
        else:
            self.name = name
        self.input = input
        self.output = output
        C_in = self.input.shape()[1]
        C_out = self.output.shape()[1]
        tmp = nn.Linear(C_in, C_out)
        self.weight = mt.Tensor("weight", (C_in, C_out), self.device)
        self.weight.from_np(np.asfortranarray(tmp.weight.data).T)
        self.bias = mt.Tensor("bias", (1, C_out), self.device)
        self.bias.from_np(np.array(tmp.bias.data.reshape(1, -1)))
    
    def forward(self):
        return mt.fc_forward(self.input, self.weight, self.bias, self.output)
    
    def backward(self):
        return mt.fc_backward(self.input, self.weight, self.bias, self.output)
       
    def parameter(self):
        return [self.input, self.weight, self.bias]
    
    def trained_parameter(self):
        return [self.weight, self.bias]
    

class MyConv(MyLayer):
    def __init__(self, input: mt.Tensor, output: mt.Tensor, device="gpu", kernel_size=3, name=None):
        super().__init__(device)
        if name is None:
            self.name = 'conv'
        else:
            self.name = name
        self.input = input
        self.output = output
        N, C_in, H, W = self.input.shape()
        C_out = output.shape()[1]
        tmp = nn.Conv2d(C_in, C_out, 3, 1, padding='same')
        self.kernel = mt.Tensor("kernel", (C_out, C_in, kernel_size, kernel_size), self.device)
        self.kernel.from_np(np.asfortranarray(tmp.weight.data))
    
    def forward(self):
        return mt.conv_forward(self.input, self.kernel, self.output)
    
    def backward(self):
        return mt.conv_backward(self.input, self.kernel, self.output)
       
    def parameter(self):
        return [self.input, self.kernel]
    
    def trained_parameter(self):
        return [self.kernel]
    

class MyMaxPooling(MyLayer):
    def __init__(self, input: mt.Tensor, output: mt.Tensor, device="gpu", pool_size=2, stride=2, name=None):
        super().__init__(device)
        if name is None:
            self.name = 'maxpooling'
        else:
            self.name = name
        self.input = input
        self.output = output
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self):
        self.mask = mt.max_pooling_forward(self.input, self.output, self.pool_size, self.stride)
        return 
    
    def backward(self):
        return mt.max_pooling_backward(self.input, self.output, self.mask, self.pool_size, self.stride)
    
    def parameter(self):
        return [self.input]
       

class MyCrossEntropyLoss(MyLayer):
    def __init__(self, input: mt.Tensor, output: mt.Tensor, label: mt.Tensor, device="gpu", name=None):
        super().__init__(device)
        if name is None:
            self.name = 'crossentropyloss'
        else:
            self.name = name
        self.input = input
        self.output = output
        self.label = label
        self.middle = mt.Tensor("middle", input.shape(), self.device)

    def forward(self):
        mt.softmax(self.input, self.middle)
        pred = np.argmax(self.middle.to_np(), axis=-1)
        mt.cross_entropy_loss(self.middle, self.output, self.label)        
        return pred
    
    def backward(self):
        return mt.cross_entropy_loss_with_softmax_backward(self.input, self.middle, self.output, self.label)
    
    def parameter(self):
        return [self.input, self.middle, self.output]