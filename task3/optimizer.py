import sys
sys.path.append('build')
import mytensor as mt
import numpy as np

class MySGD():
    def __init__(self, parameters, trained_parameters, lr):
        self.parameters = parameters
        self.trained_parameters = trained_parameters
        self.lr = lr

    def step(self):
        grad = 0
        val = 0
        for param in self.trained_parameters:
            grad = param.grad_to_np()
            val = param.to_np()
            val -= self.lr * grad
            param.from_np(val)
    
    def zero_grad(self):
        for param in self.parameters:
            param.grad_from_np(np.zeros(param.shape()))

class SmartSGD():
    def __init__(self, parameters, trained_parameters, lr, decay):
        self.parameters = parameters
        self.trained_parameters = trained_parameters
        self.lr = lr
        self.decay = decay

    def step(self):
        grad = 0
        val = 0
        for param in self.trained_parameters:
            grad = param.grad_to_np()
            val = param.to_np()
            val -= self.lr * grad
            param.from_np(val)
    
    def schedular_step(self):
        self.lr *= self.decay
    
    def zero_grad(self):
        for param in self.parameters:
            param.grad_from_np(np.zeros(param.shape()))


