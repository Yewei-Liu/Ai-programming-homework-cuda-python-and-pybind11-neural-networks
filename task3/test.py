import sys
sys.path.append('build')

import mytensor as mt
import numpy as np
import torch
import torch.nn.functional as F

def l1_dist(a, b):
    assert a.shape == b.shape, "shape must be same !!!!!!!"
    return np.mean(abs(a - b))

def test(name, a, b):
    print(f"{name}  error: {l1_dist(a,b):.8f}")

def test_mt_ts(name, a:mt.Tensor, b:torch.Tensor):
    test(name, a.to_np(), b.detach().numpy())

def grad_test_mt_ts(name, a:mt.Tensor, b:torch.Tensor):
    test(name, a.grad_to_np(), b.grad.numpy())

def from_tensor(a:mt.Tensor, b:torch.Tensor):
    a.from_np(b.detach().numpy())

def grad_from_tensor(a:mt.Tensor, b:torch.Tensor):
    a.grad_from_np(b.detach().numpy())

#relu
print("relu")
shape = (4, 5)
x_mt = mt.Tensor("x", shape, "gpu")
y_mt = mt.Tensor("y", shape, "gpu")
x_np = np.random.normal(loc=0, scale=1, size=shape)
x_ts = torch.tensor(x_np, requires_grad=True)
y_ts = F.relu(x_ts)
x_mt.from_np(x_np)
mt.relu_forward(x_mt, y_mt)
test("relu forward", y_ts.detach().numpy(), y_mt.to_np())
grad = torch.randn(shape, dtype=torch.double)
y_ts.backward(gradient=grad)
y_mt.grad_from_np(grad.detach().numpy())
mt.relu_backward(x_mt, y_mt)
test("relu backward", x_mt.grad_to_np(), x_ts.grad.numpy())
print()

#sigmoid
print("sigmoid")
shape = (4, 5)
x_mt = mt.Tensor("x", shape, "gpu")
y_mt = mt.Tensor("y", shape, "gpu")
x_np = np.random.normal(loc=0, scale=1, size=shape)
x_ts = torch.tensor(x_np, requires_grad=True)
y_ts = F.sigmoid(x_ts)
x_mt.from_np(x_np)
mt.sigmoid_forward(x_mt, y_mt)
test("sigmoid forward", y_ts.detach().numpy(), y_mt.to_np())
grad = torch.randn(shape, dtype=torch.double)
y_ts.backward(gradient=grad)
y_mt.grad_from_np(grad.detach().numpy())
mt.sigmoid_backward(x_mt, y_mt)
test("sigmoid backward", x_mt.grad_to_np(), x_ts.grad.numpy())
print()

#fully connected layer
print("fully connected layer")
input_shape = (6, 10)
weight_shape = (10, 20)
bias_shape = (1, 20)
output_shape = (6, 20)
input_mt = mt.Tensor("input", input_shape, "gpu")
weight_mt = mt.Tensor("weight", weight_shape, "gpu")
bias_mt = mt.Tensor("bias", bias_shape, "gpu")
output_mt = mt.Tensor("output", output_shape, "gpu")
input_ts = torch.randn(input_shape, requires_grad=True)
weight_ts = torch.randn(weight_shape, requires_grad=True)
bias_ts = torch.randn(bias_shape, requires_grad=True)
output_ts = input_ts @ weight_ts + bias_ts
from_tensor(input_mt, input_ts)
from_tensor(weight_mt, weight_ts)
from_tensor(bias_mt, bias_ts)
from_tensor(output_mt, output_ts)
mt.fc_forward(input_mt, weight_mt, bias_mt, output_mt)
test_mt_ts("fc forward", output_mt, output_ts)
grad = torch.randn(output_shape, dtype=torch.double)
output_ts.backward(gradient=grad)
grad_from_tensor(output_mt, grad)
mt.fc_backward(input_mt, weight_mt, bias_mt, output_mt)
grad_test_mt_ts("fc backward input", input_mt, input_ts)
grad_test_mt_ts("fc backward weight", weight_mt, weight_ts)
grad_test_mt_ts("fc backward bias", bias_mt, bias_ts)
print()

#Convolution layer
print("convolutional")
input_shape = (4, 2, 16, 16)
kernel_shape = (3, 2, 3, 3)
output_shape = (4, 3, 16, 16)
input_mt = mt.Tensor("input", input_shape, "gpu")
kernel_mt = mt.Tensor("kernel", kernel_shape, "gpu")
output_mt = mt.Tensor("output", output_shape, "gpu")
input_ts = torch.randn(input_shape, requires_grad=True)
kernel_ts = torch.randn(kernel_shape, requires_grad=True)
output_ts = F.conv2d(input_ts, kernel_ts, stride=1, padding="same")
from_tensor(input_mt, input_ts)
from_tensor(kernel_mt, kernel_ts)
mt.conv_forward(input_mt, kernel_mt, output_mt)
test_mt_ts("conv forward", output_mt, output_ts)
grad = torch.randn(output_shape, dtype=torch.double)
output_ts.backward(gradient=grad)
grad_from_tensor(output_mt, grad)
mt.conv_backward(input_mt, kernel_mt, output_mt) 
grad_test_mt_ts("conv backward input", input_mt, input_ts)
grad_test_mt_ts("conv backward kernel", kernel_mt, kernel_ts)
print()

#Max Pooling layer
print("max pooling layer")
input_shape = (4, 3, 16, 16)
output_shape = (4, 3, 8, 8)
input_mt = mt.Tensor("input", input_shape, "gpu")
output_mt = mt.Tensor("output", output_shape, "gpu")
mask_mt = mt.Tensorbool("mask", input_shape, "gpu")
input_ts = torch.randn(input_shape, requires_grad=True)
output_ts = F.max_pool2d(input_ts, 2, 2)
from_tensor(input_mt, input_ts)
mask_mt = mt.max_pooling_forward(input_mt, output_mt, 2, 2)
test_mt_ts("max pooling forward", output_mt, output_ts)
grad = torch.randn(output_shape, dtype=torch.double)
output_ts.backward(gradient=grad)
grad_from_tensor(output_mt, grad)
mt.max_pooling_backward(input_mt, output_mt, mask_mt, 2, 2)
grad_test_mt_ts("max pooling backward", input_mt, input_ts)
print()

#Softmax and cross entropy loss
print("Softmax and cross entropy loss")
input_shape = (5, 10)
output_shape = (5, 10)
label_shape = (5, )
loss_shape = (5, )
input_mt = mt.Tensor("input", input_shape, "gpu")
output_mt = mt.Tensor("output", output_shape, "gpu")
input_ts = torch.randn(input_shape, requires_grad=True)
output_ts = F.softmax(input_ts, dim=-1)
from_tensor(input_mt, input_ts)
mt.softmax(input_mt, output_mt)
test_mt_ts("softmax forward", output_mt, output_ts)
loss_mt = mt.Tensor("loss", loss_shape, "gpu")
label_mt = mt.Tensorint("label", label_shape, "gpu")
label_ts = torch.randint(low=0, high=10, size=label_shape)
loss_ts = F.cross_entropy(input_ts, label_ts)
from_tensor(label_mt, label_ts)
mt.cross_entropy_loss(output_mt, loss_mt, label_mt)
test("cross entropy loss forward", np.mean(loss_mt.to_np()), loss_ts.detach().numpy())
grad = torch.randn((1, ), dtype=torch.double)
loss_ts.backward(gradient=grad[0])
grad_from_tensor(loss_mt, grad.expand((5, )) / 5)
mt.cross_entropy_loss_with_softmax_backward(input_mt, output_mt, loss_mt, label_mt)
grad_test_mt_ts("softmax and cross entropy loss backward", input_mt, input_ts)
print()






