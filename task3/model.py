import sys
sys.path.append('build')
import mytensor as mt
import numpy as np
from layers import *
from optimizer import MySGD
from dataset import get_dataset
import torch.nn.functional as F
from torch import optim

class MyMLP():
    def __init__(self, device="gpu", batch_size=100, name=None):
        self.name = name
        self.device = device
        self.batch_size = batch_size

        self.input = mt.Tensor("input", [self.batch_size, 784], self.device)
        self.x1 = mt.Tensor("x1", [self.batch_size, 128], self.device)
        self.y1 = mt.Tensor("y1", [self.batch_size, 128], self.device)
        self.x2 = mt.Tensor("x2", [self.batch_size, 16], self.device)
        self.y2 = mt.Tensor("y2", [self.batch_size, 16], self.device)
        self.x3 = mt.Tensor("x3", [self.batch_size, 10], self.device)
        self.loss = mt.Tensor("loss", [self.batch_size], self.device)
        self.label = mt.Tensorint("label", [self.batch_size], self.device)

        self.linear1 = MyLinear(self.input, self.x1, self.device)
        self.relu1 = MyRelu(self.x1, self.y1, self.device)
        self.linear2 = MyLinear(self.y1, self.x2, self.device)
        self.relu2 = MyRelu(self.x2, self.y2, self.device)
        self.linear3 = MyLinear(self.y2, self.x3, self.device)
        self.lossfunc = MyCrossEntropyLoss(self.x3, self.loss, self.label)

        self.layers = [self.linear1, self.relu1, self.linear2, self.relu2, self.linear3, self.lossfunc]
        self.parameters = []
        self.trained_parameters = []
        for layer in self.layers:
            self.parameters.extend(layer.parameter())
            self.trained_parameters.extend(layer.trained_parameter())

    
    def forward(self, input: np.array, label: np.array):
        self.input.reshape([self.batch_size, 1, 28, 28])
        self.input.from_np(input)
        self.label.from_np(label)
        self.input.reshape([self.batch_size, 784])
        self.linear1.forward()
        self.relu1.forward()
        self.linear2.forward()
        self.relu2.forward()
        self.x3.from_np(np.ones(self.x3.shape()) * 10000)
        self.linear3.forward()
        pred = self.lossfunc.forward()

        return pred

    def backward(self):
        self.loss.grad_from_np(np.ones(self.loss.shape()) / self.batch_size)
        self.input.reshape([self.batch_size, 784])
        self.lossfunc.backward()
        self.linear3.backward()
        self.relu2.backward()
        self.linear2.backward()
        self.relu1.backward()
        self.linear1.backward()
        
    
    def get_loss(self):
        return self.loss.to_np()

class CNNMNIST():
    def __init__(self, device="gpu", batch_size=100, name=None):
        self.name = name
        self.device = device
        self.batch_size = batch_size

        self.input = mt.Tensor("input", [self.batch_size, 1, 28, 28], self.device)
        self.x1 = mt.Tensor("x1", [self.batch_size, 16, 28, 28], self.device)
        self.y1 = mt.Tensor("y1", [self.batch_size, 16, 28, 28], self.device)
        self.x2 = mt.Tensor("x2", [self.batch_size, 32, 28, 28], self.device)
        self.x3 = mt.Tensor("x3", [self.batch_size, 32, 14, 14], self.device)
        self.x4 = mt.Tensor("x4", [self.batch_size, 64, 14, 14], self.device)
        self.y4 = mt.Tensor("y4", [self.batch_size, 64, 14, 14], self.device)
        self.x5 = mt.Tensor("x5", [self.batch_size, 128, 14, 14], self.device)
        self.x6 = mt.Tensor("x6", [self.batch_size, 128, 7, 7], self.device)
        self.x7 = mt.Tensor("x7", [self.batch_size, 512], self.device)
        self.y7 = mt.Tensor("y7", [self.batch_size, 512], self.device)
        self.x8 = mt.Tensor("x8", [self.batch_size, 10], self.device)
        self.loss = mt.Tensor("loss", [self.batch_size], self.device)
        self.label = mt.Tensorint("label", [self.batch_size], self.device)

        self.conv1 = MyConv(self.input, self.x1, self.device)
        self.relu1 = MyRelu(self.x1, self.y1, self.device)
        self.conv2 = MyConv(self.y1, self.x2, self.device)
        self.pooling2 = MyMaxPooling(self.x2, self.x3, self.device)
        self.conv3 = MyConv(self.x3, self.x4, self.device)
        self.relu3 = MyRelu(self.x4, self.y4, self.device)
        self.conv4 = MyConv(self.y4, self.x5, self.device)
        self.pooling4 = MyMaxPooling(self.x5, self.x6, self.device)

        self.x6.reshape((self.batch_size, 6272))
        self.linear1 = MyLinear(self.x6, self.x7, self.device)
        self.relu = MyRelu(self.x7, self.y7, self.device)
        self.linear2 = MyLinear(self.y7, self.x8, self.device)
        self.loss_func = MyCrossEntropyLoss(self.x8, self.loss, self.label, self.device)
        self.x6.reshape((self.batch_size, 128, 7, 7))

        self.layers = [self.conv1, self.relu1, self.conv2, self.pooling2, self.conv3, self.relu3, self.conv4,
                       self.pooling4, self.linear1, self.relu, self.linear2, self.loss_func]
        self.parameters = []
        self.trained_parameters = []
        for layer in self.layers:
            self.parameters.extend(layer.parameter())
            self.trained_parameters.extend(layer.trained_parameter())
    
    def forward(self, input: np.array, label: np.array):
        self.input.from_np(input)
        self.label.from_np(label)
        self.conv1.forward()
        self.relu1.forward()
        self.conv2.forward()        
        self.pooling2.forward()
        self.conv3.forward()
        self.relu3.forward()
        self.conv4.forward()
        self.pooling4.forward()
        self.x6.reshape((self.batch_size, 6272))
        self.linear1.forward()
        self.relu.forward()
        self.linear2.forward()
        #self.x8.show()
        pred = self.loss_func.forward()
        #self.loss.show()
        self.x6.reshape((self.batch_size, 128, 7, 7))

        return pred

    def backward(self):
        self.loss.grad_from_np(np.ones(self.loss.shape()) / self.batch_size)

        self.x6.reshape((self.batch_size, 6272))
        self.loss_func.backward()
        self.linear2.backward()
        self.relu.backward()
        self.linear1.backward()
        self.x6.reshape((self.batch_size, 128, 7, 7))
        self.pooling4.backward()
        self.conv4.backward()
        self.relu3.backward()
        self.conv3.backward()
        self.pooling2.backward()
        self.conv2.backward()
        self.relu1.backward()
        self.conv1.backward()
    
    def get_loss(self):
        return self.loss.to_np()
    

class CNNIMAGENET():
    def __init__(self, device="gpu", batch_size=100, name=None):
        self.name = name
        self.device = device
        self.batch_size = batch_size

        self.input = mt.Tensor("input", [self.batch_size, 3, 224, 224], self.device)
        self.x1 = mt.Tensor("x1", [self.batch_size, 4, 224, 224], self.device)
        self.x2 = mt.Tensor("x2", [self.batch_size, 4, 112, 112], self.device)
        self.y2 = mt.Tensor("y2", [self.batch_size, 4, 112, 112], self.device)
        self.x3 = mt.Tensor("x3", [self.batch_size, 8, 112, 112], self.device)
        self.x4 = mt.Tensor("x4", [self.batch_size, 8, 56, 56], self.device)
        self.y4 = mt.Tensor("y4", [self.batch_size, 8, 56, 56], self.device)
        self.x5 = mt.Tensor("x5", [self.batch_size, 16, 56, 56], self.device)
        self.x6 = mt.Tensor("x6", [self.batch_size, 16, 28, 28], self.device)
        self.y6 = mt.Tensor("y6", [self.batch_size, 16, 28, 28], self.device)
        self.x7 = mt.Tensor("x7", [self.batch_size, 32, 28, 28], self.device)
        self.x8 = mt.Tensor("x8", [self.batch_size, 32, 14, 14], self.device)
        self.y8 = mt.Tensor("y8", [self.batch_size, 32, 14, 14], self.device)
        self.x9 = mt.Tensor("x9", [self.batch_size, 64, 14, 14], self.device)
        self.x10 = mt.Tensor("x10", [self.batch_size, 64, 7, 7], self.device)
        self.y10 = mt.Tensor("y10", [self.batch_size, 3136], self.device)
        self.x11 = mt.Tensor("x11", [self.batch_size, 1536], self.device)
        self.y11 = mt.Tensor("y11", [self.batch_size, 1536], self.device)
        self.x12 = mt.Tensor("x12", [self.batch_size, 1000], self.device)
        self.loss = mt.Tensor("loss", [self.batch_size], self.device)
        self.label = mt.Tensorint("label", [self.batch_size], self.device)

        self.conv1 = MyConv(self.input, self.x1, self.device)
        self.pool1 = MyMaxPooling(self.x1, self.x2, self.device)
        self.relu1 = MyRelu(self.x2, self.y2, self.device)
        self.conv2 = MyConv(self.y2, self.x3, self.device)
        self.pool2 = MyMaxPooling(self.x3, self.x4, self.device)
        self.relu2 = MyRelu(self.x4, self.y4, self.device)
        self.conv3 = MyConv(self.y4, self.x5, self.device)
        self.pool3 = MyMaxPooling(self.x5, self.x6, self.device)
        self.relu3 = MyRelu(self.x6, self.y6, self.device)
        self.conv4 = MyConv(self.y6, self.x7, self.device)
        self.pool4 = MyMaxPooling(self.x7, self.x8, self.device)
        self.relu4 = MyRelu(self.x8, self.y8, self.device)
        self.conv5 = MyConv(self.y8, self.x9, self.device)
        self.pool5 = MyMaxPooling(self.x9, self.x10, self.device)
        self.relu5 = MyRelu(self.x10, self.y10, self.device)

        self.linear1 = MyLinear(self.y10, self.x11, self.device)
        self.relu6 = MyRelu(self.x11, self.y11, self.device)
        self.linear2 = MyLinear(self.y11, self.x12, self.device)
        self.loss_func = MyCrossEntropyLoss(self.x12, self.loss, self.label, self.device)

        self.layers = [self.conv1, self.pool1, self.relu1, 
                       self.conv2, self.pool2, self.relu2, 
                       self.conv3, self.pool3, self.relu3, 
                       self.conv4, self.pool4, self.relu4, 
                       self.conv5, self.pool5, self.relu5, 
                       self.linear1, self.relu6, self.linear2, self.loss_func]
        self.parameters = []
        self.trained_parameters = []
        for layer in self.layers:
            self.parameters.extend(layer.parameter())
            self.trained_parameters.extend(layer.trained_parameter())
    
    def forward(self, input: np.array, label: np.array):
        self.input.from_np(input)
        self.label.from_np(label)
        self.y10.reshape([self.batch_size, 64, 7, 7])
        self.conv1.forward()
        self.pool1.forward()
        self.relu1.forward()
        self.conv2.forward()
        self.pool2.forward()
        self.relu2.forward()
        self.conv3.forward()
        self.pool3.forward()
        self.relu3.forward()
        self.conv4.forward()
        self.pool4.forward()
        self.relu4.forward()
        self.conv5.forward()
        self.pool5.forward()
        self.relu5.forward()
        self.y10.reshape([self.batch_size, 3136])
        self.linear1.forward()
        self.relu6.forward()
        self.linear2.forward()
        pred = self.loss_func.forward()
        return pred

    def backward(self):
        self.loss.grad_from_np(np.ones(self.loss.shape()) / self.batch_size)
        self.y10.reshape([self.batch_size, 3136])
        self.loss_func.backward()
        self.linear2.backward()
        self.relu6.backward()
        self.linear1.backward()
        self.y10.reshape([self.batch_size, 64, 7, 7])
        self.relu5.backward()
        self.pool5.backward()
        self.conv5.backward()
        self.relu4.backward()
        self.pool4.backward()
        self.conv4.backward()
        self.relu3.backward()
        self.pool3.backward()
        self.conv3.backward()
        self.relu2.backward()
        self.pool2.backward()
        self.conv2.backward()
        self.relu1.backward()
        self.pool1.backward()
        self.conv1.backward()

    def get_loss(self):
        return self.loss.to_np()


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 16)
        self.linear3 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.linear1(x.view(x.shape[0], -1)))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='MNIST')
    args = parser.parse_args()
    BATCH_SIZE= 1
    DATASET = args.dataset
    EPOCH = 10
    DEVICE = "gpu"
    LR = 0.01
    VAL_SPLIT = 0.1


    model = MyMLP(batch_size=BATCH_SIZE)
    test_model = TestModel()

    model.linear1.weight.from_np(np.asfortranarray(test_model.linear1.weight.data).T)
    model.linear1.bias.from_np(np.asfortranarray(test_model.linear1.bias.data).reshape(1, -1))
    model.linear2.weight.from_np(np.asfortranarray(test_model.linear2.weight.data).T)
    model.linear2.bias.from_np(np.asfortranarray(test_model.linear2.bias.data).reshape(1, -1))
    model.linear3.weight.from_np(np.asfortranarray(test_model.linear3.weight.data).T)
    model.linear3.bias.from_np(np.asfortranarray(test_model.linear3.bias.data).reshape(1, -1))
    

    optimizer = MySGD(model.parameters, LR)
    test_optimizer = optim.SGD(test_model.parameters(), LR)
    loss_func = nn.CrossEntropyLoss(reduce='mean')
    train_data, train_label, test_data, test_label = get_dataset(args)
    indices = np.arange(len(test_data))
    np.random.shuffle(indices)
    val_size = round(VAL_SPLIT * len(test_data))
    val_data = test_data[indices[:val_size]]
    val_label = test_label[indices[:val_size]]
    test_data = test_data[indices[val_size:]]
    test_label = test_label[indices[val_size:]]
    TRAIN_ITER = len(train_data) // BATCH_SIZE
    VAL_ITER = len(val_data) // BATCH_SIZE

    for epoch in range(EPOCH):
        indices = np.arange(len(train_data))
        np.random.shuffle(indices)
        train_loss = 0
        train_acc = 0
        for i in range(TRAIN_ITER):
            optimizer.zero_grad()
            data = train_data[0: BATCH_SIZE]
            label = train_label[0: BATCH_SIZE]
            pred = model.forward(data, label)
            train_acc += (pred == label).sum()
            train_loss += np.mean(model.get_loss())
            model.backward()
            optimizer.step()
            print(f"my acc: {np.mean(pred == label)}")

            test_optimizer.zero_grad()
            data_tensor = torch.tensor(data, dtype = torch.float)
            output = test_model(data_tensor)
            print(f"test acc: {np.mean(np.argmax(output.detach().numpy(), axis=-1) == label)}")
            loss = loss_func(output, torch.tensor(label))
            loss.backward()
            test_optimizer.step()

            print(test_model.linear3.weight.data.T)
            model.linear3.weight.show()
            print(test_model.linear3.weight.grad.data.T)

            print(output)
            model.x3.show()

            print(f"err: {np.sum(abs(model.linear2.bias.grad_to_np() - np.asfortranarray(test_model.linear2.bias.grad.data).reshape(1, -1)))}")

            if i == 10:
                assert 0

