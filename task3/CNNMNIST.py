import sys
sys.path.append('build')
import mytensor as mt
import numpy as np
from model import CNNMNIST, MyMLP
from optimizer import MySGD, SmartSGD
from dataset import get_dataset
import argparse
import time
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='MNIST')
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--model", type=str, default='CNNMNIST')
args = parser.parse_args()

DATASET = args.dataset
BATCH_SIZE = args.batch_size
EPOCH = args.epoch
DEVICE = "gpu"
LR = args.lr
VAL_SPLIT = 0.1
MODEL = args.model

train_data, train_label, test_data, test_label = get_dataset(args)
indices = np.arange(len(test_data))
np.random.shuffle(indices)
val_size = round(VAL_SPLIT * len(test_data))
val_data = test_data[indices[:val_size]]
val_label = test_label[indices[:val_size]]
test_data = test_data[indices[val_size:]]
test_label = test_label[indices[val_size:]]
print(f"{DATASET} with train data:{len(train_data)}, val_data: {len(val_data)}, test data: {len(test_data)}")
TRAIN_ITER = len(train_data) // BATCH_SIZE
VAL_ITER = len(val_data) // BATCH_SIZE
TEST_ITER = len(test_data) // BATCH_SIZE

# TRAIN_ITER = 5
# VAL_ITER = 5
# TEST_ITER = 5

if MODEL == 'CNNMNIST':
    model = CNNMNIST(DEVICE, BATCH_SIZE)
elif MODEL == 'MyMLP':
    model = MyMLP(DEVICE, BATCH_SIZE)
else:
    raise NotImplementedError
optimizer = SmartSGD(model.parameters, model.trained_parameters, LR, 0.7)

writer = SummaryWriter()
start_time = time.time()
for epoch in range(EPOCH):
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)
    train_loss = 0
    train_acc = 0
    for i in range(TRAIN_ITER):
        optimizer.zero_grad()
        data = train_data[indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
        label = train_label[indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
        pred = model.forward(data, label)
        train_acc += (pred == label).sum()
        train_loss += np.mean(model.get_loss())
        model.backward()
        optimizer.step()
    optimizer.schedular_step()
    train_loss /= TRAIN_ITER
    train_acc = train_acc / (TRAIN_ITER * BATCH_SIZE)
    writer.add_scalars("loss", {f'train': train_loss}, epoch + 1)
    writer.add_scalars("acc", {f'train': train_acc}, epoch + 1)
    indices = np.arange(len(val_data))
    np.random.shuffle(indices)
    val_acc = 0
    val_loss = 0
    for i in range(VAL_ITER):
        data = val_data[indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
        label = val_label[indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
        pred = model.forward(data, label)
        val_acc += (pred == label).sum()
        val_loss += np.mean(model.get_loss())
    val_loss /= VAL_ITER
    val_acc = val_acc / (VAL_ITER * BATCH_SIZE)
    writer.add_scalars("loss", {f'val': val_loss}, epoch + 1)
    writer.add_scalars("acc", {f'val': val_acc}, epoch + 1)
    print(f"epoch: {epoch + 1}, train loss: {train_loss}, train acc: {train_acc}, val loss: {val_loss}, val acc: {val_acc}")

writer.flush()
writer.close()
end_time = time.time()
print(f'training_time: {end_time - start_time:.1f}s')

def test(model):
    test_acc = 0
    for i in range(TEST_ITER):
        indices = np.arange(len(test_data))
        np.random.shuffle(indices)
        data = test_data[indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
        label = test_label[indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
        pred = model.forward(data, label)
        test_acc += (pred == label).sum()
    test_acc /= (TEST_ITER * BATCH_SIZE)
    print(f'all tst acc: {test_acc}')


test(model)

