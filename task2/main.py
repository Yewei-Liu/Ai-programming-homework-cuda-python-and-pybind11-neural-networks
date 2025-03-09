import numpy as np
import torch
from torch import nn 
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time


BATCH_SIZE = 64
NUM_WORKERS = 2
LEARNING_RATE = 0.0001
NUM_EPOCH = 50
VALIDATION_SPLIT = 0.1
DROPOUT_RATIO = 0.5
PARALLEL = True

################################################################################
#0. detect GPU
################################################################################
print("start: 0. detect GPU")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device_count = torch.cuda.device_count()
print(f"Number of GPUs available: {device_count}")
if device_count > 0:
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

print("end 0. detect GPU")
print()
################################################################################
#1.load and preprocess MNIST
################################################################################
print("start: 1.load and preprocess MNIST")
#load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

train_dataset = datasets.MNIST(root='./datasets',
                                 train=True,
                                 transform=transform,
                                 download=False)

train_size = int((1 - VALIDATION_SPLIT) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

test_dataset = datasets.MNIST(root='./datasets',
                                train=False,
                                transform=transform,
                                download=False)
                                
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=NUM_WORKERS,
                          shuffle=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS,
                        shuffle=False)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1000,
                         num_workers=NUM_WORKERS,
                         shuffle=False)

print("train_data_num: " + str(len(train_dataset)))
print("val_data_num:   " + str(len(val_dataset)))
print("test_data_num:  " + str(len(test_dataset)))

print("end:   1.load and preprocess MNIST")
print()

#################################################################################
#2.CNN Construction
#################################################################################
print("start: 2.CNN Construction")
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.dropout = nn.Dropout(p = DROPOUT_RATIO)
        
        self.fc1 = nn.Linear(256, 128)  
        self.fc2 = nn.Linear(128, 10) 

    def forward(self, x):

        x = self.pool1(self.conv1_2(F.relu(self.conv1_1(x))))
        x = self.dropout(x)
        x = self.pool2(self.conv2_2(F.relu(self.conv2_1(x))))
        x = self.dropout(x)
        x = self.pool3(self.conv3_2(F.relu(self.conv3_1(x))))
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, 256)  
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.linear1 = nn.Linear(784, 128)
#         self.linear2 = nn.Linear(128, 16)
#         self.linear3 = nn.Linear(16, 10)

#     def forward(self, x):
#         x = F.relu(self.linear1(x.view(x.shape[0], -1)))
#         x = F.relu(self.linear2(x))
#         x = self.linear3(x)
#         return x

    
model = CNN().to(device)
if PARALLEL:
    model = nn.DataParallel(model)
print(model)
print("end:   2.CNN Construction")
print()

##########################################################################
#3.loss function and optimizer
##########################################################################
print("start: 3.loss function and optimizer")

#loss function
loss_func = nn.CrossEntropyLoss() 
print(loss_func)

#optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#optimizer = optim.SGD(model.parameters(), 0.01)
print(optimizer)

print("end:   3.loss function and optimizer")
print()

#########################################################################
#4.training
#########################################################################
print("start: 4.training")
if PARALLEL:
    suffix = 'parallel'
else:
    suffix = 'noneparallel'

#train
def train(model, optimizer):
    writer = SummaryWriter()
    print(device)
    start_time = time.time()
    for epoch in range(NUM_EPOCH):
        model.train()
        epoch_losses = []
        correct = 0
        total = 0
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            total += len(x)
            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == labels).sum().item()
            loss = loss_func(y_pred, labels)
            epoch_losses.append(loss)
            loss.backward()
            optimizer.step()
        epoch_loss = torch.Tensor(epoch_losses).mean()
        train_acc = correct / total
        writer.add_scalars("loss", {f'train_{suffix}': epoch_loss}, epoch + 1)
        writer.add_scalars("acc", {f'train_{suffix}': train_acc}, epoch + 1)

        model.eval()
        val_epoch_loss = []
        correct = 0
        total = 0
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)
                y_pred = model(x)
                val_loss = loss_func(y_pred, labels)
                val_epoch_loss.append(val_loss)
                _, predicted = torch.max(y_pred, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_loss = torch.Tensor(val_epoch_loss).mean()
        val_acc = correct / total
        writer.add_scalars("loss", {f'val_{suffix}': val_loss}, epoch + 1)
        writer.add_scalars("acc", {f'val_{suffix}': val_acc}, epoch + 1)
        print(f"epoch: {epoch + 1}  train loss: {epoch_loss:.5f}  val loss:{val_loss:.5f}  train acc:{train_acc:.5f}  val acc:{val_acc:.5f}")
    
    writer.flush()
    writer.close()
    end_time = time.time()
    print(f'training_time: {end_time - start_time:.1f}s')

train(model, optimizer)

print("end:   4.training")
print()

#########################################################################
#5.test
#########################################################################
print("start: 5.test")

def test(model):
    with torch.no_grad():
        sum = 0
        for x, labels in test_loader:
            y_pred = torch.argmax(model(x.to(device)), dim=1)
            sum += torch.sum(y_pred == labels.to(device))
        print(f'all tst acc: {sum/len(test_dataset)}')

def test_each_label(model, target_label):
    with torch.no_grad():
        images, labels = zip(*[(image, label) for image, label in test_dataset if label==target_label])
        images, labels = torch.tensor(np.array(images)).to(device), torch.tensor(labels).to(device)
        acc = torch.mean(torch.argmax(model(images), dim=1) == labels, dtype=torch.float64)
        print(f"{target_label} tst acc: {acc}")    

test(model)
for label in range(10):
    test_each_label(model, label)

print("end:   5.test")
print()
