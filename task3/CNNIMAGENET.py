import sys
sys.path.append('build')
import mytensor as mt
import numpy as np
from model import CNNIMAGENET
from optimizer import SmartSGD
import argparse
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()

DATA_PATH = '/home/liuyewei/imagenet'
BATCH_SIZE = args.batch_size
EPOCH = args.epoch
DEVICE = "gpu"
LR = args.lr
VAL_SPLIT = 0.1

model = CNNIMAGENET(DEVICE, BATCH_SIZE)
optimizer = SmartSGD(model.parameters, model.trained_parameters, LR, 1)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = datasets.ImageNet(root=DATA_PATH, split='train', transform=train_transform)
test_data = datasets.ImageNet(root=DATA_PATH, split='val', transform=test_transform)
print(len(train_data))
print(len(test_data))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True)


writer = SummaryWriter()
start_time = time.time()
for epoch in range(EPOCH):
    train_loss = 0
    train_acc = 0
    idx = 0
    for data, label in train_loader:
        idx += 1
        optimizer.zero_grad()
        data = data.detach().numpy()
        label = label.detach().numpy()
        pred = model.forward(data, label)
        train_acc += (pred == label).mean()
        train_loss += np.mean(model.get_loss())
        print(np.mean(model.get_loss()))
        model.backward()
        optimizer.step()
    optimizer.schedular_step()
    train_loss /= idx
    train_acc /= idx
    writer.add_scalars("loss", {f'train': train_loss}, epoch + 1)
    writer.add_scalars("acc", {f'train': train_acc}, epoch + 1)
    print(f"epoch: {epoch + 1}, train loss: {train_loss}, train acc: {train_acc}")

writer.flush()
writer.close()
end_time = time.time()
print(f'training_time: {end_time - start_time:.1f}s')

def test(model):
    test_acc = 0
    idx = 0
    for data, label in test_loader:
        idx += 1
        data = data.detach().numpy()
        label = label.detach().numpy()
        pred = model.forward(data, label)
        test_acc += (pred == label).mean()
    test_acc /= idx
    print(f"test acc: {test_acc}")


test(model)

