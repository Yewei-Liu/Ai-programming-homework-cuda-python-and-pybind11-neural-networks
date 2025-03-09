import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

def get_dataset(args):
    if args.dataset == "MNIST":
        data_root = "datasets/MNIST"
        trainset = datasets.MNIST(root=data_root, train=True, download=True)
        testset = datasets.MNIST(root=data_root, train=False, download=True)
        train_data = np.array(trainset.data).reshape(-1, 1, 28, 28).astype(float)
        train_data = (train_data / 255) - 0.5
        train_label = np.array(trainset.targets)
        test_data = np.array(testset.data).reshape(-1, 1, 28, 28).astype(float)
        test_data = (test_data / 255) - 0.5
        test_label = np.array(testset.targets)
        return train_data, train_label, test_data, test_label

    else:
        assert 0, 'dataset don\'t exist'

if __name__ == '__main__':
    import sys
    sys.path.append('build')
    import mytensor as mt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='MNIST')
    args = parser.parse_args()
    print(args)
    train_data, train_label, test_data, test_label = get_dataset(args)
    print(train_data.shape)
    input = mt.Tensor("input", (4, 1, 28, 28), "gpu")
    input.from_np(train_data[0:4])
    input.show()
