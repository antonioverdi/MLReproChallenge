'''Testing CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchsummary import summary

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *

parser = argparse.ArgumentParser(description='ResNet56 pruning experiment testing properties')
parser.add_argument('--model_dir', type=str,  default='trained_models', help='directory of trained models. Should all be of same model type')
parser.add_argument('--arch', type=str, default="resnet56", help="model type to load pretrained weights into")
parser.add_argument('--log_dir', type=str, default="accuracy_logs", help='directory to save accuracy logs from pretrained models')
args = parser.parse_args()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=normalize, download=True),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

PATH = './cifar_resnet50.pth'
net = ResNet50()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net.load_state_dict(torch.load(PATH))
net.eval()


criterion = nn.CrossEntropyLoss()

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0) 
            correct += predicted.eq(targets).sum().item()
    
    acc = 100.*correct/total
    print(acc)

# test()

summary(net, (3, 32, 32), batch_size=128, device='cuda')