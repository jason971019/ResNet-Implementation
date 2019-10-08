from __future__ import print_function
import time
import torch
import random
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


########## for reproducing the same results ##########
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic=True



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
test_bs = 128

######################## Create Dataset and DataLoader ########################
mean = [0.49139968, 0.48215827, 0.44653124]
std = [0.24703233, 0.24348505, 0.26158768]



train_tfm = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])

test_tfm = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])

Train_Set = datasets.CIFAR10(root='./data', train=True, transform=train_tfm, download=True)
Test_Set = datasets.CIFAR10(root='./data', train=False, transform=test_tfm, download=True)

train_loader = DataLoader(Train_Set, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

test_loader = DataLoader(Test_Set, batch_size=test_bs, shuffle=False,
                      num_workers=0, pin_memory=True)

######################## Helper Function ########################
def train(model, data_loader, optimizer, epoch, verbose=True):
    model.train()
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss   = F.cross_entropy(output, target)
        loss_avg += loss.item()
        loss.backward()
        optimizer.step()
        verbose_step = len(data_loader) // 10
        if batch_idx % verbose_step == 0 and verbose:
            print('Train Epoch: {}  Step [{}/{} ({:.0f}%)]  Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
    return loss_avg / len(data_loader)

def test(model, data_loader):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()   

        test_loss /= len(data_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))   
    return float(correct) / len(data_loader.dataset)

def adjust_learning_rate(base_lr, optimizer, epoch, epoch_list=None):
    # Set base_lr as initial LR and adjust learning rate by *0.1 at assigned epochs
    base_lr=0.1
    if epoch_list is not None:
        index = 0
        for i, e in enumerate(epoch_list, 1):
            if epoch >= e:
                index = i
            else:
                break
        lr = base_lr*(0.1**index)

    # way to change lr in model
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    print('lr:', lr)
    
    return lr

######################## Build Model ########################
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.inplanes = 3
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.pre_layer = nn.Sequential( 
        nn.Conv2d(3, 64, 7, 2, 3, bias=False),
        nn.BatchNorm2d(64), 
        nn.ReLU(), 
        nn.MaxPool2d(3, 2, 1) , 
        )
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.layer1 = self._make_layer(64, 64, 3) 
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2) 
        self.layer4 = self._make_layer(256, 512, 3, stride=2) 
        self.fc = nn.Linear(512, num_classes)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channel, out_channel, block_num, stride=1):
      shortcut = nn.Sequential( 
      nn.Conv2d(in_channel, out_channel, 1, stride, bias=False), 
      nn.BatchNorm2d(out_channel), 
      ) 
      layers = []
      layers.append(BasicBlock(in_channel, out_channel, stride, shortcut))
      for i in range(1, block_num):
        layers.append(BasicBlock(out_channel, out_channel)) 
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet34():
    return ResNet()

net = resnet34().to(device)

# Define Loss and optimizer
learning_rate = 0.1
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

# Train the model
num_epochs = 150

StartTime = time.time()
loss, val_acc, lr_curve = [], [], []
for epoch in range(num_epochs):
    learning_rate = adjust_learning_rate(learning_rate, optimizer, epoch, epoch_list=[80, 110, 130])
    train_loss = train(net, train_loader, optimizer, epoch, verbose=True)
    valid_acc  = test(net, test_loader)
    loss.append(train_loss)
    val_acc.append(valid_acc)
    lr_curve.append(learning_rate)


EndTime = time.time()
print('Time Usage: ', str(datetime.timedelta(seconds=int(round(EndTime-StartTime)))))



plt.figure()
plt.plot(loss)
plt.title('Train Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

plt.figure()
plt.plot(val_acc)
plt.title('Valid Acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.figure()
plt.plot(lr_curve)
plt.title('Learning Rate')
plt.xlabel('epochs')
plt.ylabel('lr')
plt.yscale('log')
plt.show()
