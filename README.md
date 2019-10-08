# ResNet-Implementation

## ResNet
一般 CNN 中，當層數增多時，介於 0~1 之間的回傳梯度使乘積愈來愈小，甚至趨於零，梯度值近乎消失，導致權重 (weight) 無法有效更新，訓練速度緩慢。
### Resnet 殘差網路：在網路中加入 Short cut 的設計
<img src="https://github.com/jason971019/ResNet-Implementation/blob/master/shortcut.png" width="500" height="350">  
從上圖可發現，F(x)+x 多了 ”x” 使在 backpropagation 做 chain rule 時至少微分後仍保有一個 1 存在，解決梯度消失的問題。

## Data：Cifar-10
取用 Cifar-10 資料對飛機、汽車、船、貓、狗等10個種類進行 traing 並做出預測。
<img src="https://github.com/jason971019/ResNet-Implementation/blob/master/cifar-10.jpg" width="500" height="350">

## Code
### Adjust Learning Rate
在指定的 epochs 更新 learning rate 有效增加 training 速度
```python
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
    
    return lr
```
### Build Model
```python
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
```    
### Training
```python
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
```
## Result
### Learning Rate
![image](https://github.com/jason971019/ResNet-Implementation/blob/master/learning%20rate.png)  
分別在epochs 80、110、130做learning rate的更新，可發現：  
epoch = 80 的更新後 accuracy 有顯著提升  
epoch = 110 的更新後 accuracy 些許上升  
epoch = 130 的更新後 accuracy 幾乎趨緩  
### Accuracy
![image](https://github.com/jason971019/ResNet-Implementation/blob/master/accuracy.png)  
大約在 110 個 epochs 後 Accuracy 逐漸趨緩，可得知相較於一般 CNN，Resnet 可大幅減少訓練次數，節省訓練時間。
