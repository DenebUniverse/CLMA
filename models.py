import torch
import torch.nn as nn
import torchvision.models as models
        
class Vgg_16(nn.Module):
    def __init__(self, num_classes=1000):  # num_classes
        super(Vgg_16, self).__init__()
        self.model = models.vgg16(pretrained=True).cuda().eval()
        features = list(self.model.features)
        self.features = torch.nn.ModuleList(features).cuda().eval()
        self.avgpool = self.model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        for ii, model in enumerate(self.features):
            x = model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def prediction(self, x, internal=[]):
        layers = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if (ii in internal):
                layers.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        pred = self.classifier(x)
        return layers, pred

class Res_152(nn.Module):
    def __init__(self, num_classes=1000):
        super(Res_152, self).__init__()
        self.model = models.resnet152(pretrained=True).cuda().eval()
        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.maxpool = self.model.maxpool
        self.relu = self.model.relu
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        self.avgpool = self.model.avgpool
        self.fc = self.model.fc
        
    def prediction(self, x, internal=[]):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        layers = []
        x = self.layer1(x)
        layers.append(x)
        x = self.layer2(x)
        layers.append(x)
        x = self.layer3(x)
        layers.append(x)
        x = self.layer4(x)
        layers.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        pred = self.fc(x)
        return layers, pred

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
    
class Inc_v3(nn.Module):
    def __init__(self, num_classes=1000):
        super(Inc_v3, self).__init__()
        self.model = models.inception_v3(pretrained=True).cuda().eval()
        self.aux_logits = self.model.aux_logits
        self.transform_input = self.model.transform_input
        self.Conv2d_1a_3x3 = self.model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = self.model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = self.model.Conv2d_2b_3x3
        self.maxpool1 = self.model.maxpool1
        self.Conv2d_3b_1x1 = self.model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = self.model.Conv2d_4a_3x3
        self.maxpool2 = self.model.maxpool2
        self.Mixed_5b = self.model.Mixed_5b
        self.Mixed_5c = self.model.Mixed_5c
        self.Mixed_5d = self.model.Mixed_5d
        self.Mixed_6a = self.model.Mixed_6a
        self.Mixed_6b = self.model.Mixed_6b
        self.Mixed_6c = self.model.Mixed_6c
        self.Mixed_6d = self.model.Mixed_6d
        self.Mixed_6e = self.model.Mixed_6e
        self.Mixed_7a = self.model.Mixed_7a
        self.Mixed_7b = self.model.Mixed_7b
        self.Mixed_7c = self.model.Mixed_7c
        self.avgpool = self.model.avgpool
        self.dropout = self.model.dropout
        self.fc = self.model.fc
    
    def _forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        
    def prediction(self, x, internal=[]):
        layers = []
        x = self.Conv2d_1a_3x3(x)
        layers.append(x)
        x = self.Conv2d_2a_3x3(x)
        layers.append(x)
        x = self.Conv2d_2b_3x3(x)
        layers.append(x)
        x = self.maxpool1(x)
        layers.append(x)
        x = self.Conv2d_3b_1x1(x)
        layers.append(x)
        x = self.Conv2d_4a_3x3(x)
        layers.append(x)
        x = self.maxpool2(x)
        layers.append(x)
        x = self.Mixed_5b(x)
        layers.append(x)
        x = self.Mixed_5c(x)
        layers.append(x)
        x = self.Mixed_5d(x)
        layers.append(x)
        x = self.Mixed_6a(x)
        layers.append(x)
        x = self.Mixed_6b(x)
        layers.append(x)
        x = self.Mixed_6c(x)
        layers.append(x)
        x = self.Mixed_6d(x)
        layers.append(x)
        x = self.Mixed_6e(x)
        layers.append(x)
        x = self.Mixed_7a(x)
        layers.append(x)
        x = self.Mixed_7b(x)
        layers.append(x)
        x = self.Mixed_7c(x)
        layers.append(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        pred = self.fc(x)
        return layers, pred
    
class BasicConv2d(nn.Module):
 
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)
 
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
 
 
# 标准Inception module
class Mixed_5b(nn.Module):
 
    def __init__(self):
        super(Mixed_5b, self).__init__()
 
        # branch0: 1*1
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
 
        # branch1: 1*1, 5*5
        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )
 
        # branch2: 1*1, 3*3, 3*3
        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )
 
        # branch3: avgPool, 1*1
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )
 
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)       # 96+64+96+64 = 320
        return out
 
# figure 16.
class Block35(nn.Module):
 
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
 
        self.scale = scale
 
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
 
        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
 
        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )
 
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)
 
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out
 
 
# Reduction-A figure7.
class Mixed_6a(nn.Module):
 
    def __init__(self):
        super(Mixed_6a, self).__init__()
 
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
 
        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )
 
        self.branch2 = nn.MaxPool2d(3, stride=2)
 
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out
 

class Block17(nn.Module):
 
    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
 
        self.scale = scale
 
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
 
        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )
 
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)     # 论文为1154，此处为1088，这个参数必须与输入的一样
        self.relu = nn.ReLU(inplace=False)
 
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out
 
 
# Reduction-B figure 18.
class Mixed_7a(nn.Module):
 
    def __init__(self):
        super(Mixed_7a, self).__init__()
 
        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )
 
        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )
 
        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )
 
        self.branch3 = nn.MaxPool2d(3, stride=2)
 
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out
 

class Block8(nn.Module):
 
    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
 
        self.scale = scale
        self.noReLU = noReLU
 
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
 
        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
 
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)
 
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out
 
 
class IncRes_v2(nn.Module):
 
    def __init__(self, num_classes=1000):
        super(IncRes_v2, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)

        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )

        self.mixed_6a = Mixed_6a()

        self.repeat_1 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        )

        self.mixed_7a = Mixed_7a()

        self.repeat_2 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)
 
    def features(self, input):
        layers = []
        x = self.conv2d_1a(input)
        layers.append(x)
        x = self.conv2d_2a(x)
        layers.append(x)
        x = self.conv2d_2b(x)
        layers.append(x)
        x = self.maxpool_3a(x)
        layers.append(x)
        x = self.conv2d_3b(x)
        layers.append(x)
        x = self.conv2d_4a(x)
        layers.append(x)
        x = self.maxpool_5a(x)
        layers.append(x)
        x = self.mixed_5b(x)
        layers.append(x)
        x = self.repeat(x)
        layers.append(x)
        x = self.mixed_6a(x)
        layers.append(x)
        x = self.repeat_1(x)
        layers.append(x)
        x = self.mixed_7a(x)
        layers.append(x)
        x = self.repeat_2(x)
        layers.append(x)
        x = self.block8(x)
        layers.append(x)
        x = self.conv2d_7b(x)
        layers.append(x)
        return layers, x
 
    def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x
 
    def forward(self, input):
        _, x = self.features(input)
        x = self.logits(x)
        return x
        
    def prediction(self, input, internal=[]):
        layers, x = self.features(input)
        x = self.logits(x)
        return layers, x
