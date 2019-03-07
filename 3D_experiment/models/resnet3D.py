"""
ResNet50 (C2D) for spatiotemporal task. Only ResNet50 backbone structure was implemented here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from models.non_local import NLBlockND


class Bottleneck(nn.Module):
    """
    Bottleneck block structure used in ResNet 50. 
    As mentioned in Section 4. 2D ConvNet baseline (C2D), 
    all convolutions are in essence 2D kernels that prcoess the input frame-by-frame 
    (implemented as (1 x k x k) kernels). 
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, padding=(0, 1, 1), downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=(1, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    """C2D with ResNet 50 backbone.
    The only operation involving the temporal domain are the pooling layer after the second residual block.
    For more details of the structure, refer to Table 1 from the paper. 
    Padding was added accordingly to match the correct dimensionality.
    """
    def __init__(self, block, layers, num_classes=400, non_local=False):
        self.inplanes = 64
        super(ResNet3D, self).__init__()
        
        # first convolution operation has essentially 2D kernels
        # output: 64 x 16 x 112 x 112
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=2, padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # output: 64 x 8 x 56 x 56
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2)
        
        # output: 256 x 8 x 56 x 56
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, d_padding=0)
        
        # pooling on temporal domain
        # output: 256 x 4 x 56 x 56
        self.pool_t = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1))
        
        # output: 512 x 4 x 28 x 28
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, padding=(2, 1, 1))
        
        # add one non-local block here
        # output: 1024 x 4 x 14 x 14
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, padding=(2, 1, 1), non_local=non_local)

        # output: 2048 x 4 x 7 x 7
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, padding=(2, 1, 1))
        
        # output: 2048 x 1
        self.avgpool = nn.AvgPool3d(kernel_size=(4, 7, 7))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, padding=(0, 1, 1), d_padding=(2, 0, 0), non_local=False):
        downsample = nn.Sequential(
                            nn.Conv3d(self.inplanes, planes * block.expansion, 
                                      kernel_size=1, stride=stride, padding=d_padding, bias=False), 
                            nn.BatchNorm3d(planes * block.expansion)
                        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, padding, downsample))
        self.inplanes = planes * block.expansion
        
        last_idx = blocks
        if non_local:
            last_idx = blocks - 1
            
        for i in range(1, last_idx):
            layers.append(block(self.inplanes, planes))
        
        # add non-local block here
        if non_local:
            layers.append(NLBlockND(in_channels=1024, dimension=3))
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.pool_t(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet3D50(non_local=False, **kwargs):
    """Constructs a C2D ResNet-50 model.
    """
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], non_local=non_local, **kwargs)
    return model



if __name__=='__main__':
    # Test case of 32 frames (224 x 224 x 3) input of batch size 1
    img = Variable(torch.randn(1, 3, 32, 224, 224))
    net = resnet3D50(non_local=True)
    count = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count += 1
            print(name)
    print (count)
    out = net(img)
    print(out.size())
