#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Group Normalization" Implementation
20193640 Jungwon Choi
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from model.GroupNorm import GroupNorm2d

#===============================================================================
''' 2D Norm Selector '''
class Norm2d():
    def __init__(self, input_size, norm_type='BN', num_groups=None, init_value=1):
        self.input_size = input_size
        self.norm_type = norm_type
        self.num_groups = num_groups
        self.init_value = init_value

    def get(self):
        if self.norm_type == 'BN':
            return nn.BatchNorm2d(self.input_size)
        elif self.norm_type == 'LN':
            # return nn.LayerNorm(self.input_size)
            return GroupNorm2d(self.input_size, self.input_size) # same as LN
        elif self.norm_type == 'IN':
            return nn.InstanceNorm2d(self.input_size)
        elif self.norm_type == 'GN':
            if self.num_groups == None:
                assert False, "Assign the number of groups!"
            # return nn.GroupNorm(self.num_groups, self.input_size)
            return GroupNorm2d(self.input_size, self.num_groups,
                                            init_value=self.init_value)
        else:
            assert False, "Input Proper Norm Function"

#===============================================================================
''' BasicBlock Architecture '''
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, input_size, output_size, stride=1, norm_type='BN', num_groups=None):
        super(BasicBlock, self).__init__()

        # Basicblock layers (refer: https://github.com/facebook/fb.resnet.torch)
        self.conv1 = nn.Conv2d(input_size, output_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n1 = Norm2d(output_size, norm_type=norm_type, num_groups=num_groups).get()
        self.conv2 = nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.n2 = Norm2d(output_size, norm_type=norm_type, num_groups=num_groups, init_value=0).get()
        # Each residual block’s last normalization layer where we initialize γ by 0

        # Identity shortcut
        self.shortcut = nn.Sequential()

        # Projection shortcut (size matching)
        if stride != 1 or input_size != self.expansion*output_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_size, self.expansion*output_size, kernel_size=1, stride=stride, bias=False),
                Norm2d(self.expansion*output_size, norm_type=norm_type, num_groups=num_groups).get()
            )

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

#===============================================================================
''' Bottleneck Architecture '''
class Bottleneck(nn.Module):
    expansion = 4  # 64 -> 256
    def __init__(self, input_size, output_size, stride=1, norm_type='BN', num_groups=None):
        super(Bottleneck, self).__init__()

        # Bottlenect layers (refer: https://github.com/facebook/fb.resnet.torch)
        self.conv1 = nn.Conv2d(input_size, output_size, kernel_size=1, bias=False)
        self.n1 = Norm2d(output_size, norm_type=norm_type, num_groups=num_groups).get()
        self.conv2 = nn.Conv2d(output_size, output_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n2 = Norm2d(output_size, norm_type=norm_type, num_groups=num_groups).get()
        self.conv3 = nn.Conv2d(output_size, self.expansion*output_size, kernel_size=1, bias=False)
        self.n3 = Norm2d(self.expansion*output_size, norm_type=norm_type, num_groups=num_groups, init_value=0).get()
        # Each residual block’s last normalization layer where we initialize γ by 0

        # Identity shortcut
        self.shortcut = nn.Sequential()

        # Projection shortcut (size matching)
        if stride != 1 or input_size != self.expansion*output_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_size, self.expansion*output_size, kernel_size=1, stride=stride, bias=False),
                Norm2d(self.expansion*output_size, norm_type=norm_type, num_groups=num_groups).get()
            )

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = F.relu(self.n2(self.conv2(out)))
        out = self.n3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

#===============================================================================
''' ResNet Architecture for CIFAR-10 '''
class ResNet(nn.Module):
    def __init__(self, block, num_blocks=None, depth=None, norm_type='BN', num_groups=None):
        super(ResNet, self).__init__() # init nn.Module
        self.norm_type=norm_type
        self.num_groups=num_groups

        # Layer info reference: https://github.com/facebook/fb.resnet.torch

        # For CIFAR-10
        if num_blocks == None and depth != None:
            self.dataset = 'cifar10'
            self.input_size = 16
            self.num_classes = 10
            assert (depth - 2) % 6 == 0, "depth should be one of 56, 110"
            num_blocks = int((depth - 2) / 6)

            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.n1 = Norm2d(16, norm_type=self.norm_type, num_groups=self.num_groups).get()
            self.layer1 = self._make_layer(block, 16, num_blocks, stride=1)
            self.layer2 = self._make_layer(block, 32, num_blocks, stride=2)
            self.layer3 = self._make_layer(block, 64, num_blocks, stride=2)
            self.linear = nn.Linear(64, self.num_classes)

            # Layer weight initialization
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d):
                    # He initialization for conv
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(layer, Norm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

            # Initialize γ by 0 for each block's last norm layer
            for layer in self.modules():
                if isinstance(layer, BasicBlock):
                    if layer.n2.weight is not None:
                        nn.init.constant_(layer.n2.weight, 0)

        # For Imagenet
        elif num_blocks != None and depth == None:
            self.dataset = 'imagenet'
            self.input_size = 64
            self.num_classes = 1000

            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.n1 = Norm2d(64, norm_type=self.norm_type, num_groups=self.num_groups).get()
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512*block.expansion, self.num_classes)

            # Layer weight initialization
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d):
                    # He initialization for conv
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(layer, Norm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

            # Initialize γ by 0 for each block's last norm layer
            for layer in self.modules():
                if isinstance(layer, Bottleneck):
                    if layer.n3.weight is not None:
                        nn.init.constant_(layer.n3.weight, 0)

        else:
            assert False, "Please choose proper num_blocks or depth."

    # Auto layer maker by number of blocks
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.input_size, planes, stride,
                        norm_type=self.norm_type, num_groups=self.num_groups))
            self.input_size = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataset == 'cifar10':
            out = F.relu(self.n1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.avg_pool2d(out, kernel_size=8, stride=1)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

        elif self.dataset == 'imagenet':
            out = F.relu(self.n1(self.conv1(x)))
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, kernel_size=7, stride=1)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

        else:
            assert False, "Invalid Network"

#===============================================================================
# ResNet 56, 110 for CIFAR-10
def ResNet56(norm_type='BN', num_groups=None):
    return ResNet(BasicBlock, depth=56,
                    norm_type=norm_type, num_groups=num_groups)
def ResNet110(norm_type='BN', num_groups=None):
    return ResNet(BasicBlock, depth=110,
                    norm_type=norm_type, num_groups=num_groups)

#===============================================================================
# ResNet 50, 101 for Imagenet
def ResNet50(norm_type='BN', num_groups=None):
    return ResNet(Bottleneck, num_blocks=[3,4,6,3],
                    norm_type=norm_type, num_groups=num_groups)
def ResNet101(norm_type='BN', num_groups=None):
    return ResNet(Bottleneck, num_blocks=[3,4,23,3],
                    norm_type=norm_type, num_groups=num_groups)

#===============================================================================
''' Test ResNet '''
if __name__ == '__main__':
    # For CIFAR-10
    net = ResNet56('BN')
    inputs = torch.randn(32,3,32,32)
    outputs = net(inputs)
    print(outputs.size())

    net = ResNet56('GN', 16)
    inputs = torch.randn(32,3,32,32)
    outputs = net(inputs)
    print(outputs.size())

    net = ResNet110('LN') # ('GN', 32)
    inputs = torch.randn(32,3,32,32)
    outputs = net(inputs)
    print(outputs.size())

    net = ResNet110('IN') # ('GN', 1)
    inputs = torch.randn(32,3,32,32)
    outputs = net(inputs)
    print(outputs.size())

    # For Imagenet (Not Experimented)
    net = ResNet50('BN')
    inputs = torch.randn(32,3,224,224)
    outputs = net(inputs)
    print(outputs.size())

    net = ResNet101('GN', 16)
    inputs = torch.randn(32,3,224,224)
    outputs = net(inputs)
    print(outputs.size())
