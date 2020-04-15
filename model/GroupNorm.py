#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Group Normalization" Implementation
20193640 Jungwon Choi
'''
import torch
import torch.nn as nn

#===============================================================================
''' Group normalization implementation '''
class GroupNorm2d(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5, init_value=1):
        super(GroupNorm2d, self).__init__() # init nn.Module
        self.num_groups = num_groups
        self.eps = eps # preventing division of 0

        # Parameter initialization
        # gamma
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1)*init_value)
        # beta
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups

        # Check group division error
        assert C % G == 0, \
        "The number of channels should be divisible for the number of groups."

        # Calculate mean and variance of each group
        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        # Normalize each group
        x = (x-mean) / (var+self.eps).sqrt()

        # Reshape tensor for output
        x = x.view(N, C, H, W)

        # Linear transform per channel
        return x * self.weight + self.bias

#===============================================================================
''' Test group normalization '''
if __name__ == '__main__':
    inputs = torch.randn(32, 64, 32, 32)
    gn = GroupNorm2d(64, 16)
    outputs = gn(inputs)
    print(outputs.size())
