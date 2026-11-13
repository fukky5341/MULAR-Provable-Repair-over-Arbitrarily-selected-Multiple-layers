
'''Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA.operators import GELU


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, act, stride=1):
        super(BasicBlock, self).__init__()
        self.act = {
            'sig': nn.Sigmoid(),
            'tan': nn.Tanh(),
            'leaky': nn.LeakyReLU(),
            'gelu': GELU(),
            'silu': nn.SiLU()
        }[act]
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1
                                                                        # , bias=False
                                                                        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, 
                            #    bias=False
                               )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride
                        #   , bias=False
                          ),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.show = False
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # out = self.act(out)
        return out



class ResNet_otheract(nn.Module):
    def __init__(self, block, num_blocks, act, num_classes=10):
        super(ResNet_otheract, self).__init__()
        self.in_planes = 64
        self.act = {
            'sig': nn.Sigmoid(),
            'tan': nn.Tanh(),
            'leaky': nn.LeakyReLU(),
            'gelu': GELU(),
            'silu': nn.SiLU()
        }[act]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], act, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], act, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], act, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], act, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, act, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, act, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.layer1[0](out))
        out = self.act(self.layer1[1](out))
        out = self.act(self.layer2[0](out))
        out = self.act(self.layer2[1](out))
        out = self.act(self.layer3[0](out))
        out = self.act(self.layer3[1](out))
        out = self.act(self.layer4[0](out))
        out = self.act(self.layer4[1](out))
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out
    
    def split(self):
        return nn.Sequential(self.conv1, self.bn1, nn.ReLU(), 
                             self.layer1[0], nn.ReLU(), self.layer1[1], nn.ReLU(), 
                             self.layer2[0], nn.ReLU(), self.layer2[1], nn.ReLU(), 
                             self.layer3[0], nn.ReLU(), self.layer3[1], nn.ReLU(), 
                             self.layer4[0], nn.ReLU(), self.layer4[1] , 
                            #  self.layer1,
                            #  self.layer2, 
                            #  self.layer3, 
                            #  self.layer4, 
                            #  , nn.ReLU(), self.layer3, nn.ReLU(), self.layer4,
            ), nn.Sequential(
                # self.layer4[1],
              nn.ReLU(),
            self.avgpool,
            self.flatten,
            self.linear
            )





def ResNet18_otheract(act, classes=10):
    return ResNet_otheract(BasicBlock, [2, 2, 2, 2], act, num_classes=classes)

