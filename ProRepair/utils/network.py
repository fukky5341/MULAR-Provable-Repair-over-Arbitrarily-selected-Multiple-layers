
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA.operators import GELU


class FNN(nn.Module):

    def __init__(self):
        super(FNN, self).__init__()

        self.classifier = nn.Sequential(
                    nn.Linear(5, 50),
                    nn.ReLU(),
                    nn.Linear(50, 50),
                    nn.ReLU(),
                    nn.Linear(50, 50),
                    nn.ReLU(),
                    nn.Linear(50, 50),
                    nn.ReLU(),
                    nn.Linear(50, 50),
                    nn.ReLU(),
                    nn.Linear(50, 50),
                    nn.ReLU(),
                    nn.Linear(50, 5)
        )

    def forward(self, x):

        output = self.classifier(x)
        return output

    def split(self, ind=11):
        return nn.Sequential(self.classifier[0: ind]), nn.Sequential(self.classifier[ind:])


cfg = {
    'CNN8': [64, 'M', 128, 'M', 256, 'M', 512, 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class FNN_3_100(nn.Module):
    def __init__(self):
        super(FNN_3_100,self).__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(nn.Linear(28*28, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 10),
        )

    def forward(self, x, all=False, act=False):
        x = self.flatten(x)
        if not all and not act:
            return self.classifier(x)
        elif all and (not act):
            intermediate_outputs = []
            for layer in self.classifier:
                x = layer(x)
                # if isinstance(layer, nn.ReLU):
                #     continue
                intermediate_outputs.append(x)
            return intermediate_outputs
        
    def split(self, ind=3):
        return nn.Sequential(nn.Flatten(), self.classifier[0: ind]), nn.Sequential(self.classifier[ind:])


class FNN_3_100_gelu(nn.Module):
    def __init__(self):
        super(FNN_3_100_gelu,self).__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(nn.Linear(28*28, 100),
                                        GELU(),
                                        nn.Linear(100, 100),
                                        GELU(),
                                        nn.Linear(100, 10),
        )

    def forward(self, x, all=False, act=False):
        x = self.flatten(x)
        return self.classifier(x)

    def split(self, ind=3):
        return nn.Sequential(nn.Flatten(), self.classifier[0: ind]), nn.Sequential(self.classifier[ind:])
    

class FNN_6_100_gelu(nn.Module):
    def __init__(self):
        super(FNN_6_100_gelu,self).__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(nn.Linear(28*28, 100),
                                        GELU(),
                                        nn.Linear(100, 100),
                                        GELU(),
                                        nn.Linear(100, 100),
                                        GELU(),
                                        nn.Linear(100, 100),
                                        GELU(),
                                        nn.Linear(100, 100),
                                        GELU(),
                                        nn.Linear(100, 10),
        )

    def forward(self, x, all=False, act=False):
        x = self.flatten(x)
        return self.classifier(x)

    def split(self, ind=7):
        return nn.Sequential(nn.Flatten(), self.classifier[0: ind]), nn.Sequential(self.classifier[ind:])

 
class FNN_9_100_gelu(nn.Module):
    def __init__(self):
        super(FNN_9_100_gelu,self).__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(nn.Linear(28*28, 100),
                                        GELU(),
                                        nn.Linear(100, 100),
                                        GELU(),
                                        nn.Linear(100, 100),
                                        GELU(),
                                        nn.Linear(100, 100),
                                        GELU(),
                                        nn.Linear(100, 100),
                                        GELU(),
                                        nn.Linear(100, 100),
                                        GELU(),
                                        nn.Linear(100, 100),
                                        GELU(),
                                        nn.Linear(100, 100),
                                        GELU(),
                                        nn.Linear(100, 10),
        )

    def forward(self, x, all=False, act=False):
        x = self.flatten(x)
        return self.classifier(x)

    def split(self, ind=15):
        return nn.Sequential(nn.Flatten(), self.classifier[0: ind]), nn.Sequential(self.classifier[ind:])


class FNN_9_200_gelu(nn.Module):
    def __init__(self):
        super(FNN_9_200_gelu,self).__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(nn.Linear(28*28, 200),
                                        GELU(),
                                        nn.Linear(200, 200),
                                        GELU(),
                                        nn.Linear(200, 200),
                                        GELU(),
                                        nn.Linear(200, 200),
                                        GELU(),
                                        nn.Linear(200, 200),
                                        GELU(),
                                        nn.Linear(200, 200),
                                        GELU(),
                                        nn.Linear(200, 200),
                                        GELU(),
                                        nn.Linear(200, 200),
                                        GELU(),
                                        nn.Linear(200, 10),
        )

    def forward(self, x, all=False, act=False):
        x = self.flatten(x)
        return self.classifier(x)

    def split(self, ind=15):
        return nn.Sequential(nn.Flatten(), self.classifier[0: ind]), nn.Sequential(self.classifier[ind:])


class FNN_6_100(nn.Module):
    # define the structure of the network
    def __init__(self):
        super(FNN_6_100,self).__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(nn.Linear(28*28, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 10),
        )

    def forward(self, x, all=False, act=False):
        x = self.flatten(x)
        if not all and not act:
            return self.classifier(x)
        elif all and (not act):
            intermediate_outputs = []
            for layer in self.classifier:
                x = layer(x)
                # if isinstance(layer, nn.ReLU):
                #     continue
                intermediate_outputs.append(x)
            return intermediate_outputs

    def split(self, ind=7):
        return nn.Sequential(nn.Flatten(), self.classifier[0: ind]), nn.Sequential(self.classifier[ind:])


class FNN_9_100(nn.Module):
    # define the structure of the network
    def __init__(self):
        super(FNN_9_100,self).__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(nn.Linear(28*28, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 10),
        )

    def forward(self, x, all=False, act=False):
        x = self.flatten(x)
        if not all and not act:
            return self.classifier(x)
        elif all and (not act):
            intermediate_outputs = []
            for layer in self.classifier:
                x = layer(x)
                # if isinstance(layer, nn.ReLU):
                #     continue
                intermediate_outputs.append(x)
            return intermediate_outputs

    def split(self, ind=13):
        return nn.Sequential(nn.Flatten(), self.classifier[0: ind]), nn.Sequential(self.classifier[ind:])


class FNN_9_200(nn.Module):
    def __init__(self):
        super(FNN_9_200,self).__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(nn.Linear(28*28, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, 10),
        )

    def forward(self, x, all=False, act=False):
        x = self.flatten(x)
        if not all and not act:
            return self.classifier(x)
        elif all and (not act):
            intermediate_outputs = []
            for layer in self.classifier:
                x = layer(x)
                # if isinstance(layer, nn.ReLU):
                #     continue
                intermediate_outputs.append(x)
            return intermediate_outputs
    
    def split(self, ind=11):
        return nn.Sequential(nn.Flatten(), self.classifier[0: ind]), nn.Sequential(self.classifier[ind:])

class CNN_small(nn.Module):
    def __init__(self, nclasses=10):
        super(CNN_small,self).__init__()
        self.network = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=7, stride=5, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(288, 100),
                nn.ReLU(),
                nn.Linear(100, nclasses),
        )

    def forward(self, x):
        out = self.network(x)
        return out
    
    def split(self, ind=None):
        return nn.Sequential(*self.network[0: -2]), nn.Sequential(*self.network[-2:])
    
class CNN4(nn.Module):
    def __init__(self):
        super(CNN4, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=0),
            nn.ReLU(), 
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),# output: 32 x 6 x 6
            nn.Flatten(), 
            nn.Linear(1152, 100),
            nn.ReLU(),
            nn.Linear(100, 10))

    def forward(self, x):
        out = self.network(x)
        return out

    def split(self, ind=None):
        return nn.Sequential(*self.network[0: -2]), nn.Sequential(*self.network[-2:])


class CNN6(nn.Module):
    def __init__(self):
        super(CNN6, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2), # output: 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2), # output: 8 x 8
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2), # output: 4 x 4
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2), # output: 2 x 2

            nn.Flatten(), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10))

    def forward(self, x):
        out = self.network(x)
        return out

    def split(self, ind=14):
        return nn.Sequential(*self.network[0: ind]), nn.Sequential(*self.network[ind:])


class CNN8(nn.Module):
    def __init__(self):
        super(CNN8, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), # output: 32 x 16 x 16

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 8 x 8

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 4 x 4

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),         
            nn.MaxPool2d(2, 2), # output: 256 x 2 x 2

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 512 x 1 x 1

            nn.Flatten(), 
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, x):
        out = self.network(x)
        return out

    def split(self, ind=21):
        return nn.Sequential(*self.network[0: ind]), nn.Sequential(*self.network[ind:])


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.network = nn.Sequential(
        nn.Conv2d(3, 6, (5, 5), stride=1, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=(2, 2), stride=2, padding=0),

        nn.Conv2d(6, 16, (5, 5), stride=1, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=(2, 2), stride=2, padding=0),
        
        nn.Flatten(),
        nn.Linear(16 * 8 * 8, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10),
        )

    def forward(self, x):
        out = self.network(x)
        return out

    def split(self, ind=10):
        return nn.Sequential(*self.network[0: ind]), nn.Sequential(*self.network[ind:])


class CNN8_sig(nn.Module):
    def __init__(self):
        super(CNN8_sig, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Sigmoid(), # output: 32 x 16 x 16

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # output: 64 x 8 x 8

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # output: 128 x 4 x 4

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Sigmoid(),         
            nn.MaxPool2d(2, 2), # output: 256 x 2 x 2

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # output: 512 x 1 x 1

            nn.Flatten(), 
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 10))

    def forward(self, x):
        out = self.network(x)
        return out

    def split(self, ind=21):
        return nn.Sequential(*self.network[0: ind]), nn.Sequential(*self.network[ind:])



class VGG(nn.Module):
    def __init__(self, vgg_name, classes=10, bn=True):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], bn)
        self.linear1 = nn.Linear(512, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, classes)
        self.act = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.features(x)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.act(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, bn):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif bn:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        # convert to nn.adaptive_avg_pool2d
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)
    
    def split(self):
        return nn.Sequential(*self.features, nn.Flatten(), self.linear1,), nn.Sequential(
                             self.act, 
                             self.linear2, 
                             self.act,
                             self.classifier
                            )


class VGG_otheract(nn.Module):
    def __init__(self, vgg_name, act, classes=10, bn=True):
        super(VGG_otheract, self).__init__()
        self.act = {
            'sig': nn.Sigmoid(),
            'tan': nn.Tanh(),
            'leaky': nn.LeakyReLU(),
            'gelu': GELU(),
            'silu': nn.SiLU()
        }[act]
        self.features = self._make_layers(cfg[vgg_name], bn)
        self.linear1 = nn.Linear(512, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.features(x)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.act(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, bn):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif bn:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           self.act]
                in_channels = x
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           self.act]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)
    
    def split(self):
        return nn.Sequential(*self.features, nn.Flatten(), self.linear1, self.act, self.linear2, ), nn.Sequential(

                             self.act,
                             self.classifier
                            )



class VGG16_img(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='VGG16'):
        super(VGG16_img, self).__init__()
        self.in_channels = 3
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:10])
        self.features4 = self._make_layers(cfg[vgg_name][10:14])
        self.features5 = self._make_layers(cfg[vgg_name][14:])
        self.dense1 = nn.Linear(25088, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x, feature=False):
        f1_ = self.features1[0:2](x)
        f1 = self.features1[2:](f1_)
        f2_ = self.features2[0:2](f1)
        f2 = self.features2[2:](f2_)
        f3_ = self.features3[0:3](f2)
        f3 = self.features3[3:](f3_)
        f4_ = self.features4[0:3](f3)
        f4 = self.features4[3:](f4_)
        f5_ = self.features5[0:3](f4)
        f5 = self.features5[3:](f5_)
        f5 = f5.view(f5.size(0), -1)
        d1 = F.relu(self.dense1(f5))
        d2 = F.relu(self.dense2(d1))
        out = d2.view(d2.size(0), -1)
        out = self.classifier(out)
        return out



    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def split(self, ind=None):
        return nn.Sequential(self.features1, self.features2, self.features3, self.features4, self.features5, nn.Flatten(), self.dense1), nn.Sequential(
            nn.ReLU(), self.dense2, nn.ReLU(), self.classifier)


class VGG19_img(nn.Module):  # for ai lancet
    def __init__(self, vgg_name='VGG19', num_class=10):
        super(VGG19_img, self).__init__()
        self.in_channels = 3
        # self.features = self._make_layers(cfg[vgg_name])
        self.features1 = self._make_layers(cfg[vgg_name][0:3])
        self.features2 = self._make_layers(cfg[vgg_name][3:6])
        self.features3 = self._make_layers(cfg[vgg_name][6:9])
        self.features4 = self._make_layers(cfg[vgg_name][9:12])
        self.features5 = self._make_layers(cfg[vgg_name][12:])
        self.dense1 = nn.Linear(512 * 7 * 7, 512)
        self.dense2 = nn.Linear(512, 128)
        self.classifier = nn.Linear(128, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        f1 = self.features1(x)
        f2 = self.features2(f1)
        f3 = self.features3(f2)
        f4 = self.features4(f3)
        f5 = self.features5(f4)
        f5 = f5.view(f5.size(0), -1)
        d1 = F.relu(self.dense1(f5))
        d2 = F.relu(self.dense2(d1))
        out = d2.view(f5.size(0), -1)
        out = self.classifier(out)
        return out
    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def split(self):
        return nn.Sequential(self.features1, self.features2, self.features3, self.features4, self.features5, \
            nn.Flatten(), self.dense1,), \
        nn.Sequential(
                             self.act, 
                             self.dense2, 
                             self.act,
                             self.classifier
                            )


def test():
    net = VGG('VGG19')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
