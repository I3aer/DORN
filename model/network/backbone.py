import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def conv2D_bn_relu(num_in_ch, num_out_ch, stride):
    
        conv = nn.Conv2d(num_in_ch, num_out_ch, kernel_size=3, 
                         stride=stride, padding=1, bias=False)
        
        bn = nn.BatchNorm2d(num_out_ch, momentum=0.05)
      
        return nn.Sequential(OrderedDict([('conv',conv),
                                         ('bn',bn),
                                         ('relu',nn.ReLU())]))

class Blocks(nn.Module):
    '''
        Resnet building blocks.
    '''
    
    expansion = 4

    def __init__(self, num_in_ch, num_out_ch, stride=1, dilation=1, multi_grid=1):
        
        super().__init__()

        
        # 1x1 conv for feature-map pooling
        self.conv1 = nn.Conv2d(num_in_ch, num_out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_out_ch)
        
        # 3x3 conv
        padding = dilation * multi_grid, 
        dilation = dilation * multi_grid,
        self.conv2 = nn.Conv2d(num_out_ch, num_out_ch, kernel_size=3, stride=stride, 
                               padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(num_out_ch)
        
        # 1x1 conv to scale up feature maps
        self.conv3 = nn.Conv2d(num_out_ch, self.expansion*num_out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*num_out_ch)

        # identify mapping if dimensions and channels are same
        self.proj = nn.Sequential()
        
        # projection if expansion in channels/reduction in dimensions 
        if stride != 1 or num_in_ch != self.expansion*num_out_ch:
            
            self.proj = \
            nn.Sequential(OrderedDict([('conv', nn.Conv2d(num_in_ch, self.expansion*num_out_ch,
                                                          kernel_size=1, stride=stride, bias=False)),
                                       ('bn', nn.BatchNorm2d(self.expansion*num_out_ch))])
                         )

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # elem-wise add and relu
        out += self.proj(x)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    '''
        ResNet-101 module.
    '''
    def __init__(self, block, num_blocks):
        
        self.num_in_ch = 128
        
        super().__init__()
        
        # not trainable layers 
        self.conv1_1 = conv2D_bn_relu(3, 64, 2)
        self.conv1_2 = conv2D_bn_relu(64, 64, 1)
        self.conv1_3 = conv2D_bn_relu(64, 128, 1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0]) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) 
        
        # trainable layers
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1, dilation=2) 
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1, dilation=4, multi_grid=(1, 1, 1)) 

    def _make_layer(self, block, num_out_ch, blocks, stride=1, dilation=1, multi_grid=1):
        '''
            construct building blocks of ResNet.
        '''
        
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        
        layers.append(block(self.num_in_ch, num_out_ch, stride, dilation=dilation, multi_grid=generate_multi_grid(0, multi_grid)))
        
        self.num_in_ch = num_out_ch * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.num_in_ch, num_out_ch, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        out = self.conv1_1(x)
        out = self.conv1_2(out)
        out = self.conv1_3(out)
        
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        return out
    
    def set_BN_eval_mode(self):
        '''
            Set BN layers to evaluation mode to use population mean
            and variances with those scale and shift parameters.
        '''
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

def resnet101():
    
    resnet101 = ResNet(Blocks, [3, 4, 23, 3])
    
    # freeze conv1, conv2, conv3, maxpool, block1, and block2 layers
    for l, layers in enumerate(resnet101.children()):
        if l < 6:
            for param in layers.parameters():
                param.requires_grad = False
        else:
            break

    return resnet101

if __name__ == '__main__':
    
    net = resnet101()
