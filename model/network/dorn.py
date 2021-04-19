import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from dorn.model import utils 
from dorn.model.network.backbone import resnet101
                        
config = utils.read_config()

class FullImageEncoder(nn.Module):
    '''
        captures global texture information as monocular depth cue.
    '''
    
    def __init__(self):
        
        super().__init__()
        
        drop_prob = config['TRAIN'].getfloat('dropout_prob')
        
        self.global_pooling = nn.AvgPool2d(16, 16, ceil_mode=True)  

        self.global_dropout = nn.Dropout2d(p=drop_prob)
        
        self.global_fc = nn.Linear(2048 * 4 * 5, 512) 
         
        self.conv_depth = nn.Conv2d(512, 512, 1) 
        
        self.interp = nn.UpsamplingBilinear2d((49, 65))
        
    def forward(self, x):
        
        # # reduce spatial dimensions to preserve memory
        x = self.global_pooling(x)
        
        x = self.global_dropout(x)
        
        # flatten the tensor for fc layer
        x = x.view(-1, 2048 * 4 * 5)  
        
        x = F.relu(self.global_fc(x))
        
        # reshape to 1x1 feature maps with 512 channels
        x = x.view(-1, 512, 1, 1)
        
        x = F.relu(self.conv_depth(x))
        
        # copy output of conv_depth to each pixel
        out = self.interp(x)
        
        return out

class SceneUnderstandingModule(nn.Module):
    '''
        This module consisting of three parallel components:
        atrous spatial pyramid pooling (aspp), cross-channel 
        learner, and a full image encoder.
    '''
    
    def __init__(self):
        
        super().__init__()
        
        drop_prob = config['TRAIN'].getfloat('dropout_prob')
        
        # capture context information from whole image
        self.encoder = FullImageEncoder()
        
        # cross channel learner
        self.aspp1 = nn.Sequential(OrderedDict([('conv',nn.Conv2d(2048, 512, 1)),
                                                ('relu_1',nn.ReLU(inplace=True)),
                                                ('depth/conv', nn.Conv2d(512, 512, 1)),
                                                ('relu_2', nn.ReLU(inplace=True))]
                                             )
                                  )
        
        # this component extracts multi-scale feature maps                       
        self.aspp2 = nn.Sequential(OrderedDict([('conv',nn.Conv2d(2048, 512, 3, padding=6, dilation=6)),
                                               ('relu_1', nn.ReLU(inplace=True)),
                                               ('depth/conv',nn.Conv2d(512, 512, 1)),
                                               ('relu_2', nn.ReLU(inplace=True))]
                                             )
                                  )
        
        # the following layers extracts multi-scale features
        self.aspp3 = nn.Sequential(OrderedDict([('conv',nn.Conv2d(2048, 512, 3, padding=12, dilation=12)),
                                               ('relu_1',nn.ReLU(inplace=True)),
                                               ('depth/conv', nn.Conv2d(512, 512, 1)),
                                               ('relu_2',nn.ReLU(inplace=True))]
                                             )
                                  )
        
        self.aspp4 = nn.Sequential(OrderedDict([('conv',nn.Conv2d(2048, 512, 3, padding=18, dilation=18)),
                                                ('relu_1', nn.ReLU(inplace=True)),
                                                ('depth/conv', nn.Conv2d(512, 512, 1)),
                                                ('relu_2',nn.ReLU(inplace=True))]
                                             )
                                  )
        
        self.concat_process = nn.Sequential(OrderedDict([('drop/concat',nn.Dropout2d(p=drop_prob)),
                                                        ('conv_1',nn.Conv2d(512 * 5, 2048, 1)),
                                                        ('relu',nn.ReLU(inplace=True)),
                                                        ('drop/conv', nn.Dropout2d(p=drop_prob)),
                                                        ('conv_2', nn.Conv2d(2048, 142, 1))]
                                                     )
                                           )
        
        # upsample spatial dimensions to the given size
        self.zoom = nn.UpsamplingBilinear2d(size=(385, 513))

    def forward(self, x):
        
        x1 = self.encoder(x)

        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)

        # comprehensive understanding of the input
        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        x7 = self.concat_process(x6)
        
        out = self.zoom(x7)
        
        return out
    
class OrdinalRegressionLayer(nn.Module):
    
    def __init__(self):

        super().__init__()

    def forward(self, x):
        """
            Compute ordinal labels and probabilities of each pixel.
        """
        # ordinal outputs for 2k and 2k+1 where k=0,...,K-1
        ord_2k0 = torch.unsqueeze(x[:, 0::2, :, :].clone(), dim=1)
        ord_2k1 = torch.unsqueeze(x[:, 1::2, :, :].clone(), dim=1)
        
        logits = torch.cat((ord_2k0, ord_2k1), dim=1)
        
        ord_probs = F.softmax(logits, dim=1)[:,1,...]
        
        # if \cap l(h,w) = sum_k={0,K-1} Pk(h,w)>0.5 see (5) in DORN
        ord_labels = torch.floor(ord_probs + 0.5).sum(dim=1, keepdim=True)
        
        return ord_labels, ord_probs
    
class DORN(nn.Module):
    
    def __init__(self):
        
        super().__init__()
    
        self.feat_extractor = resnet101()
        
        self.scu_module = SceneUnderstandingModule()
        
        self.orl_layer = OrdinalRegressionLayer()
        
    def forward(self, x):
        
        back_feat = self.feat_extractor(x)
    
        scu_feat = self.scu_module(back_feat)
    
        return self.orl_layer(scu_feat)
        
if __name__ == "__main__":
    
    dorn = DORN()
    
    print(dorn)
    
    path_model = config['MODEL']['kitti']
    
    model_dict = utils.get_model(path_model)
    
    # load the pretrained model
    dorn.load_state_dict(model_dict)
    
    tb = utils.graph_visualize(dorn, config)
    
    tb.close()
    
    print('network was generated!')

    
