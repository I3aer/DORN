import os
import sys
from PIL import Image
import torch
from configparser import ConfigParser, ExtendedInterpolation

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dir_path = os.path.dirname(os.path.realpath(__file__))

def read_config():
    
    config = ConfigParser(interpolation=ExtendedInterpolation())
    
    config_path = os.path.join(dir_path,'config.ini')
    
    config.read(config_path)
    
    return config

@torch.no_grad()
def transform_img(img, config):

    means = config['IMG_NORM']['RGB_mu']
    means = list(map(float,means.split(',')))
    
    stds = config['IMG_NORM']['RGB_std']
    stds = list(map(float,stds.split(',')))
    
    # conver to tensor. Pixel values are scaled to [0,1]
    my_transforms = [transforms.ToTensor()]
    
    # rescale pixel values to [0,255]
    scale = transforms.Lambda(lambda t: 255*t)
    my_transforms.append(scale)
    
    # normalize image data
    im_norm = transforms.Normalize(means, stds)
    my_transforms.append(im_norm)
    
    # network accepts BGR
    rgb2bgr = transforms.Lambda(lambda t: t[[2,1,0],:,:])
    my_transforms.append(rgb2bgr)
    
    # resize img tensor
    in_w = config['INPUT'].getint('width')
    in_h = config['INPUT'].getint('height')
    im_resize = transforms.Resize(((in_h, in_w)))
    #my_transforms.append(im_resize)
    
    transform = transforms.Compose(my_transforms)
    
    # img_tensor in the shape of [1xCxHXW]
    img_tensor = torch.unsqueeze(transform(img),0)
    
    return img_tensor

def graph_visualize(net, config):
    '''
        Log the network model for visualization.
        @parameters:
         net: network model.
         config: configuration parameters.
        @return: The SummaryWriter that output 
                to ./runs/ directory by default.
    '''
    
    w = config['INPUT'].getint('width')
    h = config['INPUT'].getint('height')
    c = config['INPUT'].getint('channel')
           
    # create an event file        
    tb = SummaryWriter()
    
    # generate dummy input image
    img = 255*torch.rand((1,c,h,w)).to(device)
    
    # add graph and image to tensorboard
    tb.add_graph(net, input_to_model=img)
    
    return tb

def get_model(path_npy=None):
    '''
        get state dict of the model.
    '''
    
    if (path_npy is None):
        print('path cannot be None')
        sys.exit(1)
    
    model_dict = torch.load(path_npy, map_location=device)
    
    return model_dict

def get_depth(labels, config=None):
    """
        compute depth values using the predicted
        ordinal labels for each pixel positions.
        Actually, DORN predicts index of threshold
        t_i where i takes values in [0,...,K-1].
    """
    if not isinstance(labels,torch.Tensor):
        labels = torch.from_numpy(labels)
        
    if config == None:
        config = read_config()
    
    max_depth = config['INPUT'].getfloat('max_depth')
    min_depth = config['INPUT'].getfloat('min_depth')
    sid_bins = config['INPUT'].getint('sid_bins')

    alpha = torch.log(torch.tensor(min_depth, device=device)).detach()
    beta  = torch.log(torch.tensor(max_depth, device=device)).detach()
    K  = torch.tensor(sid_bins, device=device).float().detach()

    # divide log interval [alpha, beta] linearly into K intervals
    # Note that intervals are labeled from 0, then use K - 1 
    resolution = (beta - alpha)/(K-1)

    # depth = exp(label*resolution + alpha) where label={0,...,K}
    depth = torch.exp(labels*resolution + alpha)
    
    return depth.float()


def get_labels(depth, config=None):
    '''
        Discretize depth measurements to ordinal labels.
    '''
    if not isinstance(depth,torch.Tensor):
        depth = torch.from_numpy(depth).to(device)
        
    if config == None:
        config = read_config()

    max_depth = config['INPUT'].getfloat('max_depth')
    min_depth = config['INPUT'].getfloat('min_depth')
    sid_bins = config['INPUT'].getint('sid_bins')
  
    alpha = torch.log(torch.tensor(min_depth, device=device)).detach()
    beta  = torch.log(torch.tensor(max_depth, device=device)).detach()
    K  = torch.tensor(sid_bins, device=device).float().detach()

    # divide log interval [alpha, beta] linearly into K intervals.
    # Note that intervals are labeled from 0, then use K - 1  
    resolution = (beta - alpha)/(K-1)
    
    ord_labels = (torch.log(depth) - alpha)/resolution
    
    return torch.round(ord_labels).int()

class MergeCrops():
    """
       Merge predictions and gt depths of four corner crops.
    """
    
    def __init__(self, config):
        
        in_w = config['INPUT'].getint('width')
        in_h = config['INPUT'].getint('height')
        
        self.img_w = config['NUSC_IMG'].getint('width')
        self.img_h = config['NUSC_IMG'].getint('height')
    
        # width and height indices of the four crops
        self.w_indices = [(0, in_w), (self.img_w - in_w, self.img_w)]
        self.h_indices = [(0, in_h), (self.img_h - in_h, self.img_h)]
        
    def __call__(self, in_tensor, gt_depths):
        
        C = in_tensor.shape[1]
        
        out_tensor = torch.zeros((1, C, self.img_h, self.img_w),\
                     dtype=torch.float32, device=device, requires_grad=False)
        
        depth_map =  torch.zeros((1, 1, self.img_h, self.img_w),\
                     dtype=torch.float32, device=device, requires_grad=False)
        
        # detach counts so no gradient will backpropagate
        counts = torch.zeros((1, 1, self.img_h, self.img_w),\
                 dtype=torch.float32, device=device, requires_grad=False)
        
        idx = 0
        for h0,h1 in self.h_indices:
            for w0,w1 in self.w_indices:
            
                out_tensor[0, :, h0:h1, w0:w1] += in_tensor[idx, ...]
                
                depth_map[0, :, h0:h1, w0:w1] += gt_depths[idx, ...]
            
                counts[0, 0, h0:h1, w0:w1] = counts[0, 0, h0:h1, w0:w1] + 1.
                
                idx += 1
                
        out_tensor = out_tensor / counts
        
        depth_map = depth_map / counts
        
        return out_tensor, depth_map
