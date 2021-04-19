import os
from PIL import Image
import numpy as np

from dorn.model.network.dorn import DORN


def read_img_gt_path(file_path, in_dir, gt_dir):
    '''
        Read paths to pairs of (img,gt)
        @partameters:
            file_path: path to the txt file including filanames.
            in_dir: directory of input images.
            gt_dir: directory of gt depths.
        @return: a list of (img_path, gt_path)
    '''
    img_gt_paths = []
    
    f_append = img_gt_paths.append
    
    path_join = os.path.join
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            
            im_path = path_join(in_dir, line.split()[0])
            gt_path = path_join(gt_dir, line.split()[1])

            f_append((im_path, gt_path))

    return img_gt_paths

def read_sample(img_gt_paths):
    '''
        read input image and target depth map
    '''
    
    img_path, gt_path = img_gt_paths
    
    gt_depth = np.load(gt_path)

    img_input = Image.open(img_path)

    return img_input, gt_depth


def set_lr_param(lr):
    '''
        set learning rates of parameters in different modules of the model.
        @parameter:
         lr: default learning rate.
        @return: a list of dictionaries. Each of them defines a separate
        parameter group, and should contain with 'param' and lr' keys.
    '''
    
    depth_net = DORN()
    
    name_params = depth_net.named_parameters()
    
    names_params = iter(name_params)
    
    frozen_params = []
    lr1x_params = []
    lr10x_params = []
    lr20x_params = []
    
    for name,param in names_params:
        
        if ('bn' in name):
            frozen_params.append(param)
        elif ('feat_extractor' in name):
            if ('layer1' in name):
                frozen_params.append(param)
            elif ('layer2' in name):
                lr1x_params.append(param)
            elif ('layer3' in name):
                lr1x_params.append(param) 
            elif ('layer4' in name):      
                lr1x_params.append(param)
            else:
                frozen_params.append(param)
        elif( 'scu' in name):
            if ('weight' in name):
                lr10x_params.append(param)
            elif ('bias' in name):
                lr20x_params.append(param)
                
        training_params= [{'params': frozen_params,'lr':0},
                      {'params': lr1x_params, 'lr':1*lr},
                      {'params': lr10x_params, 'lr':10*lr}, 
                      {'params': lr20x_params, 'lr':20*lr}
                     ]
    
    return training_params