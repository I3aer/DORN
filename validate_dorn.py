import os
import glob
import torch
import json
import numpy as np

from functools import partial
from depth_estimator import DORNNET
from model.utils import read_config, device, get_depth
from model.metrics import Evaluate

import torch.multiprocessing as mp


class MergeCrops():
    """
       Merge ordinal probs of four corner crops to obtain
       predicted labels of the whole image.
    """
    
    def __init__(self, config):
        
        in_w = config['INPUT'].getint('width')
        in_h = config['INPUT'].getint('height')
        
        self.img_w = config['NUSC_IMG'].getint('width')
        self.img_h = config['NUSC_IMG'].getint('height')
    
        # width and height indices of the four crops
        self.w_indices = [(0, in_w), (self.img_w - in_w, self.img_w)]
        self.h_indices = [(0, in_h), (self.img_h - in_h, self.img_h)]
        
        self.K = config['INPUT'].getint('sid_bins')
        
    def __call__(self, ord_probs):
        
        pred_probs = torch.zeros((1, self.K, self.img_h, self.img_w),
                                  dtype=torch.float32).to(device)
        
        
        # detach counts so no gradient will backpropagate
        counts = torch.zeros((1, 1, self.img_h, self.img_w), 
                             dtype=torch.float32).to(device).detach()
        
        idx = 0
        for h0,h1 in self.h_indices:
            for w0,w1 in self.w_indices:
            
                pred_probs[0, :, h0:h1, w0:w1] += ord_probs[idx, ...]
                
                counts[0, 0, h0:h1, w0:w1] = counts[0, 0, h0:h1, w0:w1] + 1.
                
                idx += 1
                
        pred_probs = pred_probs / counts
        
        
        return pred_probs

class SaveFile():
    
    config = read_config()
    
    K = config['INPUT'].getint('sid_bins')
    
    min_ = config['INPUT'].getfloat('min_depth')
    max_ = config['INPUT'].getfloat('max_depth')
    
    # depth bins according to SID, see (1) in DORN
    expo = np.arange(0,K+2)/(K+1)
    bins = min_*(max_/min_)**expo
    
    def __init__(self):
        '''
            Save depth predictions for each label to a json file.
            The file is updated with each new prediction from the
            DORN for monocular depth estimation network.
        '''
        pass

    @staticmethod
    def __update_data(l,depths,labels,data):
        '''
            target function called to update json data.
        '''
        mask = labels == l
    
        v = list(depths[mask])
        
        k = str(l)
        
        v = data['labels'][k] + v
        
        out = {k:v}
        
        return out
    
    @torch.no_grad()
    def write(self,predict_depth, target_depth):
        '''
           write predicted values for each target label. 
        '''
        
        mask = torch.logical_and(target_depth > self.min_,
                                 target_depth < self.max_)
        
        predict_depth = predict_depth[mask]
        target_depth  = target_depth[mask]
        
        try:
            with open('dorn_predictions.json','r') as json_file:
                json_data = json.load(json_file)
        except FileNotFoundError:
            json_data = {'labels':{}}
        
        if len(json_data['labels']) == 0:
            json_data['labels'] = {str(k):[] for k in range(self.K+1)}
        
        # transfer tensors onto cpu and then transform to numpy
        pred = predict_depth.to('cpu').float().numpy()
        gt_depth = target_depth.to('cpu').float().numpy()
        
        labels = np.digitize(gt_depth, self.bins) - 1
         
        update_fn = partial(self.__update_data, depths=pred,
                            labels=labels, data=json_data)
    
        outs = [update_fn(l) for l in list(range(self.K+1))]
        
        json_data['labels'] = {list(o.keys())[0]:
                               list(o.values())[0] for o in outs}
       
        
        with open('dorn_predictions.json','w+') as json_file:
            json.dump(json_data['labels'], json_file, indent=2)

if __name__ == '__main__':
    
    mp.freeze_support()
    
    config = read_config()
    
    dataset_dir = config['DATASET']['dataset_dir']
    
    img_dir = config['DATASET']['input_dir']
    gt_dir =  config['DATASET']['gt_dir']
    
    metrics = Evaluate()
    
    dorn_net = DORNNET()
    
    merge_crops = MergeCrops(config)
    
    for mode in ('train','valid'):
        
        img_dir = os.path.join(dataset_dir, mode, img_dir)
        gt_dir = os.path.join(dataset_dir, mode, gt_dir)
    
        itr_img_paths = glob.iglob(img_dir + '/*.png')
        itr_gt_paths = glob.iglob(gt_dir + '/*.npy')
    
        for i, sample in enumerate(zip(itr_img_paths, itr_gt_paths)):
            
            in_img_path, gt_depth_path = sample
            
            pred_labels, ord_probs = dorn_net(in_img_path)
            
            gt_depths = np.load(gt_depth_path)
            
            gt_depths = torch.from_numpy(gt_depths).to(device)
            
            H, W = gt_depths.shape
            
            gt_depths = gt_depths.view(1, 1, H, W)
            
            # merge ordinal probs and gt depts of 4 corner crops
            pred_probs = merge_crops(ord_probs)
            
            # convert predicted probs to labels as in DORN
            pred_labels = torch.sum(pred_probs>0.5,dim=1).view(-1,1,H,W)
    
            pred_depths = get_depth(pred_labels, config)
            
            metrics.compute(pred_depths, gt_depths)
            
            print('counter:{0:}'.format(i))
        
    metrics.results()
    
    print(metrics)
    
    

    

