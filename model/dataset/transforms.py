import random
import numpy as np
import torch
from PIL import Image
from itertools import product
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from dorn.model.utils import read_config

# global config parameters
config = read_config()

# initialize random number generator
random.seed()

class Scale():

    def __init__(self):
        '''
            Scale PIL image and npy depth map with the random scale
            factor uniformly sampled in the interval of [1.0, 1.2]
        '''
        
        # random scale factor
        self.scale = random.uniform(1.0, 1.2)

    def __call__(self, img_depth):
    
        img = img_depth['img']
        
        # new_size = (new_w, new_h)
        new_size = [int(self.scale*x) for x in img.size]
        
        img_depth['img'] = img.resize(new_size, Image.BICUBIC)
        
        gt_depth = img_depth['gt']
        
        gt_img= Image.fromarray(gt_depth).resize(new_size, Image.BILINEAR)
        
        # after scaling camera moves scale times closer
        # to a scene, therefore divide depths by scale
        img_depth['gt'] = np.asarray(gt_img)/self.scale
        
        return img_depth

class HorizFlip():
    '''
        Apply horizontal transformation.
    '''

    def __init__(self):
        # random pobability of being flipped
        self.do_flip = random.uniform(0.0, 1.0)
        
    def __call__(self, img_depth):
       
        if self.do_flip>0.5:
            img = img_depth['img'] 
            img_depth['img'] = img.transpose(method=Image.FLIP_LEFT_RIGHT)
            
            gt_depth = img_depth['gt'] 
            img_depth['gt'] = np.flip(gt_depth, axis=1)
            
        return img_depth

class Rotate():
    '''
        Rotate both image and gt with a random angle [deg]
        sampled uniformly in the interval of [-5.0, 5.0]
    '''
    
    def __init__(self):
        # random rotation [deg]
        self.angle = random.uniform(-5.0, 5.0)   
        
    def __call__(self, img_depth):
        
        img = img_depth['img'] 
        img_depth['img'] = img.rotate(self.angle, Image.BICUBIC)
        
        depth_img = Image.fromarray(img_depth['gt'])
        depth_img = depth_img.rotate(self.angle, Image.BILINEAR)
        img_depth['gt'] = np.asarray(depth_img)

        return img_depth
        
class CenterCrop():
    '''    
        Crop a fixed sized region from input image randomly.
    '''
        
    def __init__(self):
        
        w = config['INPUT'].getint('width')
        h = config['INPUT'].getint('height')
        
        self.size = (w,h)
                                      
    def __call__(self,img_depth):
        
        img = img_depth['img']
        
        img_w, img_h = img.size
        
        new_w, new_h = self.size
        
        # left,top coordinate of the random crop
        left = random.randint(0, img_w - new_w)
        top = random.randint(0, img_h - new_h)
        
        # coordinates of the box rectangle
        crop_size = (left, top, left + new_w, top + new_h)
        
        img_depth['img'] = img.crop(crop_size)
        
        gt_depth = img_depth['gt']
        
        img_depth['gt'] = gt_depth[crop_size[1]:crop_size[3], 
                                   crop_size[0]:crop_size[2]]
        
        return img_depth
    
class FourCrops():
    
    def __init__(self):
        
        self.in_w = config['INPUT'].getint('width')
        self.in_h = config['INPUT'].getint('height')
  
        self.crop = transforms.FiveCrop((self.in_h,self.in_w))
        
    def __call__(self,img_depth):
        
        img = torch.unsqueeze(img_depth['img'],0)
        
        # crop the given image into four corners
        img_depth['img'] = torch.vstack(self.crop(img)[0:-1])
        
        img_h, img_w = img.shape[2:]
        
        # width and height indices of the four crops
        w_indices = [(0, self.in_w), (img_w - self.in_w, img_w)]
        h_indices = [(0, self.in_h), (img_h - self.in_h, img_h)]
        
        gt_depth = img_depth['gt']
        
        four_gt_crops = \
        torch.zeros((4,1,self.in_h,self.in_w), dtype=torch.float32)
        
        for i,hw in enumerate(product(h_indices, w_indices)):
            h0, h1 = hw[0]
            w0, w1 = hw[1]
            four_gt_crops[i,0,:,:] = gt_depth[0, h0:h1, w0:w1]
        
        img_depth['gt'] = four_gt_crops
        
        return img_depth

class ColorJittering():
    '''
        Add some small noise to image pixels.
    '''

    def __init__(self):
        pass
        
    @staticmethod
    def adjust_pil(pil):
        
        if not isinstance(pil, Image.Image):
            raise TypeError('input to ColorJittering must be PIL image')
    
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)
        saturation = random.uniform(0.9, 1.1)

        pil = F.adjust_brightness(pil, brightness)
        pil = F.adjust_contrast(pil, contrast)
        pil = F.adjust_saturation(pil, saturation)

        return pil

    def __call__(self, img_depth):
        
        img = img_depth['img']
        img_depth['img'] = self.adjust_pil(img)

        return img_depth
    
    
class ToTensor():
    '''
        Convert each sample (PIL,npy) to pytorch.tensor
    '''
    
    def __init__(self):
        pass
    
    def __call__(self,img_depth):
        img = img_depth['img']
        img_depth['img'] = 255*transforms.ToTensor()(img)
        
        gt_depth = img_depth['gt']
        gt_depth_tensor = torch.from_numpy(gt_depth.copy())
        img_depth['gt'] = torch.unsqueeze(gt_depth_tensor,0)
        
        return img_depth
        
class ImNormalize():
    '''
        Normalize PIL image using training set mean and std. 
    '''

    def __init__(self):
        
        means = config['IMG_NORM']['RGB_mu']
        means = list(map(float,means.split(',')))
    
        stds = config['IMG_NORM']['RGB_std']
        stds = list(map(float,stds.split(',')))
        
        self.normalize = transforms.Normalize(means, stds)

    def __call__(self, img_depth):
        
        img = img_depth['img']
        
        img_depth['img'] = self.normalize(img)

        return img_depth
    
class ValidDepth():
    '''
        Validate depth map by masking out invalid depth values.
    '''
    
    def __init__(self):
        
        self.min = config['INPUT'].getfloat('min_depth')
        
        self.max = config['INPUT'].getfloat('max_depth')
                                                 
    def __call__(self, img_depth):
         
        gt_depth = img_depth['gt'].copy()
        
        mask = np.logical_or(gt_depth<self.min, gt_depth>self.max)
        
        gt_depth[mask] = 0
        
        img_depth['gt'] = gt_depth 
        
        return img_depth
    
class RGB2BGR():
    '''
        Convert RGB channels to BGR channels.
    '''
    
    def __init__(self):
        pass
    
    def __call__(self,img_depth):    
        
        img = img_depth['img']
        img_depth['img'] = img[[2,1,0],:,:]
        
        return img_depth
    
class ImResize():
    
    def __init__(self):
   
        # input image dimension
        in_w = config['INPUT'].getint('width')
        in_h = config['INPUT'].getint('height')
        
        self.im_resize = transforms.Resize(in_h, in_w)

    def __call__(self,img_depth):
        img = img_depth['img']
        
        img_depth['img'] = self.im_resize(img)

        return img_depth
        
if __name__ == '__main__':
     
    img = Image.open('../demo/scene-0002_img_7.png')
    
    depth = np.load('../demo/scene-0002_depth_7.npy')
    
    img_depth = {'img':img, 'gt':depth}
    
    config = read_config()

    transform_dict = {
        'train': transforms.Compose(
            [ValidDepth(),
             HorizFlip(),
             ColorJittering(),
             ToTensor(),
             ImNormalize(),
             RGB2BGR(),
             ImResize(),
            ]),
        'valid': transforms.Compose(
            [ToTensor(), 
             ImNormalize(),
             RGB2BGR(),
             ImResize(),
            ])
       }

    img = transform_dict['valid'](img_depth)['img']
    
    print('passed!')
    
    
    
    
    