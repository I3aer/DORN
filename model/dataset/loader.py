import os
import sys

from torch.utils.data import Dataset
from dorn.model.utils import read_config
from dataset.utils import read_img_gt_path, read_sample
from dataset.transforms import *

class DatasetLoader(Dataset):


    def __init__(self, mode=None, transform=None):
        '''
            Create a map-style dataset for training or testing with
            optinonal transforms. 
            @parameters: 
                mode: one of the followings:'train', test', or 'valid'.
                transform: a callable transform/composed of transforms
                applied to each sample.
        '''
        
        super().__init__()
        
        if mode not in ('train','test', 'valid'):
            print('invalid mode {}'.format(mode))
            sys.exit(1)
        
        self.mode = mode
        
        self.transform = transform
        
        self.config = read_config()
        
        dataset_dir = self.config['DATASET']['dataset_dir']
            
        dataset_name = self.config['DATASET']['dataset_name']
        
        filename = dataset_name + '_' + self.mode + '_files.txt'
        
        files_path = os.path.join(dataset_dir, self.mode, filename)
            
        gt_dir = self.config['DATASET']['gt_dir']
        
        gt_dir = os.path.join(dataset_dir, self.mode, gt_dir)
        
        input_dir = self.config['DATASET']['input_dir']
        
        input_dir = os.path.join(dataset_dir, self.mode, input_dir)

        self.img_gt_paths = read_img_gt_path(files_path, input_dir, gt_dir,)
        
    def __len__(self):
        '''
            Return the size of the dataset
        '''
        return len(self.img_gt_paths)

    def __getitem__(self, idx):
        '''
            Get the idx-th sample from the map-style dataset.
            @parameters:
                idx: int index of the sample
            @return: idx-th sample as dict {'img':x, 'gt':y} 
        '''
        
        img, depth = read_sample(self.img_gt_paths[idx])
        
        sample = {'img':img, 'gt':depth}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

if __name__ == '__main__':
    import numpy as np

    import matplotlib.pyplot as plt
    
    config = read_config()
    
    means = config['IMG_NORM']['RGB_mu']
    means = list(map(float,means.split(',')))
    
    stds = config['IMG_NORM']['RGB_std']
    stds = list(map(float,stds.split(',')))
    
    w = config['INPUT'].getint('width')
    h = config['INPUT'].getint('height')
    
    mode = 'train'
    
    # select transform according to the mode
    if mode == 'train':
        transform = transforms.Compose([ValidDepth(),
                                        HorizFlip(),
                                        ColorJittering(),
                                        CenterCrop(),
                                        ToTensor(),
                                        ImNormalize(),
                                        RGB2BGR()])
    else:
        transform = transforms.Compose([ValidDepth(),
                                        CenterCrop(), 
                                        ToTensor(),
                                        ImNormalize(),
                                        RGB2BGR()])

    # creata a dataset with composed transforms
    my_dataset = DatasetLoader(mode, transform)
    
    # generate a customized map-style dataloader
    dataloader = torch.utils.data.DataLoader(my_dataset, 
                                             batch_size=1, 
                                             shuffle=False, 
                                             num_workers=2)

    print('dataset num is ', len(dataloader))
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for i, sample in enumerate(dataloader):
        
        print('Epoch:{}/{}'.format(i,len(my_dataset)-1))
        
        img = torch.squeeze(sample['img']).numpy()
        
        img = np.moveaxis(img,(1,2,0),(0,1,2))
        
        ax1.imshow(img.astype(np.uint8))
        
        gt = torch.squeeze(sample['gt']).numpy()
        
        ax2.imshow(gt)
        
        plt.show(block=False)
        
        plt.show()

        