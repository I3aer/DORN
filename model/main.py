import os
import torch
from torch import optim
import torchvision

from dorn.model import utils
from dorn.model.dataset.loader import DatasetLoader 
from dorn.model.network.dorn import DORN
from dorn.model.criteria import OrdLoss
from dorn.model.metrics import Evaluate
from dorn.model.dataset import transforms as ds_transforms

def set_bn_dropout_eval(module):
    '''
        Set BN and Dropout layers to evaluation mode. Accordinyly, 
        BN layers use population mean and variances with those
        scale and shift parameters,and Dropout layers are disabled. 
    '''
    if isinstance(module, (torch.nn.BatchNorm2d,torch.nn.Dropout2d)):
        module.eval()
        
def create_loader(config):
    
    dataset_name = config['DATASET']['dataset_name']
    
    batch_size = config['TRAIN'].getint('batch_size')
    
    # the number of sub-processes to prepare batches
    num_workers = config['TRAIN'].getint('num_workers')
    
    do_shuffle = config['TRAIN'].getboolean('shuffle')

    # set this flag to True to tranfer tensors on GPU 
    pin_memory = config['TRAIN'].getboolean('pin_memory')

    if dataset_name == 'nuscenes':
        # load data from dataset objects:
            
        # transforms applied to inputs 
        transform_dict = {'train': torchvision.transforms.Compose(
                                    [ds_transforms.ValidDepth(),
                                     ds_transforms.HorizFlip(),
                                     ds_transforms.ColorJittering(),
                                     ds_transforms.ToTensor(),
                                     ds_transforms.ImNormalize(),
                                     ds_transforms.RGB2BGR(),
                                     ds_transforms.FourCrops()
                                    ]),
                            'valid':torchvision.transforms.Compose(
                                    [ds_transforms.ValidDepth(),
                                     ds_transforms.ToTensor(), 
                                     ds_transforms.ImNormalize(),
                                     ds_transforms.RGB2BGR(),
                                     ds_transforms.FourCrops()
                                    ])
                        }
        
        train_set = DatasetLoader(mode='train', transform=transform_dict['train'])
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=do_shuffle,
                                                   num_workers=num_workers, pin_memory=pin_memory)
        
        valid_set = DatasetLoader(mode='valid', transform=transform_dict['valid'])
        
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers, pin_memory=pin_memory)
        
        return train_loader, valid_loader
    
def main():
    '''
         Train the DORN and after each epoch runs a full validation. The function
         also keeps track of the best performing model (in terms of X performance
         metric), and at the end of training it saves and return that model.
    '''
    
    config = utils.read_config()

    train_loader, valid_loader = create_loader(config)
    
    resume_training = config['TRAIN'].getboolean('resume')
    
    ckpt_path = config['TRAIN']['checkpoint_path']
    
    # generate model 
    depth_net = DORN()
    
    if resume_training:
        # resume training by loading the saved checkpoint 
        
        if os.path.isfile(ckpt_path): 
            print("loading checkpoint {0}".format(ckpt_path))
            
            checkpoint = torch.load(ckpt_path)
    
            start_epoch = checkpoint['epoch'] + 1
            # previously saved model state and optimizer state
            optimizer = checkpoint['optimizer']
            model_dict = checkpoint['state_dict']
            
            # set model and optimizer states to those in checkpoint
            depth_net.load_state_dict(model_dict)
            optimizer.load_state_dict(optimizer)

    else:
        
        path_model = config['MODEL']['kitti']
        
        model_dict = utils.get_model(path_model)
        
        dorn_dict = depth_net.state_dict()
        
        # overwrite existing parameter values
        dorn_dict.update(model_dict) 
        
        # load the pretrained model
        depth_net.load_state_dict(dorn_dict)
        
        start_epoch = 0

    # transfer network onto the GPU if available
    # always do this before constructing optimizer 
    depth_net = depth_net.to(utils.device) 
    
    # log the network graph 
    tb = utils.graph_visualize(depth_net,config)

    # ordinal loss function
    criterion = OrdLoss(config)
    
    '''
        sets requires_grad attribute of the all params in the
        model to False when we are feature extracting. This is
        because that we only want to compute gradients for the
        the last layer(s) of the DORN, i.e., the 2nd conv layer
        of the scene understanding module. 
    '''
    # fix all network parameters
    for param in depth_net.parameters():
        param.requires_grad = False
        
    # trainable parameters
    train_params = []
    
    # scu block where layer(s) is/are fine-tuned
    my_block = depth_net.scu_module.concat_process
    
    # make the 2nd conv layer of the scu module trainable
    for key,layer in my_block.__dict__['_modules'].items():
        if (key in {'conv_1','conv_2', 'conv_comp'}):
            for param in layer.parameters():
                train_params.append(param)
                param.requires_grad = True
                
    lr = config['OPTIMIZER'].getfloat('learning_rate')
    wd = config['OPTIMIZER'].getfloat('weight_decay')
    omega = config['OPTIMIZER'].getfloat('momentum')
    
    # dict of parameters block and their lr
    train_params = [{'params':train_params,'lr':lr}]

    # construct the optimizer:only conv2 in scu module is optimized 
    optimizer = \
    optim.SGD(train_params, lr=lr, momentum=omega, weight_decay=wd, nesterov=True)
    
    # decay the lr by gamma once epoch no reaches one of the milestones 
    scheduler = \
    optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,20], gamma=0.1)
        
    num_epochs = config['TRAIN'].getint('num_epochs')
    
    valid_rate = config['TRAIN'].getint('valid_rate')
    
    best_acc = 0.0
    
    for epoch in range(start_epoch, num_epochs):
    
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
            
        train(depth_net, train_loader, criterion, optimizer, epoch, config, tb)
        
        # find the best model by running validation
        if (epoch+1) % valid_rate == 0:  
            metrics = validate(depth_net, valid_loader, criterion, epoch, config, tb)  
            
            epoch_acc = metrics['delta1']
            
            # deep copy the best model
            if epoch_acc > best_acc:
                
                best_acc = epoch_acc
                
                best_model_wts = depth_net.state_dict()
                    
                torch.save(best_model_wts, ckpt_path)  
             
        # decay learning rate of params after each epoch
        scheduler.step()
        
    # write all pending events
    tb.flush()

    tb.close()
    
def train(model, dataloader, criterion, optimizer, epoch, config, tb):
    '''
        Train the model for the specified number of epochs using the given
        dataset of samples and criterion function.
        @parameters:
         model: a pytorch network model.
         dataloader: iterator over a map-style dataset.
         criterion: the optimization criterion.
         optimizer: optimization algorithm to update weights.
         epoch: the iteration number over the whole dataset
         config: configuration parameters.
         tb: tensorboard to keep track of some metrics.
        @return: state_dict of the best model.
    '''
        
    # set the model to training mode
    model.train() 
    
    # set BN and Dropout to evaluation mode
    model.apply(set_bn_dropout_eval)
        
    # total loss for each epoch
    running_loss = 0 
    
    grad_accum = config['TRAIN'].getint('grad_accum')
    
    # image dimensions
    c = config['INPUT'].getint('channel')
    w = config['INPUT'].getint('width')
    h = config['INPUT'].getint('height')
    
    # for each epoch iterate over dataset
    for i,sample in enumerate(dataloader):
        
        print('batch_no:{}/{} \r'.format(i,len(dataloader)-1),end='')
        
        # transfer tensors onto the device 
        input_imgs = sample['img'].to(utils.device)
        input_imgs = input_imgs.view(-1,c,h,w)
        
        gt_depths = sample['gt'].to(utils.device)
        gt_depths = gt_depths.view(-1,1,h,w)
     
        if i % grad_accum == 0:
            # zero the buffers of parameter gradients
            optimizer.zero_grad()
        
        with torch.autograd.set_detect_anomaly(True):
            with torch.set_grad_enabled(True):
            
                # forward pass
                ord_labels, ord_probs = model(input_imgs)
                
                # compute avg loss of mb
                loss = criterion(ord_probs, gt_depths)
                
                # compute and store gradients 
                loss.backward()
                
                # compute total loss of all samples in each mini-batch.
                # Note that always release the computational graph from
                # the loss by using .item() to get its python value.
                running_loss += loss.item()*input_imgs.size(0)
                
                # update model parameters using accumulated gradients
                # e.g., if SGD is employed, w = w - lr*sum_i(grad_i)
                if (i + 1) % grad_accum == 0:
                    optimizer.step()
                
    # log average loss over all examples in the dataset
    running_loss /= dataloader.dataset.__len__()
    tb.add_scalar('loss/train', running_loss, epoch)
    print('Train loss:{0:0.3f}'.format(running_loss))

def validate(model, dataloader, criterion, epoch, config, tb):
    '''
        Test/validate the trained model.
        @parameters:
         model: the trained network.
         criterion: the optimization criterion to compute
         the loss.
         dataloader: iterator over a map-style dataset.
         epoch: the iteration number over the whole dataset
         config: configuration parameters.
         tb: tensorboard to keep tracks of some metrics.
        @return performance metrics.
    '''
    
    eval_metrics = Evaluate()
    
    # set model to evaluate mode
    model.eval()  
    
    # average loss for epoch
    running_loss = 0 
    
    # image dimensions
    c = config['INPUT'].getint('channel')
    w = config['INPUT'].getint('width')
    h = config['INPUT'].getint('height')

    for sample in dataloader:
        
        # transfer tensors onto the device
        input_imgs = sample['img'].to(utils.device)
        input_imgs = input_imgs.view(-1,c,h,w)
        
        gt_depths = sample['gt'].to(utils.device)
        gt_depths = gt_depths.view(-1,1,h,w)
        
        with torch.set_grad_enabled(False):
            
            ord_labels, ord_probs = model(input_imgs)
            
            # compute avg loss of mb
            loss = criterion(ord_probs, gt_depths)
            
            # compute total loss of all samples in each mini-batch.
            # Note that always release the computational graph from
            # the loss by using .item() to get its python value.
            running_loss += loss.item()*input_imgs.size()[0]
    
        eval_metrics.compute(pred_depths, gt_depths)
    
    # compute metrics for validation
    eval_metrics.results()
    
    # log average loss
    running_loss /= dataloader.dataset.__len__()
    tb.add_scalar('loss/valid', running_loss, epoch)
    print('Valid loss:{0:0.3f}'.format(running_loss))
    
    # log metrics
    metrics = eval_metrics.get()
    for name, val in metrics.items():
        tb.add_scalar('valid_' + name, val, epoch)
        
    print('delta-1:{0:.3f}'.format(metrics['delta1']))
        
    # write image and depth estimation to tensorboard
    pixel_means = config['IMG_NORM']['RGB_mu']
    pixel_means = list(map(float,pixel_means.split(',')))
    # BGR to RGB conversion
    pixel_means = torch.Tensor(pixel_means).view(1,-1,1,1)
    pixel_means = pixel_means.to(utils.device).detach()
    
    pixel_stds = config['IMG_NORM']['RGB_std']
    pixel_stds = list(map(float, pixel_stds.split(',')))
    pixel_stds = torch.Tensor(pixel_stds).view(1,-1,1,1)
    pixel_stds = pixel_stds.to(utils.device).detach()
    
    # remove normalization -> original RGB image 
    input_imgs = input_imgs[:,[2,1,0]]*pixel_stds + pixel_means
    img_grid = torchvision.utils.make_grid(input_imgs)
    tb.add_image('images', img_grid, epoch)
    
    pred_grid = torchvision.utils.make_grid(pred_depths)
    tb.add_image('predictions', pred_grid, epoch)
    
    gt_grid = torchvision.utils.make_grid(gt_depths)
    tb.add_image('groundtruths', gt_grid, epoch)
        
    return metrics

if __name__ == '__main__':
    main()