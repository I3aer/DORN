import torch
from PIL import Image

from dorn.model import utils 
from dorn.model.network.DORN_kitti import DORN

class DORNNET():
    
    config = utils.read_config()
    
    def __init__(self):
        
        self.depth_net = DORN() 
        
        path_model = DORNNET.config['MODEL']['kitti']
        
        print('{0} model is used!'.format('kitti'))
        
        model_dict = utils.get_model(path_model)
    
        # load the trained model's parameters
        self.depth_net.load_state_dict(model_dict)
    
        # move the network to cuda/gpu device if available
        self.depth_net.to(utils.device)
    
        # bn and dropout layers will work in evaluation mode
        self.depth_net.eval()
        
    def __call__(self,filename):
                        
        img = Image.open(filename)
        
        img_tensor = utils.transform_img(img, self.config)
        
        if self.depth_net.training:
            raise ValueError('Model is in training mode!')
        
        # disable autograd engine because of no back-propagatio 
        with torch.no_grad():
            pred_labels, ord_probs = self.depth_net(img_tensor)
        
        return pred_labels, ord_probs
    
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    filename = './model/demo/scene-0002_img_7.png'
    
    dorn = DORNNET()
    
    depth_map = dorn(filename)
    
    plt.figure('depth_map')
    
    plt.imshow(depth_map)
    
    plt.show()
    
    plt.pause(1)

    
    
        