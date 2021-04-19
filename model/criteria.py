import torch
import torch.nn as nn
from dorn.model import utils

class OrdLoss(nn.Module):

    def __init__(self, config):
        """
            Ordinal loss considers the strong ordinal correlation
            b/w depth values since they form a well-ordered set.
            Thus, minimizing the ordinal loss ensures that predic-
            tions further from true labels incur a greater penalty
            than those closer to the true label.
        """
        super().__init__()
        
        self.config = config
        
        self.K = config['INPUT'].getint('sid_bins') 
        
    def forward(self, ord_probs, gt_depths):
        """
        @Parameters: 
            ord_probs: ordinal probabilities.
            gt_depth: groundtruth depth map.
        """
        
        height, width = gt_depths.shape[2:]
            
        # nonzero values are valid target/gt values. note that index operation
        # is not differentiable. Thus, detach it from the computational graph
        mask = (gt_depths > 0).detach()
        
        gt_labels = utils.get_labels(gt_depths, self.config)
        
        # ordinal labels k=0,1,...,K-1 
        ord_labels = torch.linspace(0, self.K-1, self.K, dtype=torch.long,
                                    requires_grad=False, device=utils.device)
        
        ord_labels = ord_labels.view(1,self.K,1,1).repeat(1,1,height,width)
     
        # variable k used in the 1st and 2nd parts of the loss, see (2) in DORN
        k0 = (ord_labels < gt_labels).detach()
        k1 = (ord_labels >= gt_labels).detach()
        
        # valid ks
        vk0 = torch.logical_and(mask, k0).detach()
        vk1 = torch.logical_and(mask, k1).detach()
        
        # probabilities of bins up to l_gt for gt_label. 
        Pk0 = ord_probs[vk0]
        # probabilities of bins after and including l_gt 
        Pk1 = 1 - ord_probs[vk1]

        # sum of ordinal losses of valid pixels 
        loss = torch.sum(torch.log(torch.clamp(Pk0, 1e-8, 1e8))) +  \
               torch.sum(torch.log(torch.clamp(Pk1, 1e-8, 1e8)))

        # the number of valid pixels
        num_pixels = mask.sum()
        
        # average of pixel ordinal loss
        loss = -loss/num_pixels 
        
        return loss