import torch
import numpy as np
from dorn.model.utils import read_config, device

class Evaluate():
    
    config = read_config()
        
    K = config['INPUT'].getint('sid_bins')
    
    min_ = config['INPUT'].getfloat('min_depth')
    max_ = config['INPUT'].getfloat('max_depth')
    
    # depth bins according to SID, see (1) in DORN
    expo = np.arange(0,K+2)/(K+1)
    bins = min_*(max_/min_)**expo
    
    def __init__(self):
        
        self.delta1_sum = 0
        self.delta2_sum = 0
        self.delta3_sum = 0
        
        self.abs_rel_sum = 0
        
        self.sq_rel_sum  = 0
        
        self.si_log_0 = 0
        self.si_log_1 = 0
        
        self.inv_sq_sum = 0
        
        self.count = 0
    
    @torch.no_grad()
    def compute(self,predict_depth,target_depth):
        '''
            Compute unnormalized error metrics
        '''
        mask = torch.logical_and(target_depth >= self.min_,
                                 target_depth <= self.max_)
        
        predict_depth = predict_depth[mask]
        target_depth  = target_depth[mask]
        
        thresh = torch.max(target_depth/predict_depth, \
                           predict_depth/target_depth)
        
        delta1_sum = (thresh < 1.25     ).float().sum()
        delta2_sum = (thresh < 1.25 ** 2).float().sum()
        delta3_sum = (thresh < 1.25 ** 3).float().sum()
        
        self.delta1_sum += delta1_sum 
        self.delta2_sum += delta2_sum 
        self.delta3_sum += delta3_sum 
    
        bias = predict_depth - target_depth
        
        abs_rel_sum = (bias.abs()/target_depth).sum()
    
        self.abs_rel_sum += abs_rel_sum
        
        sq_rel_sum = (bias**2/target_depth).sum()  
        
        self.sq_rel_sum += sq_rel_sum
        
        inv_diff = 1/predict_depth - 1/target_depth
        
        self.inv_sq_sum += (inv_diff**2).sum()
        
        log_err = torch.log(predict_depth/target_depth)
        
        self.si_log_0 += (log_err**2).sum()
        
        self.si_log_1 += log_err.sum()
                            
        # number of pixels with gt depths 
        self.count += mask.int().sum()
            
    def get(self):
        '''
            return metrics by name and values.
        '''
        
        return {'delta1':self.delta1.item(),  
                'delta2':self.delta2.item(),   
                'delta3': self.delta3.item(), 
                'abs_rel':self.abs_rel.item(),
                'sq_rel':self.sq_rel.item(),  
                'si_log':self.si_log.item(),
                'irmse':self.irmse.item()}

    @torch.no_grad()
    def results(self):
        '''
            compute average (final) results
        '''
        
        self.delta1 = self.delta1_sum/self.count
        
        self.delta2 = self.delta2_sum/self.count
        
        self.delta3 = self.delta3_sum/self.count
        
        self.abs_rel = self.abs_rel_sum/self.count
        
        self.sq_rel = self.sq_rel_sum/self.count
        
        self.irmse = torch.sqrt(self.inv_sq_sum/self.count)
        
        self.si_log = self.si_log_0/self.count - \
                      (self.si_log_1/self.count)**2                
                                         
    def __str__(self):
        
        s  = 'Error Metrics:\n'
        
        s += 'delta1={0:0.3f}\ndelta2={1:0.3f}\ndelta3={2:0.3f}\n'.format(self.delta1.item(),
                                                                          self.delta2.item(),
                                                                          self.delta3.item())
              
        s += 'absolute relative error={0:0.3f}\n'.format(self.abs_rel.item())
        
        s += 'squared relative error={0:0.3f}\n'.format(self.sq_rel.item())
        
        s += 'inverse root mean square error={0:0.3f}\n'.format(self.irmse.item())
        
        s += 'scale invariant logaritmic error={0:0.3f}\n'.format(self.si_log.item())
        
        return s
        
if __name__ == '__main__' :
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    metrics = Evaluate()
  
    for i in range(2):
        gt = torch.randint(5,80,size=(3,3),dtype=torch.float32).to(device)
        pred = gt + 0.1*torch.randn((3,3),dtype=torch.float32).to(device)
        
        metrics.compute(pred,gt)
        
    metrics.results()
    
    print(metrics)
    
    
        
        
        
            
        
        
        

