import torch
import numpy as np
import caffe_pb2 as cq
from model.network.DORN_kitti import DORN
import sys

# read caffemodel
f = open('cvpr_kitti.caffemodel', 'rb')
cq2 = cq.NetParameter()
cq2.ParseFromString(f.read())
f.close()

keys = (i[0] for i in DORN().state_dict().items())

# numpy model parameters
data = {}

for l in cq2.layer:
    
    print("-------{0}-------".format(l.name))
    
    if (l.type  == 'BN'):
        
        #  0 - weight(slope), 1 - bias, 2 - mean , 3 - variance
        for i,b in enumerate(l.blobs):

            param_name = next(keys)
            
            print(param_name)
            
            if ('bn' not in param_name):
                print('{0} vs {1}'.format(l.name,param_name))
                sys.exit(1)

            params = np.array(b.data,dtype=float)
            
            if ('weight' in param_name and i==0):
                data[param_name] = torch.from_numpy(params.flatten())
            
            elif ('bias' in param_name and i==1):
                data[param_name] = torch.from_numpy(params.flatten())
        
            elif ('running_mean' in  param_name and i==2):
                data[param_name] = torch.from_numpy(params.flatten())
                
            elif ('running_var' in param_name and i==3):
                data[param_name] = torch.from_numpy(params.flatten())
                
            else:
                print('{0} vs {1}'.format(l.name,param_name))
                sys.exit(1)

        # neglect param 'num_batches_tracked'
        print('pytorch param {0} is neglected'.format(next(keys)))

    elif(l.type == 'Convolution'):
        
        for i,b in enumerate(l.blobs):
            
            param_name = next(keys)
            
            print(param_name)
        
            if('conv' not in param_name):
                print('{0} vs {1}'.format(l.name,param_name))
                sys.exit(1)
                
            #  Blobs are memory interface holding model parameters
            params = np.array(b.data,dtype=float)
        
            if ('weight' in param_name and i==0):
                data[param_name] = torch.from_numpy(params.reshape(b.shape.dim))
        
            elif ('bias' in param_name and i==1):
                data[param_name] = torch.from_numpy(params.flatten())
                
            else:
                print('{0} vs {1}'.format(l.name,param_name))
                sys.exit(1)
        
    elif (l.type == 'InnerProduct'):
        
        for i,b in enumerate(l.blobs):
            
            param_name = next(keys)
            
            print(param_name)
        
            if('fc' not in param_name):
                print('{0} vs {1}'.format(l.name,param_name))
                sys.exit(1)
            
            params = np.array(b.data,dtype=float)
            
            if ('weight' in param_name and i==0):
                data[param_name] = torch.from_numpy(params.reshape(b.shape.dim))

                
            elif ('bias' in param_name and i==1):
                params = np.array(b.data,dtype=float)
                data[param_name] = torch.from_numpy(params.reshape(b.shape.dim))
                
                
            else:
                print('{0} vs {1}'.format(l.name,param_name))
                sys.exit(1)
                
torch.save(data,'../pretrained_models/dorn_kitti.npy')
    