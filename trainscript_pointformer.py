import sys

sys.path.append('/home/tau/emenier/workspace/miniGPT/')
import miniGPT
import os.path as osp
import torch
import numpy as np
from data_utils import *


data_dir = '/home/tau/emenier/data/AirfRANS/Dataset/'
outfile = 'pointcloud_250_bc.h5'


ds = H5CloudDataset(osp.join(data_dir,outfile))
train_dataset = H5CloudDataset(osp.join(data_dir,outfile),
                               indices=np.arange(9*len(ds)//10))
val_dataset = H5CloudDataset(osp.join(data_dir,outfile),
                               indices=np.arange(9*len(ds)//10,len(ds)))

savedir = '/home/tau/emenier/data/AirfRANS/runs/point_run2/'

batch_size = 16 # how many independent sequences will we process in parallel?
lr = 3e-4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32
size_in = 2+ds.data.attrs['pts_per_bag']*ds.data.attrs['C_in']
size_out = ds.data.attrs['pts_per_bag']*ds.data.attrs['C_out']
D = 768
n_layers = 12
n_heads = 8
dropout = 0.1

print(f'D : {D:}, n_head : {n_heads:}, n_layer : {n_layers:}')

pointformer = miniGPT.ViT.PointCloudTransformer(size_in, size_out, D, n_layers, n_heads, 
                     dropout_freq=dropout,
                    gpus_to_split=None,linear_out=False).cuda()

trainer = AirfRANSGPTtrainer(pointformer,lr,
                    checkpoint_path=savedir,wd=1e-6,parallel=True)


trainer.train(train_dataset,val_dataset,batch_size=batch_size,
                epoch_length=None,val_length=None,
                patience=5000,save_every=10,device=torch.device('cuda'))