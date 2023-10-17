import sys

sys.path.append('/home/tau/emenier/workspace/miniGPT/')
import miniGPT
import os.path as osp
import torch
import numpy as np
from data_utils import *


data_dir = '/home/tau/emenier/data/AirfRANS/Dataset/'
outfile = 'formatted_AirfRANS.h5'
dataset = H5RansDataset(osp.join(data_dir,outfile))



savedir = '/home/tau/emenier/data/AirfRANS/runs/run_1'

batch_size = 64 # how many independent sequences will we process in parallel?
lr = 3e-4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32
P = 16
C = 1
D = 768
image_size = dataset.N
C_out = 2
n_layers = 12
n_heads = 8
dropout = 0.1
N = image_size**2//(P**2)

print(f'D : {D:}, n_head : {n_heads:}, n_layer : {n_layers:}')

train_dataset = RansPatchDataset(dataset,P,
                indices=np.arange(9*len(dataset)//10))
val_dataset = RansPatchDataset(dataset,P,
                indices=np.arange(9*len(dataset)//10,len(dataset)))



generator_vit = miniGPT.ViT.GeneratorViT(P, C, D, C_out, n_layers, n_heads, 
                    N, dropout_freq=0.,
                    gpus_to_split=None).cuda()

trainer = AirfRANSGPTtrainer(generator_vit,lr,
                    checkpoint_path=savedir,wd=1.e-12,parallel=True)

trainer.train(train_dataset,val_dataset,batch_size=batch_size,
                epoch_length=None,val_length=None,
                patience=5000,save_every=10,device=torch.device('cuda'))