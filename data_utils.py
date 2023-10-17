import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import sys

sys.path.append('/home/tau/emenier/workspace/miniGPT/')
import miniGPT
from tqdm import tqdm

class H5RansDataset(Dataset):
    def __init__(self,path,indices=None):


        self.data = h5py.File(path, 'r')
        self.N = self.data.attrs['N']
        self.cases = list(self.data.keys())
        if indices is None:
            self.indices = np.arange(len(self.cases))
        else: self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self,idx):
        
        group = self.data[self.cases[self.indices[idx]]]
        
        sol = group['sol'][:].T
        sol = sol.reshape(sol.shape[0],self.N,self.N)
        geometry = group['geometry'][:].T
        geometry = geometry.reshape(geometry.shape[0],self.N,self.N)
        
        u_in = torch.tensor(float(group.attrs['u_in']))
        return (torch.tensor(geometry),torch.tensor(sol),u_in)

class RansPatchDataset(Dataset):

    def __init__(self,dataset,P,indices=None):

        self.dataset = dataset
        self.indices = indices
        if indices is None:
            self.indices = np.arange(len(self.cases))
        else: self.indices = indices
        self.P = P
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self,idx):

        geometry, sol, u_in = self.dataset[self.indices[idx]]
        patches_in = to_patches(geometry[-1:],self.P)[0]
        patches_in = torch.cat([
            u_in.reshape(1,1).repeat(patches_in.shape[0],1),
            patches_in],dim=-1)
        patches_out = to_patches(sol[:2],self.P)[0]
        mask = torch.cat([geometry[-1:] for _ in range(2)],
                          dim=0)
        mask = to_patches(mask,self.P)[0]
        mask = mask < 0.5
        return patches_in, patches_out, mask

def to_patches(img,P):
    if img.ndim == 3: img = img.reshape(1,*img.shape)
    H,W = img.shape[-2],img.shape[-1]
    n_H, n_W = H//P, W//P
    patches = []
    for i in range(n_H):
        for j in range(n_W):
            patches.append(img[...,i*P:(i+1)*P,
                                j*P:(j+1)*P].flatten(start_dim=-3))
    return torch.stack(patches).permute(1,0,2)

def to_img(patches,H,W,P):

    C = patches.shape[-1]//P**2
    B = patches.shape[0]
    img = torch.empty(patches.shape[0],C,H,W)
    for i in range(H//P):
        for j in range(W//P):
            img[...,i*P:(i+1)*P,j*P:(j+1)*P] = \
                patches[:,i*(W//P) + j].reshape(B,C,P,P)
    return img



class AirfRANSGPTtrainer(miniGPT.train_utils.DecoderGPTtrainer):

    def evaluate_dataset(self,loader,train=True,desc='',length=None):

        was_training = self.gpt_model.training
        if train: self.gpt_model.train()
        else: self.gpt_model.eval()

        count = 0

        epoch_losses = []

        pbar = tqdm(loader)
        for (patches_in,patches_out, mask) in pbar:
            count += 1
            
            self.opt.zero_grad()

            loss = self.gpt_model(patches_in,patches_out, mask).mean()
            epoch_losses.append(loss.item())
            if train:
                loss.backward()
                self.opt.step()

            pbar.set_description(desc+' Loss {:.3e}'.format(
                                    np.array(epoch_losses).mean()))
        miniGPT.train_utils.monitor_memory()
        if was_training: self.gpt_model.train()
        else: self.gpt_model.eval()
        return epoch_losses