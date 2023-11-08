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
        self.case_names = [
            self.data[c].attrs['case_name'] for c in self.cases
        ]
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
        alpha = torch.tensor(float(group.attrs['alpha']))
        return (torch.tensor(geometry),torch.tensor(sol),u_in,alpha)

class RansPatchDataset(Dataset):

    def __init__(self,dataset,P,indices=None):

        self.dataset = dataset
        self.indices = indices
        if indices is None:
            self.indices = np.arange(len(self.dataset.cases))
        else: self.indices = indices
        self.P = P
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self,idx):

        geometry, sol, u_in, alpha = self.dataset[self.indices[idx]]
        patches_in = to_patches(geometry[-1:],self.P)[0]
        patches_in = torch.cat([
            u_in.reshape(1,1).repeat(patches_in.shape[0],1),
            alpha.reshape(1,1).repeat(patches_in.shape[0],1),
            patches_in],dim=-1)
        patches_out = to_patches(sol[:2],self.P)[0]
        mask = torch.cat([geometry[-1:] for _ in range(2)],
                          dim=0)
        mask = to_patches(mask,self.P)[0]
        mask = mask < 0.5
        return patches_in, patches_out, mask


class H5CloudDataset(Dataset):

    def __init__(self,path,indices=None,return_selector=False):

        self.data = h5py.File(path, 'r')
        self.n_patches = self.data.attrs['max_patches']
        self.cases = list(self.data.keys())
        self.case_names = [
            self.data[c].attrs['case_name'] for c in self.cases
        ]
        if indices is None:
            self.indices = np.arange(len(self.cases))
        else: self.indices = indices
        self.return_selector = return_selector
    def __len__(self):
        return len(self.indices)

    def pad(self,tensor,pad_func=torch.zeros):
        padding = pad_func(self.n_patches-tensor.shape[0],*tensor.shape[1:])
        padded_tensor = torch.cat([tensor,padding],dim=0)
        return padded_tensor

    def __getitem__(self,idx):
        group = self.data[self.cases[self.indices[idx]]]
        
        patches_in = self.pad(torch.tensor(group['patches_in'][:]))
        patches_out = self.pad(torch.tensor(group['patches_out'][:]))

        mask = torch.tensor(group['mask'][:])
        mask = self.pad(mask,pad_func=torch.ones)
        mask = mask >0.5
        patches_out[mask] = 0.
        
        u_in = torch.tensor(float(group.attrs['u_in']))
        alpha = torch.tensor(float(group.attrs['alpha']))
        
        patches_in = torch.cat([
            u_in.reshape(1,1).repeat(patches_in.shape[0],1),
            alpha.reshape(1,1).repeat(patches_in.shape[0],1),
            patches_in],dim=-1)
            
        if self.return_selector:
            selector = torch.tensor(group['selector'][:])
            selector = self.pad(selector,pad_func=torch.zeros)
            selector = selector>0.5
            
            return patches_in, patches_out, mask, selector

        else:
            return patches_in, patches_out, mask


def gather_finetuning_data(dict_outputs,patch_dataset,model,H,W,P,
                            batch_size=64):
    i = 0
    finetuning_input = []
    finetuning_output = []
    while i<len(patch_dataset):
        tmp = i
        patches_in = []
        patches_out = []
        masks = []
        f_out = []
        for j in range(tmp,min(len(patch_dataset),tmp+batch_size)):
            p_i, p_o, m = patch_dataset[i]
            patches_in.append(p_i)
            patches_out.append(p_o)
            name = patch_dataset.dataset.case_names[
                patch_dataset.dataset.indices[patch_dataset.indices[i]]]
            #names.append(name)
            #print(name)
            f_out.append(torch.tensor(dict_outputs[name],dtype=torch.float))
            i+=1
        patches_in = torch.stack(patches_in)
        patches_out = torch.stack(patches_out)
        with torch.no_grad():
            out = model.predict(patches_in.cuda()).detach()
        out = to_img(out,H,W,P)
        finetuning_input.append(out)
        finetuning_output.append(torch.stack(f_out))
        #print(i)
    return torch.cat(finetuning_input,dim=0), torch.cat(finetuning_output,dim=0)[:,0]






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

def get_cloud_idx(bool_mask,C=2):
    bool_mask = bool_mask.reshape(bool_mask.shape[0],-1,C)
    bool_mask = bool_mask.reshape(-1,bool_mask.shape[-1])
    for i in range(bool_mask.shape[-1]-1):
        assert(bool_mask[:,i]==bool_mask[:,i+1]).all(), f'Bool mask is not coherent at columns {i} and {i+1}.'

    return bool_mask[:,0]

def to_point_cloud(selector,patches,C=2,C_patches=None):
    if C_patches is None:
        C_patches = C
    #selector = selector.reshape(selector.shape[0],-1,C)
    #selector = selector.reshape(-1,selector.shape[-1])
    selector = get_cloud_idx(selector)
    patches = patches.reshape(patches.shape[0],-1,C_patches)
    patches = patches.reshape(-1,patches.shape[-1])
    patches = patches[selector]
    return patches

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