import os
import os.path as osp
import numpy as np
import json
import pyvista as pv
import copy
import h5py
from tqdm import tqdm
import scipy.interpolate
import scipy.spatial
import pymetis as metis
import networkx
import multiprocessing as mp


class Bag:
    def __init__(self,data,original_idx):
        self.data = data
        self.original_idx = original_idx
    
    def split(self,x_axis):
        xmin,ymin = self.data.min(0)
        xmax,ymax = self.data.max(0)
        xlims = (xmin,xmax)
        ylims = (ymin,ymax)
        mid_x = (xlims[1]+xlims[0])/2
        mid_y = (ylims[1]+ylims[0])/2
        ratio_x = (self.data[:,0]<mid_x).sum()/self.data.shape[0]
        ratio_y = (self.data[:,1]<mid_y).sum()/self.data.shape[0]
        #x_axis = abs(ratio_x-0.5)<abs(ratio_y-0.5)
        if x_axis:
            xlim1 = (xlims[0],mid_x)
            xlim2 = (mid_x,xlims[1])
            ylims = ylims
            idx1 = np.where(self.data[:,0]<mid_x)[0]
            idx2 = np.where(self.data[:,0]>=mid_x)[0]
            
        else:
            ylim1 = (ylims[0],mid_y)
            ylim2 = (mid_y,ylims[1])
            xlims = xlims
            idx1 = np.where(self.data[:,1]<mid_y)[0]
            idx2 = np.where(self.data[:,1]>=mid_y)[0]
            
        return (Bag(self.data[idx1],self.original_idx[idx1]),
               Bag(self.data[idx2],self.original_idx[idx2]))
    
def create_bags(bag,max_nmbr,x_axis):
    if bag.data.shape[0]<max_nmbr:
        return [bag]
    else:
        x_axis = not x_axis
        bag1,bag2 = bag.split(x_axis)
        return *create_bags(bag1,max_nmbr,x_axis), *create_bags(bag2,max_nmbr,x_axis)

class metis_bag:
    def __init__(self,indices,data):
        self.original_idx = indices
        self.data = data

def metis_split(coords,max_nmbr):
    tri = scipy.spatial.Delaunay(coords)
    edges = set()
    for triangle in tri.simplices:
        edge = sorted([triangle[0], triangle[1]])
        edges.add((edge[0], edge[1]))
        edge = sorted([triangle[0], triangle[2]])
        edges.add((edge[0], edge[1]))
        edge = sorted([triangle[1], triangle[2]])
        edges.add((edge[0], edge[1])) 
    graph = networkx.Graph(list(edges)) 
    dec = 0
    counts = np.array([max_nmbr+1])
    while counts.max()>max_nmbr:
        edgecuts,parts = metis.part_graph( int(len(coords)/(max_nmbr*(0.99-dec))),graph)
        counts = np.array([(parts==i).sum() for i in np.unique(parts)])
        dec+=0.01
    tags = np.unique(parts)
    bags = []
    for t in tags:
        bag_indices = np.where(parts == t)[0]
        bags.append(
            metis_bag(
                bag_indices,coords[bag_indices]
            )
        )
    return bags

def repeat_to_length(array,target_len):
    cat = np.concatenate([array,array],axis=-1)
    if target_len<len(cat):
        return cat[:target_len]
    else: return repeat_to_length(cat,target_len)

def repeat_to_length_0(array,target_len):
    cat = np.concatenate([array,np.zeros(len(array))],axis=-1)
    if target_len<len(cat):
        return cat[:target_len]
    else: return repeat_to_length_0(cat,target_len)

def parse_one_case(data_dir,name,pts_per_bag):
    Uinf, angle = float(name.split('_')[2]), float(name.split('_')[3])
    intern = pv.read(osp.join(data_dir, name, name + '_internal.vtu'))
    coords = intern.points
    idx = (coords[:,0]>-0.5) *  (coords[:,0]<1.5) * (coords[:,1]<1) * (coords[:,1]>-.5)
    coords = intern.points[idx,:2]
    values = intern.point_data['U'][idx,:2]
    #bags = create_bags(Bag(coords,np.arange(coords.shape[0])),pts_per_bag,False)
    bags = metis_split(coords,pts_per_bag)
    patches_in = []
    patches_out = []
    masks = []
    selectors = []
    for b in bags:
        uvals = np.linalg.norm(values[b.original_idx],axis=-1)
        geometry_mask = uvals<1e-6
        to_flatten = np.concatenate([b.data,geometry_mask.reshape(-1,1)],axis=1)
        #to_flatten = b.data
        flat_in = to_flatten.reshape(-1)
        geometry_mask = np.stack([geometry_mask for _ in range(values.shape[-1])]).T
        geometry_mask = geometry_mask.reshape(-1)

        flat_out = values[b.original_idx].reshape(-1)
        patches_in.append(
            #np.concatenate([flat_in,np.zeros(pts_per_bag*coords.shape[-1]-len(flat_in))])
            repeat_to_length_0(flat_in,pts_per_bag*to_flatten.shape[-1])
            )
        patches_out.append(
            #np.concatenate([flat_out,np.zeros(pts_per_bag*values.shape[-1]-len(flat_in))])
            repeat_to_length_0(flat_out,pts_per_bag*values.shape[-1])
            )
        selector = (np.arange(len(patches_out[-1]))<len(flat_out))
        selectors.append(selector)
        mask = copy.copy(selector)==False # Reverse selector, mask cancels values if True
        mask[:len(flat_out)][geometry_mask] =  True
        masks.append(mask)
    return np.stack(patches_in), np.stack(patches_out), np.stack(masks), np.stack(selectors), Uinf, angle
    
def add_group(i,lst,h5file,data_dir,pts_per_bag):
    name = lst[i]
    if not osp.isdir(osp.join(data_dir,name)): 
            return 0

    #patches_in, patches_out, mask, selector, u_in, alpha = \
    return i,*parse_one_case(data_dir,name,pts_per_bag)

if __name__=='__main__':
    pts_per_bag = 250

    data_dir = '/home/tau/emenier/data/AirfRANS/Dataset/'
    outfile = 'pointcloud_250_bc.h5'
    f = h5py.File(osp.join(data_dir,outfile), 'a')
    lst = os.listdir(data_dir)
    lst = [n for n in lst if osp.isdir(osp.join(data_dir,n))]

    def func_to_map(i):
        return add_group(i,lst,f,data_dir,pts_per_bag)

    patches_lengths = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for ret in tqdm(pool.imap_unordered(func_to_map,range(len(lst)))):
            if not isinstance(ret,tuple):
                continue

            i,patches_in, patches_out, mask, selector, u_in, alpha = ret

            patches_lengths.append(patches_in.shape[0])

            grp = f.require_group('case_{}'.format(i))
            grp.create_dataset('patches_in', data=patches_in, dtype='f')
            grp.create_dataset('patches_out',data=patches_out,dtype='f')
            grp.create_dataset('selector',data=selector,dtype='b')
            grp.create_dataset('mask',data=mask,dtype='b')
            grp.attrs['u_in'] = u_in
            grp.attrs['alpha'] = alpha
            grp.attrs['case_name'] = lst[i]
            f.flush()


        
    f.attrs['max_patches'] = np.array(patches_lengths).max()
    """
    max_patches = 0
    for i,name in enumerate(tqdm(lst)):

        if not osp.isdir(osp.join(data_dir,name)): 
            continue

        patches_in, patches_out, mask, selector, u_in, alpha = \
                    parse_one_case(data_dir,name,pts_per_bag)

        grp = f.require_group('case_{}'.format(i))
        grp.create_dataset('patches_in', data=patches_in, dtype='f')
        grp.create_dataset('patches_out',data=patches_out,dtype='f')
        grp.create_dataset('selector',data=selector,dtype='b')
        grp.create_dataset('mask',data=mask,dtype='b')
        grp.attrs['u_in'] = u_in
        grp.attrs['alpha'] = alpha
        grp.attrs['case_name'] = name
        max_patches = max(max_patches,patches_in.shape[0])
        #f.flush()

    f.attrs['max_patches'] = max_patches"""
    f.attrs['pts_per_bag'] = pts_per_bag
    f.attrs['C_in'] = 3
    f.attrs['C_out'] = 2
    f.close()