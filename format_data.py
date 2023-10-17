import os
import os.path as osp
import numpy as np
import json
import pyvista as pv
import copy
import h5py
from tqdm import tqdm
import scipy.interpolate
N = 256

data_dir = '/home/tau/emenier/data/AirfRANS/Dataset/'
outfile = 'formatted_AirfRANS.h5'

with open(osp.join(data_dir,'manifest.json'), 'r') as f:
    manifest = json.load(f)
manifest_train = manifest['full_train']

def parse_one_case(data_dir,name):
        
    
    dataset = pv.read(osp.join(data_dir, name, name + '_internal.vtu'))
    xmin, xmax = dataset.points[:,0].min(), dataset.points[:,0].max()
    ymin, ymax = dataset.points[:,1].min(), dataset.points[:,1].max()
    xcoords = np.linspace(xmin,xmax,N)
    ycoords = np.linspace(ymin,ymax,N)
    x,y = np.meshgrid(np.linspace(-0.5,1.5,N),np.linspace(-0.5,1,N))
    
    coords = np.concatenate([x.reshape(-1,1),y.reshape(-1,1)],axis=1)
    out = coords
    for i in range(2):
        u = scipy.interpolate.griddata(\
            (dataset.points[:,0],dataset.points[:,1]),
            dataset.point_data['U'][:,i],
            coords,method='linear').reshape(-1,1)
        out = np.concatenate([out,u],axis=1)

    p = scipy.interpolate.griddata(\
            (dataset.points[:,0],dataset.points[:,1]),
            dataset.point_data['p'],
            coords,method='linear').reshape(-1,1)

    out = np.concatenate([out,p],axis=1)

    geometry = copy.copy(out[:,:3])
    geometry[np.linalg.norm(out[:,2:4],axis=1)!=0,-1]=1

    u_in = name.split('_')[2]

    return geometry, out[:,2:], u_in

f = h5py.File(osp.join(data_dir,outfile), 'a')
lst = os.listdir(data_dir)
for i,name in enumerate(tqdm(lst)):

    if not osp.isdir(osp.join(data_dir,name)): 
        continue

    geometry, sol, u_in = parse_one_case(data_dir,name)

    grp = f.require_group('case_{}'.format(i))
    grp.create_dataset('geometry', data=geometry, dtype='f')
    grp.create_dataset('sol',data=sol,dtype='f')
    grp.attrs['u_in'] = u_in
    f.flush()

f.attrs['N'] = N

f.close()