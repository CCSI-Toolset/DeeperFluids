import pandas as pd
import os
from pathlib import Path
from glob import glob
import pickle
import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata
from multiprocessing import Pool
import time
from .utils import lock_file_for_MP, pkl_save, get_grid_folder, get_Ny, get_sims


def create_from_args(args):
    '''
    # TODO: clean up / remove lock files that get created?
                flexible (argument-based) number of processors for MP?
    '''

    # are the necessary grids already made / is the raw data preprocessed?
    if args.meta_data == 'PNNL':
        
        processed_folder = get_grid_folder(args)
        Path(processed_folder).mkdir(parents=True, exist_ok=True)
        processed_file_names = ["{:03d}.pkl".format(x) for x in range(1,get_sims(args)+1)]
        for i, fn in enumerate(processed_file_names):
            processed_grid_path = os.path.join(processed_folder, fn)
            try:
                file_size = os.path.getsize(processed_grid_path)
            except os.error:
                file_size = 0
            if file_size < 1000: # the grids are at least 1 KB
                # so if the file size isn't that large, it needs to be created, and we try to do that here:
                print('\nMaking the grid for sim #{}!\n'.format(i))
                lock_file_for_MP(processed_grid_path, process_folder, args=args, out_fn = processed_grid_path) 
                # (we're using locking so that multiple calls to grid.py can be made simultaneously)

        # are the necessary files preprocessed? 
        # (it's possible that they aren't if multiple calls were made to grid.py and one of those other calls hasn't finished yet)
        grid_exists = 0
        for fn in processed_file_names:
            processed_grid_path = os.path.join(processed_folder, fn)
            try:
                file_size = os.path.getsize(processed_grid_path)
            except os.error:
                file_size = 0
            if file_size > 1000:
                grid_exists += 1 # this grid has been made (using 1000 bytes is a rough heuristic; TODO: get a better rule of thumb?)
        grid_exists = grid_exists == len(processed_file_names) # are all meta_sims sims present?
        if not grid_exists:
            print('\nGrid data does not yet exist. It is possible that another process/job is finishing it.\n')
            return 0
    else:
        return 0 # no other data set up yet, just PNNL

    return 1


def loadfile(files_and_args, save_loc=None):
    fn, args = files_and_args
    D = pd.read_csv(fn)
    x = D['X (m)'].values.astype('float32')
    y = D['Y (m)'].values.astype('float32')
    columns = D.columns
    z = D[columns[args.meta_channel]].values.astype('float32')
    grid_x, grid_y, grid_z = interpData(x,y,z,
                                        Nx=args.meta_gridSize,
                                        Ny=get_Ny(args),
                                        delta_x=None,nextPow2=None,
                                        method='linear')
    if not save_loc:
        save_loc = os.path.join(args.meta_outputDir, args.meta_data, "channel_" + 
                            str(args.meta_channel), 
                            "grid_size_" + str(args.meta_gridSize) + (
                            '_' + str(get_Ny(args)) if args.meta_Ny_over_Nx is not None else ''),
                             'grid_x_grid_y.pkl')
    if not os.path.exists(save_loc):
        pkl_save({'grid_x':grid_x, 'grid_y':grid_y}, save_loc) 
    return grid_z.astype('float32')


def interpData(x,y,z,Nx=None,Ny=None,delta_x=None,nextPow2=False,method='linear'):
    '''
    This function takes 3 lists of points (x,y,z) and maps them to a 
    rectangular grid. Either Nx or Ny must be set or delta_x must be set. 
    e.g. 
    
    x = y = z = np.random.rand(30)
    grid_x, grid_y, grid_z = interpData(x,y,z,Nx=128,Ny=128)
    
    or 
    
    grid_x, grid_y, grid_z = interpData(x,y,z,delta_x=1e-3,nextPow2=True)
    '''
    
    eps = 1e-4 # needed to make sure that the interpolation does not have nans. 
    def _NextPowerOfTwo(number):
        # Returns next power of two following 'number'
        return np.ceil(np.log2(number))
    
    if Nx == None and Ny == None:
        assert delta_x != None
        delta_y = delta_x
        Nx = int((x.max() - x.min())/delta_x)
        Ny = int((y.max() - y.min())/delta_y)

    if nextPow2:
        Nx = 2**_NextPowerOfTwo(Nx)
        Ny = 2**_NextPowerOfTwo(Ny)
        
    grid_x, grid_y = np.mgrid[x.min()+eps:x.max()-eps:Nx*1j,y.min()+eps:y.max()-eps:Ny*1j]
    grid_z = griddata(np.array([x,y]).T, z, (grid_x, grid_y), method=method)
    return grid_x, grid_y, grid_z


def getInt(f):
    return int(f.split('_')[-1].replace('.csv',''))


# process all the files/timesteps in the folder
def process_folder(args = None, out_fn = None):
    fd = os.path.join(args.meta_dataDir,out_fn.split('/')[-1][:-4])
    out = []
    fns = glob(os.path.join(fd,'*.csv'))
    L = np.argsort(list(map(getInt,fns)))
    orderedFiles = [fns[i] for i in L]

    numThreads = 15
    pool_manager = Pool(numThreads)
    files_and_args = [(f, args) for f in orderedFiles]
    out = list(pool_manager.map(loadfile, files_and_args))
    out = np.array(out) 
    with open(out_fn,'wb') as fid:
        pickle.dump(out,fid)
