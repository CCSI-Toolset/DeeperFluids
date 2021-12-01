from filelock import FileLock, Timeout
from pathlib import Path
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from numpy.linalg import svd


def lock_file_for_MP(file_to_lock, update_function, **kwargs_for_update_function):
    '''
    this locks a file to prevent others processes from updating it, useful during multiprocessing.

    the idea is to allow multiple processes/machines to run experiments without duplicating
    work or having out-of-date information.
    '''

    Path(file_to_lock).touch()
    lock_path = file_to_lock + ".lock"
    lock = FileLock(lock_path)
    try:
        with lock.acquire(timeout=5):
            result = update_function(**kwargs_for_update_function)
    except Timeout:
        print("Another instance of this application currently holds the lock.")
        return None

    return result

def get_grid_folder(args):
    return os.path.join(args.meta_outputDir, args.meta_data, 
            "channel_" + str(args.meta_channel), "grid_size_" + str(args.meta_gridSize)
            + ('_' + str(get_Ny(args)) if args.meta_Ny_over_Nx is not None else '') )

def get_Ny(args):
    if args.meta_Ny_over_Nx is None:
        return args.meta_gridSize
    return int(args.meta_gridSize * float(args.meta_Ny_over_Nx))

def get_simLength(args):
    if args.meta_simLength is None:
        return 500
    return int(args.meta_simLength)

def get_sims(args):
    if args.meta_sims is None:
        return 50
    return int(args.meta_sims)

def zip_equal(*iterables):
    # check for length difference
    num = None
    for i, iterable in enumerate(iterables):
        if num is None:
            num = len(iterable)
        else:
            assert len(iterable) == num, f'iterable #{i+1} had length {len(iterable)} instead of {num}'
    # zip('ABCD', 'xy') --> Ax By
    sentinel = object()
    iterators = [iter(it) for it in iterables]
    while iterators:
        result = []
        for it in iterators:
            elem = next(it, sentinel)
            if elem is sentinel:
                return
            result.append(elem)
        yield tuple(result)

def pkl_save(D,fn):
    with open(fn,'wb') as fid:
        pickle.dump(D,fid)

def pkl_load(fn):
    with open(fn,'rb') as fid:
        D = pickle.load(fid)
        return D

class CCSI_2D(Dataset):
    def __init__(self,
                 dataFiles,
                 simLen = 500,
                 numToKeep=np.infty,
                 mean_std=(0,1),
                 window = None, memory = None, args = None, return_start=False):
        '''
        TODO: normalize inlet velocity to be between 0 and 1
                and/or add layer that projects physics vars to feature-map shape
                then adds this to deconv feature map in AE?
        '''
        self.dataFiles = dataFiles
        if numToKeep < len(self.dataFiles):
            self.dataFiles = self.dataFiles[:numToKeep]

        self.numToKeep = numToKeep
        self.simLen = simLen
        self.t = np.linspace(0,1,simLen).astype('float32')
        self.mean, self.std = mean_std
        self.window = window
        self.memory = memory
        self.return_start = return_start

        # Get the inlet velocity
        with open('configs/inlet_velocities.txt') as fid:
            txt = fid.read().splitlines()
        inletVelocity = np.array(list(map(float,txt[1:]))).astype('float32')
        self.inletMx = np.max(inletVelocity)
        self.inletMn = np.min(inletVelocity)

        data = []
        for fn in self.dataFiles:
            idx = int(fn.split('/')[-1].replace('.pkl','')) - 1
            D = pkl_load(fn)
            if len(D.shape) == 3: #add a channel dimension
                D = np.expand_dims(D,1)
            data.append((D,inletVelocity[idx]))

        self.data = data

        if self.window:
            self.samplable_frames_per_sim = self.simLen-self.window
            self.args = args
            self.n_channels = 1 if not hasattr(self.args.meta_channel,'__len__') else len(self.args.meta_channel)
            assert self.data[0][0].shape == (self.simLen, self.n_channels, self.args.meta_gridSize, 
                                            get_Ny(self.args)), self.data[0][0].shape

    def __len__(self):
        if self.window is None:
            return self._default_len()
        else:
            return self._len_with_window()
            
    def _default_len(self):
        return self.simLen*len(self.data)
            
    def _len_with_window(self):
        return self.samplable_frames_per_sim * len(self.data)

    def __getitem__(self, idx):
        if self.window is None:
            return self._default_get(idx)
        else:
            return self._get_with_window(idx)

    def _default_get(self, idx):
        q,r_idx = np.divmod(idx,self.simLen)
        X,p = self.data[q]
        X = X[r_idx].astype('float32')
        p_x = np.hstack([p, self.t[r_idx]])
        X = (X - self.mean) / self.std
        return X, p_x
    
    def _get_with_window(self, idx):
        sim_num, inputs_latest_timestep = np.divmod(idx, self.samplable_frames_per_sim)
        X, inlet_vel = self.data[sim_num]
        X = (X - self.mean) / self.std
        X = X.astype('float32')
        p = np.vstack([[inlet_vel]*self.simLen, self.t]).T

        x = torch.zeros((self.memory, self.n_channels, self.args.meta_gridSize, get_Ny(self.args)))
        p_x = torch.zeros((2, self.memory))

        for i in range(self.memory): # for each timestep of input memory,
            if inputs_latest_timestep - i >= 0: # if the timestep is present in the data (i.e., it's greater than 0)
                x[i] = torch.tensor(X[inputs_latest_timestep - i]) # add it to the input at the appropriate memory slot
        y = torch.tensor(X[inputs_latest_timestep+1 : inputs_latest_timestep+self.window+1])
        
        for i in range(self.memory): # for each timestep of input memory,
            if inputs_latest_timestep - i >= 0: # if the timestep is present in the data (i.e., it's greater than 0)
                p_x[:, i] = torch.tensor(p[inputs_latest_timestep - i]) # add it to the input at the appropriate memory slot
        p_y = torch.tensor(p[inputs_latest_timestep+1 : inputs_latest_timestep+self.window+1])
        
        if not self.return_start:
            return x, y, p_x, p_y
        start = inputs_latest_timestep - self.memory + 1
        return x, y, p_x, p_y, start


def get_PNNL_train_test_data(args, numSamplesToKeep=None, window=None, memory=None, return_start=False):

    numSamplesToKeep = np.infty if not numSamplesToKeep else numSamplesToKeep
    if args.lv_DEBUG:
        numSamplesToKeep = 2
    
    grid_folder = get_grid_folder(args)
    sims = glob(os.path.join(grid_folder,'[0-9][0-9][0-9].pkl'))
    sims = sorted(sims)
    numSims = len(sims)
    idx = int(args.meta_testSplit*numSims)
    testInds = np.linspace(1,numSims-2,idx).astype('int')
    trainInds = list(set(np.arange(0,numSims)).difference(set(testInds)))
    testSimFiles = [sims[idx] for idx in testInds]
    trainSimFiles = [sims[idx] for idx in trainInds]
    if args.meta_data == 'PNNL':
        assert len(testSimFiles) == int(get_sims(args)*args.meta_testSplit), len(testSimFiles)
        assert len(trainSimFiles) == get_sims(args)-int(get_sims(args)*args.meta_testSplit), len(trainSimFiles)

    if args.lv_standardize:
        # TODO: modify for 2+ channel data
        data = []
        for fn in trainSimFiles:
            data.append(pkl_load(fn))
        data = np.array(data)
        mean_std = data.mean(), data.std()
    else:
        mean_std = 0, 1
    trainDataset = CCSI_2D(trainSimFiles, numToKeep=numSamplesToKeep, mean_std = mean_std,
                 window=window, memory=memory, args=args, return_start=return_start)
    testDataset = CCSI_2D(testSimFiles, numToKeep=numSamplesToKeep, mean_std = mean_std, 
                window=window, memory=memory, args=args, return_start=return_start)

    return trainDataset, testDataset
        

def computeSVD(data, simLen):
    # data should of size numFrames x vecLength
    # the columns of spatialVecs are the spatial PODs
    numSamps = len(data)
    data = data.reshape(numSamps,-1)
    spatialVecs, S, vh = svd(data.T, full_matrices=False)
    return spatialVecs, S

def get_SVD_vectors(args=None):
    '''
    computes the SVD of the PNNL training data if not already done,
    returns the U matrix
    '''
    processed_folder = get_grid_folder(args)
    if not args.lv_standardize:
        svdOutFile = os.path.join(processed_folder, "SVD_vectors.pth")
    else:
        svdOutFile = os.path.join(processed_folder, "SVD_vectors_of_standardized_data.pth")
    if os.path.exists(svdOutFile):
        print('loading SVD vectors')
        D = pkl_load(svdOutFile)
        spatialVecs = D['spatialVecs']
    else:
        print('computing SVD vectors')
        trainDataset, _ = get_PNNL_train_test_data(args)
        trainDataLoader = DataLoader(dataset=trainDataset, batch_size=len(trainDataset), num_workers=args.meta_workers)
        X, _ = next(iter(trainDataLoader))
        assert X.shape == torch.Size(((get_sims(args)-int(get_sims(args)*args.meta_testSplit))*get_simLength(args), 
                1, args.meta_gridSize, get_Ny(args))), (
                f'assumed we had 1 channel and 80% train data but got data with shape {X.shape}')
        spatialVecs, S = computeSVD(X, get_simLength(args))
        D = {'spatialVecs':spatialVecs, 'S':S}
        pkl_save(D, svdOutFile)
    return spatialVecs


class CCSI_Latent(Dataset):
    def __init__(self,
                 data,
                 simLen = 500,
                 args = None, override_args = None,
                 numToKeep=np.infty,
                 train = True, abs_x_mx = None, abs_p_mx = None):

        if not override_args:
            override_args = {}

        self.latent_dim = args.meta_latent_dimension
        if train:
            n_sims = get_sims(args)-int(get_sims(args)*args.meta_testSplit)
        else:
            n_sims = int(get_sims(args) * args.meta_testSplit)
        assert torch.cat(data,dim=0).shape == torch.Size([n_sims*simLen, self.latent_dim])
        self.data = torch.cat(data,dim=0).reshape(n_sims, simLen, -1)
        assert self.data.shape[-1] == self.latent_dim, (self.data.shape[-1], self.latent_dim, n_sims, simLen)
        self.n_sims = numToKeep if numToKeep<np.infty else n_sims
        self.data = self.data[:self.n_sims]

        self.simLen = simLen

        self.doPreprocess = args.lin_normalize
        self.w = args.lin_window if 'lin_window' not in override_args else override_args['lin_window']
        self.memory= args.lin_memory
        # TODO: instead of the following checks, use an arg when memory is the last dim
        assert self.memory != 2, 'preprocessing code will quietly break if memory == # physics vars = 2' 
        assert self.memory != args.meta_latent_dimension, 'preprocessing code will quietly break if memory == latent dim' 
        self.samplable_frames_per_sim = self.simLen-self.w

        if self.doPreprocess and train:
            self.prep_for_normalization()
        elif self.doPreprocess:
            self.abs_x_mx = abs_x_mx
            self.abs_p_mx = abs_p_mx

    def prep_for_normalization(self):
        D = np.vstack(self.data)
        assert D.shape == torch.Size([self.simLen * self.n_sims, self.latent_dim])
        self.abs_x_mx = torch.tensor(np.max(abs(D), axis=0))
        self.abs_p_mx = self.abs_x_mx[-2:]
        if torch.cuda.is_available():
            self.cuda_abs_x_mx = self.abs_x_mx.cuda()
            self.cuda_abs_x_mx_extra_dim = self.abs_x_mx[:,None].cuda()
            self.cuda_abs_p_mx = self.abs_p_mx.cuda()
            self.cuda_abs_p_mx_extra_dim = self.abs_p_mx[:,None].cuda()
                 
    def __len__(self):
        return self.samplable_frames_per_sim * len(self.data)

    def preprocess_x(self,x):
        if x.device.type != 'cpu':
            if x.shape[-1] != self.abs_x_mx.shape[0]: # if memory is the last dim
                abs_x_mx = self.cuda_abs_x_mx_extra_dim
            else:
                abs_x_mx = self.cuda_abs_x_mx
            return x/abs_x_mx
        else:
            abs_x_mx = self.abs_x_mx
            if x.shape[-1] != self.abs_x_mx.shape[0]:
                return x/abs_x_mx[:,None]
            return x/abs_x_mx

    def preprocess_p(self,p):
        if p.device.type != 'cpu':
            if p.shape[-1] != self.abs_p_mx.shape[0]:
                abs_p_mx = self.cuda_abs_p_mx_extra_dim
            else:
                abs_p_mx = self.cuda_abs_p_mx
            return p/abs_p_mx
        else:
            abs_p_mx = self.abs_p_mx
            if p.shape[-1] != self.abs_p_mx.shape[0]:
                return p/abs_p_mx[:,None]
            return p/abs_p_mx

    def invPreprocess_x(self,xnew):
        if xnew.device.type != 'cpu':
            abs_x_mx = self.cuda_abs_x_mx
        else:
            abs_x_mx = self.abs_x_mx
        x = xnew*abs_x_mx
        return x

    def invPreprocess_p(self,pnew):
        if pnew.device.type != 'cpu':
            abs_p_mx = self.cuda_abs_p_mx
        else:
            abs_p_mx = self.abs_p_mx
        p = pnew*abs_p_mx
        return p
    
    def __getitem__(self, idx):
        sim_num, inputs_latest_timestep = np.divmod(idx, self.samplable_frames_per_sim)
        X, p = self.data[sim_num], self.data[sim_num][:,-2:]
        
        x = torch.zeros((self.latent_dim, self.memory))
        p_x = torch.zeros((2, self.memory))
        
        for i in range(self.memory): # for each timestep of input memory,
            if inputs_latest_timestep - i >= 0: # if the timestep is present in the data (i.e., it's greater than 0)
                x[:, i] = X[inputs_latest_timestep - i] # add it to the input at the appropriate memory slot
        y = X[inputs_latest_timestep+1 : inputs_latest_timestep+self.w+1]
        
        for i in range(self.memory): # for each timestep of input memory,
            if inputs_latest_timestep - i >= 0: # if the timestep is present in the data (i.e., it's greater than 0)
                p_x[:, i] = p[inputs_latest_timestep - i] # add it to the input at the appropriate memory slot
        p_y = p[inputs_latest_timestep+1 : inputs_latest_timestep+self.w+1]
        
        if self.doPreprocess:
            x = self.preprocess_x(x)
            y = self.preprocess_x(y)
            p_x = self.preprocess_p(p_x)
            p_y = self.preprocess_p(p_y)
        
        start = inputs_latest_timestep - self.memory + 1
        return x, y, p_x, p_y, start

def jacobian(X,device='cpu'):
    f1 = X[:,0,:,:]
    f2 = X[:,1,:,:]

    df1_dx = f1[:,:,1:] - f1[:,:,:-1]
    df1_dx = torch.cat([df1_dx,torch.zeros((f2.shape[0],f2.shape[1],1)).to(device)], axis=2)

    df1_dy = f1[:,1:,:] - f1[:,:-1,:]
    df1_dy = torch.cat([df1_dy,torch.zeros((df1_dy.shape[0],1,f1.shape[2])).to(device)], axis=1)

    df2_dx = f2[:,:,1:] - f2[:,:,:-1]
    df2_dx = torch.cat([df2_dx,torch.zeros((f2.shape[0],f2.shape[1],1)).to(device)], axis=2)

    df2_dy = f2[:,1:,:] - f2[:,:-1,:]
    df2_dy = torch.cat([df2_dy,torch.zeros((df1_dy.shape[0],1,f1.shape[2])).to(device)], axis=1)

    return torch.stack([df1_dx, df1_dy, df2_dx, df2_dy], axis=1)

# http://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node69.html
# When creating the stream function, the second channel of X is not going to be used.
# It's there so we don't have to change the AE model code.
def stream2uv(X,device='cpu'):
    u = X[:,0,1:,:] - X[:,0,:-1,:]
    w = torch.unsqueeze(u[:,-1,:],axis=1)
    u = torch.cat([u,w],axis=1)
    v = X[:,0,:,1:] - X[:,0,:,:-1]
    w = torch.unsqueeze(u[:,:,-1],axis=2)
    v = torch.cat([v,w],axis=2)
    return torch.stack([u,v], axis=1)

def convertSimToImage(X):
    # X = [frames,channels,h,w]
    mid = 128
    M = 255
    mx = X.max()
    mn = X.min()
    X = (X - mn)/(mx - mn)

    #C = np.uint8(M*B)
    C = (M*X).type(torch.uint8)

    if C.shape[1] == 2:
        out_shape = C.shape
        Xrgb = torch.zeros((out_shape[0],3,out_shape[2],out_shape[3])).type(torch.uint8)
        filler = mid*torch.ones(C.shape[2:]).type(torch.uint8)
        filler = filler.unsqueeze(axis=0)
        for idx, frame in enumerate(C):
            #Xrgb[idx] = torch.cat([frame[0].unsqueeze(axis=0),filler,frame[1].unsqueeze(axis=0)],axis=0)
            Xrgb[idx] = torch.cat([frame,filler],axis=0)
            #Xrgb[idx] = torch.cat([filler,frame],axis=0)
    else:
        Xrgb = C
    return Xrgb

def printNumModelParams(model):
    layers_req_grad = 0
    tot_layers = 0

    params_req_grad = 0
    tot_params = 0

    for param in model.named_parameters():
        #print(param[0])
        if (param[1].requires_grad):
            layers_req_grad += 1
            params_req_grad += param[1].nelement()
        tot_layers += 1
        tot_params += param[1].nelement()
    print("{0:,} layers require gradients (unfrozen) out of {1:,} layers".format(layers_req_grad, tot_layers))
    print("{0:,} parameters require gradients (unfrozen) out of {1:,} parameters".format(params_req_grad, tot_params))

def writeMessage(msg, file_name):
    # Write to file.
    print(msg)
    myFile = open(file_name+".txt", "a")
    myFile.write(msg)
    myFile.write("\n")
    myFile.close()

# Loss Functions
def rmse(preds, labels, args=None):
    d = (preds - labels)**2
    d = d.mean()
    try:
        r = d.sqrt()
    except:
        r = np.sqrt(d)
    return r

def MSE(pred, target, args=None):
    return torch.nn.MSELoss()(pred, target)

def L2_relative_vector_loss(pred, target, args=None):
    first_frame_axis = 1
    if len(pred.shape) == 3: # there's a simLen/memory axis present (in addition to batch and latent_vector)
        first_frame_axis = 2
    assert 1<len(pred.shape)<4, pred.shape

    # confirm that this reshaping of pred will cause the axis we compute the norm over to represent a frame/vector
    temp = pred.reshape(torch.prod(torch.tensor(pred.shape[:first_frame_axis])), -1)
    assert temp.shape[-1] ==  args.meta_latent_dimension, temp.shape
    if len(pred.shape) == 3:
        assert (temp[args.lin_window-1] == pred[0,args.lin_window-1].flatten()).all(), pred.shape
    else:
        assert (temp[0] == pred[0].flatten()).all()

    # combine the batch and simLen axes into the 1st axis, then compute the norm per frame (by operating across 2nd axis).
    # each frame, the relative error is norm(pred-target)/norm(target), then return the mean of this rel error across frames
    return torch.mean(torch.norm((pred - target).reshape(torch.prod(torch.tensor(pred.shape[:first_frame_axis])), -1), dim=1) /
                         torch.norm(target.reshape(torch.prod(torch.tensor(pred.shape[:first_frame_axis])), -1), dim=1) )

    
def L2_relative_loss(pred, target, args=None):
    '''
    frame version of L2 relative loss
    '''
    first_frame_axis = 1
    if len(pred.shape) == 5: # there's a simLen/memory axis present (in addition to batch, ch, nx, ny)
        first_frame_axis = 2
    assert 3<len(pred.shape)<6, pred.shape

    # confirm that this reshaping of pred will cause the axis we compute the norm over to represent a frame
    temp = pred.reshape(torch.prod(torch.tensor(pred.shape[:first_frame_axis])), -1)
    channels = 1
    if hasattr(args.meta_channel,'__len__'):
        channels = len(args.meta_channel)
    assert temp.shape[-1] ==  channels*args.meta_gridSize*get_Ny(args), temp.shape
    if len(pred.shape) == 5:
        assert (temp[args.lin_window-1] == pred[0,args.lin_window-1].flatten()).all(), pred.shape
    else:
        assert (temp[0] == pred[0].flatten()).all()

    # combine the batch and simLen axes into the 1st axis, then compute the norm per frame (by operating across 2nd axis).
    # each frame, the relative error is norm(pred-target)/norm(target), then return the mean of this rel error across frames
    return torch.mean(torch.norm((pred - target).reshape(torch.prod(torch.tensor(pred.shape[:first_frame_axis])), -1), dim=1) /
                         torch.norm(target.reshape(torch.prod(torch.tensor(pred.shape[:first_frame_axis])), -1), dim=1) )

def L1_loss(pred, target, args=None):
    return torch.mean(torch.abs(pred - target))

def jacobian_loss(pred, target, device='cpu'):
    return L1_loss(jacobian(pred, device), jacobian(target, device))
