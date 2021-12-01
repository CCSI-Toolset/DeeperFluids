import pandas as pd
import os
from pathlib import Path
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.cuda as cuda
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .utils import ( lock_file_for_MP, printNumModelParams, rmse, jacobian, stream2uv, get_PNNL_train_test_data,
                     L1_loss, jacobian_loss, L2_relative_loss, MSE)
from .models import get_lv_model
from args import hash_latent_vector_hyperparams, write_args_to_file


def create_from_args(args):
    '''
    using the hash of the hyperparameter values, add this experiment to (and possibly create) master experiment status file.
    if it already exists in the master status file and the latent vectors have been created, return 1 (latent vectors exist).
    otherwise, build latent vectors (by training AE/SVD model)
    '''
    # were the latents created?
    status_path = os.path.join(args.meta_outputDir, 'experiment_status.csv')
    latents_created = lock_file_for_MP(status_path, track_experiment_status, args=args)
    if latents_created:
        return 1

    # then make them!
    _, _, iter_folder = get_folder_paths(args)
    model_path = os.path.join(iter_folder, "last_latent_vector_model.pth")
    bestLoss = np.infty
    if args.meta_data == 'PNNL':
        # build them here.
        # only one job/process gets to work on the latent vectors (unlike the grid, where multiple calls to grid.py can be made simultaneously).
        # use a different args.lv_ConfigIter to have multiple jobs create latent vectors at the same time using the same config,
        # or use different configs for each job 
        bestLoss = lock_file_for_MP(model_path, train_AE, args=args, iter_folder=iter_folder, model_path=model_path)
        if bestLoss == None:
            print("didn't get lock on model_path---it's still being updated or there is some issue preventing a lock on the path")
            return 0
    
    # write out the associated best loss and latent vectors
    assert bestLoss < np.infty, 'bestLoss should be set now'
    vecs_path = os.path.join(iter_folder, "latent_vectors.pth")
    write_out_vecs(vecs_path, model_path, args)
    return lock_file_for_MP(status_path, record_completed_status, args=args, bestLoss=bestLoss)


def record_completed_status(args=None, bestLoss=None):
    
    status_path = os.path.join(args.meta_outputDir, 'experiment_status.csv')
    lv_hyperparams = hash_latent_vector_hyperparams(args)
    status = pd.read_csv(status_path)
    status.loc[(status.lv_hyperparams==lv_hyperparams) & 
                    (status.lv_ConfigIter == args.lv_ConfigIter), 'latentTestLoss'] = bestLoss
    status.loc[(status.lv_hyperparams==lv_hyperparams) & 
                    (status.lv_ConfigIter == args.lv_ConfigIter), 'latents_created'] = 1
    status.to_csv(status_path, index=False)

    return 1


def get_folder_paths(args):

    lv_hyperparams = hash_latent_vector_hyperparams(args)
    config_folder = os.path.join(args.meta_outputDir, 'lv_' + lv_hyperparams)
    iter_zero_folder = os.path.join(config_folder, str(0))
    Path(iter_zero_folder).mkdir(parents=True, exist_ok=True)
    path_to_hyperparams_of_this_config = os.path.join(iter_zero_folder, 'hyperparameters.log')
    iter_folder = os.path.join(config_folder, str(args.lv_ConfigIter))
    Path(iter_folder).mkdir(parents=True, exist_ok=True)

    return config_folder, path_to_hyperparams_of_this_config, iter_folder


def track_experiment_status(args=None):

    _, path_to_hyperparams_of_this_config, _ = get_folder_paths(args)

    if not os.path.exists(path_to_hyperparams_of_this_config):
        # write out the config used to make these latent vectors at the latentConfigIter=0 folder
        # (even if we're doing a replicate with latentConfigIter!=0)
        assert args.lv_ConfigIter==0, 'this is a new hyperparameter configuration but you do not have latentConfigIter=0' 
        write_args_to_file(args, path_to_hyperparams_of_this_config)

    # are the necessary latent vectors already made?
    # check the master results file 
    status_path = os.path.join(args.meta_outputDir, 'experiment_status.csv')
    create_status_file = 0
    latents_created = 0
    lv_hyperparams = hash_latent_vector_hyperparams(args) 
    print(f'lV hyperparameter hash: {lv_hyperparams}')
    try:
        old_status = pd.read_csv(status_path)
        old_status_of_latents = old_status.query("lv_hyperparams==@lv_hyperparams"
                                 ).query("lv_ConfigIter==@args.lv_ConfigIter")['latents_created'].values
        num_rows_of_results_for_this_config = len(old_status_of_latents)
        if num_rows_of_results_for_this_config == 1:
            latents_created = old_status_of_latents[0]
        assert num_rows_of_results_for_this_config <= 1, 'somehow wrote this experiment to this file more than once'
    except pd.errors.EmptyDataError:
        create_status_file = 1

    if latents_created:
        return 1

    current_status = pd.DataFrame({k:str(v) for k,v in args.__dict__.items() if k[:3]!='run'}, index=[0])
    current_status['lv_hyperparams'] = lv_hyperparams
    current_status['latents_created'] = 0
    current_status['latentTestLoss'] = np.infty
    if create_status_file: # status file doesn't exist, write it out
        current_status.to_csv(status_path, index=False)
    elif num_rows_of_results_for_this_config == 0: # status file exists, but this experiment has not been seen yet
        updated_status = old_status.append(current_status, ignore_index=True)
        updated_status.to_csv(status_path, index=False)
    else:
        pass # status file exists, and this experiment has been seen before but is not done running (no need to update status)
    return 0


def write_out_vecs(vecs_path, model_path, args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_lv_model(args)
    model = model.to(device)
    if len(args.meta_gpuIDs.split(',')) > 1:
        model = nn.DataParallel(model)

    trainDataset, testDataset = get_PNNL_train_test_data(args)
    trainDataLoader = DataLoader(dataset=trainDataset, batch_size=128, num_workers=args.meta_workers)
    testDataLoader = DataLoader(dataset=testDataset, batch_size=128, num_workers=args.meta_workers)
    X, p_x = next(iter(trainDataLoader))
    X, p_x = X.to(device), p_x.to(device)
    Xhat,z = model(X, p_x)
    Xhat.shape, z.shape
    assert np.allclose(z[:,-2:].detach().cpu(),p_x.cpu())
    checkpoint = torch.load(model_path.replace('last_','best_'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    train_vecs = []
    test_vecs = []
    with torch.no_grad():
        for batch in trainDataLoader:
            x, p_x = batch
            x, p_x = x.to(device), p_x.to(device)
            _, z = model(x, p_x)
            train_vecs.append(z.cpu())
        for batch in testDataLoader:
            x, p_x = batch
            x, p_x = x.to(device), p_x.to(device)
            _, z = model(x, p_x)
            test_vecs.append(z.cpu())
    torch.save({'train': train_vecs, 'test': test_vecs,
                'frame_mean': trainDataset.mean, 'frame_std': trainDataset.std}, vecs_path)


def train_AE(args=None, iter_folder=None, model_path=None):
    # ## This code trains an encoder/decoder for 1 channel of the pnnl datasets.

    # checkpoint directory
    tensorboard_direc = os.path.join(iter_folder,"tb")

    # IF RESUMING: load optimizer and model state dictionaries first
    try:
        file_size = os.path.getsize(model_path)
    except os.error:
        file_size = 0
    checkpoint = None
    if file_size > 100:
        checkpoint = torch.load(model_path)
        last_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr'] 
        if last_lr < 5e-8 or checkpoint['last_epoch'] == args.lv_max_epochs:
            print('\nDone training!\nLR decayed to {} and {}/{} epochs were used.\n'.format(last_lr, checkpoint['last_epoch'], args.lv_max_epochs))
            best = torch.load(model_path.replace('last_','best_'))
            print('\nBest loss was {}\n'.format(best['loss']))
            return best['loss']

    # set up device/GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        print(cuda.is_available())
        print(cuda.device_count())
        print(cuda.current_device())
        print(cuda.get_device_name())

    # set up model
    model = get_lv_model(args)
    model = model.to(device)
    if len(args.meta_gpuIDs.split(',')) > 1:
        model = nn.DataParallel(model)
    printNumModelParams(model)

    # set up data
    trainDataset, testDataset = get_PNNL_train_test_data(args)
    trainDataLoader = DataLoader(dataset=trainDataset, batch_size=args.lv_batch_size, shuffle=True, pin_memory= True, 
                                 drop_last=True, num_workers=args.meta_workers)
    testDataLoader = DataLoader(dataset=testDataset, batch_size=args.lv_batch_size, pin_memory= True, num_workers=args.meta_workers)

    X, p_x = next(iter(trainDataLoader))
    X, p_x = X.to(device), p_x.to(device)
    Xhat,z = model(X, p_x)
    Xhat.shape, z.shape
    assert np.allclose(z[:,-2:].detach().cpu(),p_x.cpu()) # the physics vars p_x always occupy the last 2 coordinates of the latent vector z

    # set up loss
    def loss(pred, target, device):
        
        if args.lv_createStreamFcn:
            pred = stream2uv(pred, device)
            
        loss_dict = {'L1_loss':L1_loss, 
                    'jacobian_loss':jacobian_loss,
                    'L2_relative_loss': L2_relative_loss,
                    'MSE':MSE}
        config_loss = loss_dict[args.lv_Loss]
        L = config_loss(pred, target, args)
        Lj = 0
        if args.lv_doJacobian:
            Lj = jacobian_loss(pred, target, device)
            
        return L + Lj

    # set up LR
    max_lr = args.lv_LR
    opt = torch.optim.Adam(model.parameters(),lr=max_lr) #betas=(.5,.999))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=args.lv_patience)

    # create a summary writer.
    train_writer = SummaryWriter(os.path.join(tensorboard_direc, 'train'))
    test_writer = SummaryWriter(os.path.join(tensorboard_direc, 'valid'))

    # resume from checkpoint if available
    if checkpoint:
        if not isinstance(model, torch.nn.DataParallel):
            model.load_state_dict({(k if k[:7]!='module.' else k[7:]):v for k,v in checkpoint['model_state_dict'].items()})
        else:
            model.load_state_dict({(k if k[:7]=='module.' else 'module.'+k):v for k,v in checkpoint['model_state_dict'].items()})
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['last_epoch'] + 1
        best = torch.load(model_path.replace('last_','best_'))
        bestLoss = best['loss']

    model.eval()
    valLoss, rmse_val = validEpoch(testDataLoader, test_writer, model, loss, rmse, device, 0, tb=False, return_rmse=True)
    print("\nvalLoss: {:.4e}".format(valLoss))
    if not checkpoint:
        torch.save({
                'last_epoch': 0,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': valLoss
                }, model_path.replace('last_','best_'))
        test_writer.add_scalar(tag="Loss", scalar_value=valLoss, global_step=0)
        test_writer.add_scalar(tag='rmse', scalar_value=rmse_val, global_step=0)
        if args.lv_Model=='SVD' and not args.lv_update_SVD:
            return valLoss
        bestLoss = valLoss
        start_epoch = 1

    print('---------- Started Training ----------')
    for epoch in tqdm(range(start_epoch, args.lv_max_epochs+1)):  # loop over the dataset multiple times
        
        if opt.param_groups[0]['lr'] < 5e-8:
            print('LR decayed to <5e-8, ending training')
            break
        
        print("--- Epoch {0}/{1} ---".format(epoch, args.lv_max_epochs))
        
        model.train()
        trainLoss = trainEpoch(trainDataLoader, train_writer, model, opt, loss,
                                                  rmse, lr_scheduler, device, epoch)
        
        print("trainLoss: {:.4e}".format(trainLoss))
        print("LR: {:.4e}".format(opt.param_groups[0]['lr']))
            
        model.eval()
        valLoss = validEpoch(testDataLoader, test_writer, model, loss, rmse, device, epoch)
        print("valLoss: {:.4e}".format(valLoss))
        
        lr_scheduler.step(valLoss)

        # checkpoint progress
        torch.save({
            'last_epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': valLoss
            }, model_path)
        if valLoss < bestLoss:
            print("\nBetter valLoss: {:.4e}, Saving models...".format(bestLoss))
            torch.save({
                'last_epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': valLoss
                }, model_path.replace('last_','best_'))
            bestLoss = valLoss
    
    print('---------- Finished Training ----------')

    return bestLoss


def trainEpoch(myDataLoader, tensorboard_writer, model, opt, loss,
               metric, lr_scheduler, device, tensorboard_recorder_step):
    running_loss = 0.0
    running_rmse = 0.0
    for i, sampleBatch in enumerate(myDataLoader, start=1):

        # --- Main Training ---
        
        # gpu
        X, p_x = sampleBatch
        X = X.to(device)
        p_x = p_x.to(device)

        # zero the parameter gradients
        opt.zero_grad()

        X_hat, _ = model(X, p_x)
        combined_loss = loss(X_hat,X,device) 
        combined_loss.backward()
        opt.step()
        
        # loss
        running_loss += combined_loss.item()
        
        # RMSE
        r = metric(X_hat, X)
        running_rmse += r

    # tensorboard writes
    tensorboard_writer.add_scalar(tag="LR", scalar_value=opt.param_groups[0]['lr'], global_step=tensorboard_recorder_step)
    tensorboard_writer.add_scalar(tag="Loss", scalar_value=running_loss/len(myDataLoader), global_step=tensorboard_recorder_step)
    tensorboard_writer.add_scalar(tag='rmse', scalar_value=running_rmse/len(myDataLoader), global_step=tensorboard_recorder_step)

    return running_loss/len(myDataLoader)


def validEpoch(myDataLoader, tensorboard_writer, model, loss, metric,
               device, tensorboard_recorder_step, tb=True, return_rmse=False):
    avg_running_loss = 0.0
    avg_running_rmse = 0.0
    for _, sampleBatch in enumerate(myDataLoader, start=1):

        # --- Metrics Recording ---

        # gpu
        X, p_x = sampleBatch
        X = X.to(device)
        p_x = p_x.to(device)
        
        perc = len(X)/len(myDataLoader.dataset)

        # forward, no gradient calculations
        with torch.no_grad():
            X_hat, _ = model(X, p_x)

        # loss
        combined_loss = loss(X_hat,X,device)
        avg_running_loss += perc*(combined_loss.item())

        # metrics
        r = metric(X_hat, X)
        avg_running_rmse += perc*r

    if tb:
        tensorboard_writer.add_scalar(tag="Loss", scalar_value=avg_running_loss, global_step=tensorboard_recorder_step)
        tensorboard_writer.add_scalar(tag='rmse', scalar_value=avg_running_rmse, global_step=tensorboard_recorder_step)
    if return_rmse:
        return avg_running_loss, avg_running_rmse

    return avg_running_loss
