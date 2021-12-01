import pandas as pd
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.cuda as cuda
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import lock_file_for_MP, printNumModelParams, get_simLength, get_sims
from .models import EndToEndSurrogate, get_latent_vectors
from args import hash_lin_hyperparams, write_args_to_file

def create_from_args(args):
  
    # has this LIN been trained?
    '''
    using the hash of the hyperparameter values, add this experiment to master experiment status file.
    if it already exists in the master status file and the LIN rollouts have been created, return 1.
    otherwise, create the rollout (and first train the LIN if it hasn't been trained)
    '''
    # were the rollouts created?
    status_path = os.path.join(args.meta_outputDir, 'experiment_status.csv')
    rollouts_created = lock_file_for_MP(status_path, track_experiment_status, args=args)
    if rollouts_created:
        return 1

    # then make them!
    _, _, iter_folder = get_folder_paths(args)
    model_path = os.path.join(iter_folder, "last_lin_model.pth")
    bestLoss = np.infty
    if args.meta_data == 'PNNL':
        # build them here.
        # only one job/process gets to work on the lin (unlike the grid, where multiple calls to grid.py can be made simultaneously).
        # use a different args.lin_ConfigIter to have multiple jobs create lins at the same time using the same config,
        # or use different configs for each job 
        bestLoss = lock_file_for_MP(model_path, train_LIN, args=args, iter_folder=iter_folder, model_path=model_path)
        if bestLoss == None:
            print("didn't get lock on model_path---it's still being updated or there is some issue preventing a lock on the path")
            return 0
    
    # write out the associated best loss and surrogate-simulated frames
    assert bestLoss < np.infty, 'bestLoss should be set now'
    frames_path = os.path.join(iter_folder, "rollout_frames.pth")
    mean_test_rel_error, model = write_out_surrogate_frames(frames_path, model_path, args)
    return lock_file_for_MP(status_path, record_completed_status, args=args, rel_error=mean_test_rel_error, model=model)


def record_completed_status(args=None, rel_error=None, model=None):
    
    status_path = os.path.join(args.meta_outputDir, 'experiment_status.csv')
    lin_hyperparams = hash_lin_hyperparams(args)
    status = pd.read_csv(status_path)
    status.loc[(status.lin_hyperparams==lin_hyperparams) & 
                    (status.lin_ConfigIter == args.lin_ConfigIter), 'linTestLoss'] = rel_error
    status.loc[(status.lin_hyperparams==lin_hyperparams) & 
                    (status.lin_ConfigIter == args.lin_ConfigIter), 'rollouts_created'] = 1
    status.loc[(status.lin_hyperparams==lin_hyperparams) & 
                    (status.lin_ConfigIter == args.lin_ConfigIter), 'lin_param_count'] = sum(p.numel()
                                                                     for p in model.LIN.parameters())
    status.loc[(status.lin_hyperparams==lin_hyperparams) & 
                    (status.lin_ConfigIter == args.lin_ConfigIter), 'encoder_param_count'] = sum(p.numel()
                                                                     for p in model.encoder.parameters())
    status.loc[(status.lin_hyperparams==lin_hyperparams) & 
                    (status.lin_ConfigIter == args.lin_ConfigIter), 'decoder_param_count'] = sum(p.numel()
                                                                     for p in model.decoder.parameters())
    status.to_csv(status_path, index=False)
    return 1


def get_folder_paths(args):

    lin_hyperparams = hash_lin_hyperparams(args)
    config_folder = os.path.join(args.meta_outputDir, 'lin_' + lin_hyperparams)
    iter_zero_folder = os.path.join(config_folder, str(0))
    Path(iter_zero_folder).mkdir(parents=True, exist_ok=True)
    path_to_hyperparams_of_this_config = os.path.join(iter_zero_folder, 'hyperparameters.log')
    iter_folder = os.path.join(config_folder, str(args.lin_ConfigIter))
    Path(iter_folder).mkdir(parents=True, exist_ok=True)

    return config_folder, path_to_hyperparams_of_this_config, iter_folder


def track_experiment_status(args=None):

    _, path_to_hyperparams_of_this_config, _ = get_folder_paths(args)

    if not os.path.exists(path_to_hyperparams_of_this_config):
        # write out the config used to make this lin lin_ConfigIter=0 folder
        # (even if we're doing a replicate with lin_ConfigIter!=0)
        assert args.lin_ConfigIter==0, 'this is a new hyperparameter configuration but you do not have lin_ConfigIter=0' 
        write_args_to_file(args, path_to_hyperparams_of_this_config)

    # have the rollouts been run?
    # check the master results file 
    status_path = os.path.join(args.meta_outputDir, 'experiment_status.csv')
    rollouts_created = 0
    lin_hyperparams = hash_lin_hyperparams(args)
    print(f'LIN hyperparameter hash: {lin_hyperparams}')
    num_rows_of_results_for_this_config = 0
    old_status = pd.read_csv(status_path)
    if 'lin_hyperparams' in old_status.columns:
        old_status_of_rollouts = old_status.query("lin_hyperparams==@lin_hyperparams"
                                    ).query("lin_ConfigIter==@args.lin_ConfigIter")['rollouts_created'].values
        num_rows_of_results_for_this_config = len(old_status_of_rollouts)
        if num_rows_of_results_for_this_config == 1:
            rollouts_created = old_status_of_rollouts[0]
        assert num_rows_of_results_for_this_config <= 1, 'somehow wrote this experiment to this file more than once'

    if rollouts_created:
        return 1

    current_status = pd.DataFrame({k:str(v) for k,v in args.__dict__.items() if k[:3]!='run'}, index=[0])
    current_status['lin_hyperparams'] = lin_hyperparams
    current_status['rollouts_created'] = 0
    current_status['linTestLoss'] = np.infty
    current_status['lin_param_count'] = 0
    current_status['encoder_param_count'] = 0
    current_status['decoder_param_count'] = 0
    if num_rows_of_results_for_this_config == 0: # status file exists, but this experiment has not been seen yet
        updated_status = old_status.append(current_status, ignore_index=True)
        updated_status.to_csv(status_path, index=False)
    else:
        pass # status file exists, and this experiment has been seen before but is not done running (no need to update status)
    return 0


def write_out_surrogate_frames(frames_path, model_path, args):
    '''
    might be better to just write out vectors, which would 
    allow this code to work with models that can't fit end-to-end model on GPU.
    then we would just use the next program to load the vectors and the decoder
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    # leave memory at its value from training, 
    # as it won't affect the frame data when window=simLength-1
    # but will enable the LIN to use its window value from during training
    override_args = {'lin_window': get_simLength(args)-1, # ensures that the full rollout is run from step 0
                    'batch_size': 4, 'shuffle': False, 'drop_last': False, 
                    'end_to_end': True # required to do reconstruction from latents to frames
                    }   
    model = EndToEndSurrogate(args, override_args)
    assert len(model.trainDataset) == get_sims(args)-int(get_sims(args)*args.meta_testSplit)
    assert len(model.testDataset) == int((args.meta_testSplit)*get_sims(args))
    model = model.to(device)
    mean, std = 0, 1
    if args.lv_standardize:
        mean, std = get_latent_vectors(args, return_mean_std=True)
    model.load_from_checkpoint(model_path.replace('last_','best_'))
    train_frames = []
    test_frames = []
    mean_test_rel_error = 0
    model.eval()
    with torch.no_grad():
        for batch in model.trainDataLoader:
            rel_error, _, U_y_hat = model.frame_wise_L2_rel_error(batch, (mean, std))
            print(f'Training rel error {rel_error}', flush=True)
            train_frames.append(U_y_hat.cpu())
        for batch in model.testDataLoader:
            rel_error, _, U_y_hat = model.frame_wise_L2_rel_error(batch, (mean, std))
            print(f'Test rel error {rel_error}', flush=True)
            test_frames.append(U_y_hat.cpu())
            mean_test_rel_error += len(batch[0])/len(model.testDataLoader.dataset) * rel_error
    torch.save({'train': train_frames, 'test': test_frames}, frames_path)
    return mean_test_rel_error.cpu().item(), model


def train_LIN(args=None, iter_folder=None, model_path=None):
    # ## This code trains a LIN on latent-vectors created with the pnnl datasets.

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
        if last_lr < 5e-8 or checkpoint['last_epoch'] == args.lin_max_epochs:
            print('\nDone training!\nLR decayed to {} and {}/{} epochs were used.\n'.format(last_lr, checkpoint['last_epoch'], args.lin_max_epochs))
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

    # set up model and data
    model = EndToEndSurrogate(args)
    model = model.to(device)
    printNumModelParams(model.LIN)
    if args.lin_end_to_end_training:
        printNumModelParams(model.encoder), printNumModelParams(model.decoder)

    # check
    x, _, p_x, p_y, start =[v.to(device) for v in  next(iter(model.trainDataLoader))]
    latent_vecs, _ = model(x, p_x=p_x, p_y=p_y, start=start)
    assert np.allclose(latent_vecs[:,:,-2:].detach().cpu(), p_y.cpu())

    # set up LR
    max_lr = args.lin_LR
    opt = torch.optim.Adam(model.parameters(),lr=max_lr) #betas=(.5,.999))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=args.lin_patience)

    # create a summary writer.
    train_writer = SummaryWriter(os.path.join(tensorboard_direc, 'train'))
    test_writer = SummaryWriter(os.path.join(tensorboard_direc, 'valid'))

    # resume from checkpoint if available
    if checkpoint:
        model.load_from_checkpoint(checkpoint) # this loads the LIN and (if args.lin_end_to_end_training) the updated encoder+decoder
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['last_epoch'] + 1
        bestLoss = torch.load(model_path.replace('last_','best_'))['loss']
        
    model.eval()
    valLoss, rmse = validEpoch(model.testDataLoader, test_writer, model,  device, 0, tb=False, return_rmse=True)
    print("\nvalLoss: {:.4e}".format(valLoss))
    if not checkpoint:
        model.save_checkpoint(model_path.replace('last_','best_'), 
                                epoch=0, opt=opt, lr_scheduler=lr_scheduler, loss=valLoss)
        test_writer.add_scalar(tag="Loss", scalar_value=valLoss, global_step=0)
        test_writer.add_scalar(tag='rmse', scalar_value=rmse, global_step=0)
        bestLoss = valLoss
        start_epoch = 1

    print('---------- Started Training ----------')
    for epoch in tqdm(range(start_epoch, args.lin_max_epochs+1)):  # loop over the dataset multiple times
        
        if opt.param_groups[0]['lr'] < 5e-8:
            print('LR decayed to <5e-8, ending training')
            break
        
        print("--- Epoch {0}/{1} ---".format(epoch, args.lin_max_epochs))
        
        model.train()
        trainLoss = trainEpoch(model.trainDataLoader, train_writer, model, opt, 
                                        lr_scheduler,  device,  epoch)
        
        print("trainLoss: {:.4e}".format(trainLoss))
        print("LR: {:.4e}".format(opt.param_groups[0]['lr']))
            
        model.eval()
        valLoss = validEpoch(model.testDataLoader, test_writer, model,  device, epoch)
        print("\nvalLoss: {:.4e}".format(valLoss))
        
        lr_scheduler.step(valLoss)

        # checkpoint progress
        model.save_checkpoint(model_path, epoch=epoch, opt=opt, lr_scheduler=lr_scheduler, loss=valLoss)
        if valLoss < bestLoss:
            print("\nBetter valLoss: {:.4e}, Saving models...".format(bestLoss))
            model.save_checkpoint(model_path.replace('last_','best_'), 
                                    epoch=epoch, opt=opt, lr_scheduler=lr_scheduler, loss=valLoss)
            bestLoss = valLoss
    
    print('---------- Finished Training ----------')

    return bestLoss


def trainEpoch(myDataLoader, tensorboard_writer, model, opt, 
               lr_scheduler, device, tensorboard_recorder_step):
    running_loss = 0.0
    running_rmse = 0.0               
    for i, sampleBatch in enumerate(myDataLoader, start=1):

        # --- Main Training ---

        # zero the parameter gradients
        opt.zero_grad()

        loss, rmse, _ = model.loss([v.to(device) for v in sampleBatch])
        loss.backward()
        opt.step()
        
        # loss
        running_loss += loss.item()
        running_rmse += rmse.item()

    # tensorboard writes
    tensorboard_writer.add_scalar(tag="LR", scalar_value=opt.param_groups[0]['lr'], global_step=tensorboard_recorder_step)
    tensorboard_writer.add_scalar(tag="Loss", scalar_value=running_loss/len(myDataLoader), global_step=tensorboard_recorder_step)
    tensorboard_writer.add_scalar(tag='rmse', scalar_value=running_rmse/len(myDataLoader), global_step=tensorboard_recorder_step)

    return running_loss/len(myDataLoader)


def validEpoch(myDataLoader, tensorboard_writer, model, 
               device, tensorboard_recorder_step, tb=True,
                return_rmse=False):
    avg_loss = 0.0
    avg_rmse = 0.0
    for _, sampleBatch in enumerate(myDataLoader, start=1):

        perc = len(sampleBatch[0])/len(myDataLoader.dataset)

        # forward, no gradient calculations
        with torch.no_grad():
            loss, rmse, _ = model.loss([v.to(device) for v in sampleBatch])

        # --- Metrics Recording ---
        avg_loss += perc*(loss.item())
        avg_rmse += perc*(rmse.item())

    if tb:
        tensorboard_writer.add_scalar(tag="Loss", scalar_value=avg_loss, global_step=tensorboard_recorder_step)
        tensorboard_writer.add_scalar(tag='rmse', scalar_value=avg_rmse, global_step=tensorboard_recorder_step)
    if return_rmse:
        return avg_loss, avg_rmse
    return avg_loss
