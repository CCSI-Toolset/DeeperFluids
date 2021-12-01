import pandas as pd
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.cuda as cuda
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .LIN import get_folder_paths as LIN_get_folder_paths
from .LIN import validEpoch, write_out_surrogate_frames
from .utils import lock_file_for_MP, printNumModelParams
from .models import EndToEndSurrogate
from args import write_args_to_file, hash_e2e_hyperparams

def create_from_args(args):
  
    # has this LIN been finetuned?
    '''
    using the hash of the hyperparameter values, add this experiment to master experiment status file.
    if it already exists in the master status file and the LIN rollouts have been created, return 1.
    otherwise, create the rollout (and first finetune the LIN if it hasn't been finetuned)
    '''
    # were the rollouts created?
    status_path = os.path.join(args.meta_outputDir, 'experiment_status.csv')
    rollouts_created = lock_file_for_MP(status_path, track_experiment_status, args=args)
    if rollouts_created:
        return 1

    # then make them!
    _, _, iter_folder = get_folder_paths(args)
    model_path = os.path.join(iter_folder, "last_e2e_model.pth")
    bestLoss = np.infty
    if args.meta_data == 'PNNL':
        # build them here.
        # only one job/process gets to work on the lin (unlike the grid, where multiple calls to grid.py can be made simultaneously).
        # use a different args.e2e_ConfigIter to have multiple jobs finetune lins at the same time using the same config,
        # or use different configs for each job 
        bestLoss = lock_file_for_MP(model_path, e2e_finetune_LIN, args=args, iter_folder=iter_folder, model_path=model_path)
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
    e2e_hyperparams = hash_e2e_hyperparams(args)
    status = pd.read_csv(status_path)
    status.loc[(status.e2e_hyperparams==e2e_hyperparams) & 
                    (status.e2e_ConfigIter == args.e2e_ConfigIter), 'linTestLoss'] = rel_error
    status.loc[(status.e2e_hyperparams==e2e_hyperparams) & 
                    (status.e2e_ConfigIter == args.e2e_ConfigIter), 'rollouts_created'] = 1
    status.loc[(status.e2e_hyperparams==e2e_hyperparams) & 
                    (status.e2e_ConfigIter == args.e2e_ConfigIter), 'lin_param_count'] = sum(p.numel()
                                                                     for p in model.LIN.parameters())
    status.loc[(status.e2e_hyperparams==e2e_hyperparams) & 
                    (status.e2e_ConfigIter == args.e2e_ConfigIter), 'encoder_param_count'] = sum(p.numel()
                                                                     for p in model.encoder.parameters())
    status.loc[(status.e2e_hyperparams==e2e_hyperparams) & 
                    (status.e2e_ConfigIter == args.e2e_ConfigIter), 'decoder_param_count'] = sum(p.numel()
                                                                     for p in model.decoder.parameters())
    status.to_csv(status_path, index=False)
    return 1


def get_folder_paths(args):

    e2e_hyperparams = hash_e2e_hyperparams(args)
    config_folder = os.path.join(args.meta_outputDir, 'e2e_' + e2e_hyperparams)
    iter_zero_folder = os.path.join(config_folder, str(0))
    Path(iter_zero_folder).mkdir(parents=True, exist_ok=True)
    path_to_hyperparams_of_this_config = os.path.join(iter_zero_folder, 'hyperparameters.log')
    iter_folder = os.path.join(config_folder, str(args.e2e_ConfigIter))
    Path(iter_folder).mkdir(parents=True, exist_ok=True)

    return config_folder, path_to_hyperparams_of_this_config, iter_folder


def track_experiment_status(args=None):

    _, path_to_hyperparams_of_this_config, _ = get_folder_paths(args)

    if not os.path.exists(path_to_hyperparams_of_this_config):
        # write out the config used to make the e2e_ConfigIter=0 folder
        # (even if we're doing a replicate with e2e_ConfigIter!=0)
        assert args.e2e_ConfigIter==0, 'this is a new hyperparameter configuration but you do not have e2e_ConfigIter=0' 
        write_args_to_file(args, path_to_hyperparams_of_this_config)

    # have the rollouts been run?
    # check the master results file 
    status_path = os.path.join(args.meta_outputDir, 'experiment_status.csv')
    rollouts_created = 0
    e2e_hyperparams = hash_e2e_hyperparams(args)
    print(f'E2E hyperparams are {e2e_hyperparams}')
    num_rows_of_results_for_this_config = 0
    old_status = pd.read_csv(status_path)
    if 'e2e_hyperparams' in old_status.columns:
        old_status_of_rollouts = old_status.query("e2e_hyperparams==@e2e_hyperparams"
                                    ).query("e2e_ConfigIter==@args.e2e_ConfigIter")['rollouts_created'].values
        num_rows_of_results_for_this_config = len(old_status_of_rollouts)
        if num_rows_of_results_for_this_config == 1:
            rollouts_created = old_status_of_rollouts[0]
        assert num_rows_of_results_for_this_config <= 1, 'somehow wrote this experiment to this file more than once'

    if rollouts_created:
        return 1

    current_status = pd.DataFrame({k:str(v) for k,v in args.__dict__.items() if k[:3]!='run'}, index=[0])
    current_status['e2e_hyperparams'] = e2e_hyperparams
    current_status['rollouts_created'] = 0
    current_status['linTestLoss'] = np.infty
    if num_rows_of_results_for_this_config == 0: # status file exists, but this experiment has not been seen yet
        updated_status = old_status.append(current_status, ignore_index=True)
        updated_status.to_csv(status_path, index=False)
    else:
        pass # status file exists, and this experiment has been seen before but is not done running (no need to update status)
    return 0


def e2e_finetune_LIN(args=None, iter_folder=None, model_path=None):
    # ## This code finetunes a LIN created with the pnnl dataset.
    lr_tolerance = 5e-9

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
        assert 'encoder_state_dict' in checkpoint
        last_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr'] 
        if last_lr < lr_tolerance or checkpoint['last_epoch'] == args.e2e_max_epochs:
            print('\nDone training!\nLR decayed to {} and {}/{} epochs were used.\n'.format(last_lr, checkpoint['last_epoch'], args.e2e_max_epochs))
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
    override_args = {'end_to_end': True,
                    'lin_window': args.e2e_window,
                    'batch_size': 4, # to prevent out-of-memory issues
                    'e2e_LIN_initial_state': os.path.join(LIN_get_folder_paths(args)[2], "best_lin_model.pth")}
    # to maintain batch_size while preventing out-of-memory issues
    bs_over_4 = (int(args.e2e_batch_size) if args.e2e_batch_size else 4) // 4 
    model = EndToEndSurrogate(args, override_args)
    model = model.to(device)
    printNumModelParams(model.LIN)
    printNumModelParams(model.encoder), printNumModelParams(model.decoder)

    # check
    x, _, p_x, p_y, start =[v.to(device) for v in  next(iter(model.trainDataLoader))]
    latent_vecs, _ = model(x, p_x=p_x, p_y=p_y, start=start)
    assert np.allclose(latent_vecs[:,:,-2:].detach().cpu(), p_y.cpu()), (latent_vecs[:,:,-2:].detach().cpu(), p_y.cpu())

    # set up LR
    max_lr = args.e2e_LR
    opt = torch.optim.Adam(model.parameters(), lr=max_lr) #betas=(.5,.999))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=args.e2e_patience, eps=1e-10)

    # create a summary writer.
    train_writer = SummaryWriter(os.path.join(tensorboard_direc, 'train'))
    test_writer = SummaryWriter(os.path.join(tensorboard_direc, 'valid'))
    tensorboard_recorder_step = 0
    total_steps = 0
    tensorboard_rate = 5

    # resume from checkpoint if available
    if checkpoint:
        model.load_from_checkpoint(checkpoint) # this loads the updated LIN, encoder, and decoder
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['last_epoch'] + 1
        bestLoss = torch.load(model_path.replace('last_','best_'))['loss']
        
    model.eval()
    valLoss, rmse = validEpoch(model.testDataLoader, test_writer, model,  device, 0, tb=False, return_rmse=True)
    print("\nvalLoss: {:.4e}".format(valLoss))
    if not checkpoint:
        try:
            file_size = os.path.getsize(model_path.replace('last_','best_'))
        except os.error:
            file_size = 0
        if file_size==0: # first time that the code has gotten here (so no untrained model saved yet) 
            model.save_checkpoint(model_path.replace('last_','best_'), 
                                    epoch=0, opt=opt, lr_scheduler=lr_scheduler, loss=valLoss)
            test_writer.add_scalar(tag="Loss", scalar_value=valLoss, global_step=0)
            test_writer.add_scalar(tag='rmse', scalar_value=rmse, global_step=0)
        else: # we didn't finish an epoch yet (so no checkpoint), but we did save the untrained model (code has been here)
            assert valLoss == torch.load(model_path.replace('last_','best_'))['loss']
        bestLoss = valLoss
        start_epoch = 1

    print('---------- Started Training ----------')
    for epoch in tqdm(range(start_epoch, args.e2e_max_epochs+1)):  # loop over the dataset multiple times

        if opt.param_groups[0]['lr'] < lr_tolerance:
            print(f'LR decayed to <{lr_tolerance}, ending training')
            break
        
        print("--- Epoch {0}/{1} ---".format(epoch, args.e2e_max_epochs))
        
        model.train()
        trainLoss = trainEpoch(model.trainDataLoader, train_writer, model, opt, 
                                        lr_scheduler, device, epoch, bs_over_4)
        
        print("trainLoss: {:.4e}".format(trainLoss))
        print("LR: {:.4e}".format(opt.param_groups[0]['lr']))
            
        model.eval()
        valLoss = validEpoch(model.testDataLoader, test_writer, model, device, epoch)
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
               lr_scheduler, device, tensorboard_recorder_step, bs_over_4):
    running_loss = 0.0
    running_rmse = 0.0               
    for i, sampleBatch in enumerate(myDataLoader, start=1):

        # --- Main Training ---

        # zero the parameter gradients
        if i % bs_over_4 == 1 or bs_over_4==1:
            opt.zero_grad()

        loss, rmse, _ = model.loss([v.to(device) for v in sampleBatch])
        loss.backward()
        if i % bs_over_4 == 0:
            opt.step()
        
        # loss
        running_loss += loss.item()
        running_rmse += rmse.item()

    # tensorboard writes
    tensorboard_writer.add_scalar(tag="LR", scalar_value=opt.param_groups[0]['lr'], global_step=tensorboard_recorder_step)
    tensorboard_writer.add_scalar(tag="Loss", scalar_value=running_loss/len(myDataLoader), global_step=tensorboard_recorder_step)
    tensorboard_writer.add_scalar(tag='rmse', scalar_value=running_rmse/len(myDataLoader), global_step=tensorboard_recorder_step)

    return running_loss/len(myDataLoader)
