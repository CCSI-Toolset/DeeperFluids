from .utils import pkl_load, lock_file_for_MP, get_simLength, get_sims
import numpy as np
import os
import torch
from args import hash_lin_hyperparams, hash_e2e_hyperparams
import pandas as pd
from .LIN import get_folder_paths as lin_get_folder_paths
from .End2End_Finetune import get_folder_paths as e2e_get_folder_paths
from .models import EndToEndSurrogate, get_latent_vectors
from timeit import default_timer as timer

def create_from_args(args, e2e=False):
    if args.meta_data == 'PNNL':
        if not e2e:
            _, _, iter_folder = lin_get_folder_paths(args)
            model_path = os.path.join(iter_folder, "last_lin_model.pth")
        else:
            _, _, iter_folder = e2e_get_folder_paths(args)  
            model_path = os.path.join(iter_folder, "last_e2e_model.pth")
             
        rollout_IA_path = os.path.join(iter_folder,'rollout_IAs') 
        status_path = os.path.join(args.meta_outputDir, 'experiment_status.csv')

        # get and compare to STAR-CCM IAs if they were provided
        if os.path.exists(os.path.join(get_STARCCM_path(args), f'001_interfaceareamonitor.csv')):
            starccm_IAs = {}
            for sim in range(1,51):
                with open(os.path.join(get_STARCCM_path(args), f'{sim:03d}_interfaceareamonitor.csv')) as f:
                    x = f.readlines()
                    IA = [float(xi.split(',')[-1]) for xi in x[1:]][::10]
                    assert len(IA) == 500
                    starccm_IAs[sim] = np.array(IA)
            # this comes from deeper_fluids/IA.py
            rollout_IA_data = pkl_load(rollout_IA_path)
            idx = int(args.meta_testSplit*get_sims(args))
            # adding 1 to these inds because that makes the indices correspond to the sim # in the raw data
            indices = np.linspace(1,get_sims(args)-2,idx).astype('int') + 1
            rel_error = []
            for i in indices:
                rel_error.append(np.abs(rollout_IA_data['test'][i][-1] - starccm_IAs[i][-1]) / starccm_IAs[i][-1])
            rel_error = np.mean(rel_error)
            if not e2e:
                lock_file_for_MP(status_path, track_experiment_status, args=args, mean_error_final = rel_error)
            else:
                lock_file_for_MP(status_path, track_experiment_status_e2e, args=args, mean_error_final = rel_error)

        
        # compute the time it takes to run a rollout on average
        best_elapsed_time_per_sim = lock_file_for_MP(model_path, time_surrogate_simulation,
                                                     model_path=model_path, args=args)
        if not e2e:
            lock_file_for_MP(status_path, track_experiment_status, args=args,
                                     seconds_per_sim = best_elapsed_time_per_sim)
        else:
            lock_file_for_MP(status_path, track_experiment_status_e2e, args=args, 
                                        seconds_per_sim = best_elapsed_time_per_sim)

    else:
        return 0 # no other data set up yet, just PNNL

    return 1 


def get_STARCCM_path(args):
    if args.meta_STARCCM_IA is None:
        return "data/PNNL/STARCCM_IA/"
    return args.meta_STARCCM_IA

def track_experiment_status(args=None, mean_error_final=None,
                                            seconds_per_sim=None):

    _, path_to_hyperparams_of_this_config, _ = lin_get_folder_paths(args)

    # have the rollouts been run?
    # check the master results file 
    status_path = os.path.join(args.meta_outputDir, 'experiment_status.csv')
    lin_hyperparams = hash_lin_hyperparams(args)
    status = pd.read_csv(status_path)
    row = status.query("lin_hyperparams==@lin_hyperparams"
                        ).query("lin_ConfigIter==@args.lin_ConfigIter").index
    assert len(row) == 1, 'somehow wrote this experiment to this file more than once'

    if 'STARCCM_final_step_IA_error' not in status.columns:
        status['STARCCM_final_step_IA_error'] = None
    if 'seconds_per_sim' not in status.columns:
        status['seconds_per_sim'] = None
    if mean_error_final is not None:
        status.at[row[0],'STARCCM_final_step_IA_error'] = mean_error_final
        status.to_csv(status_path, index=False)
    elif seconds_per_sim is not None:
        status.at[row[0],'seconds_per_sim'] = seconds_per_sim
        status.to_csv(status_path, index=False)
    return 0

def track_experiment_status_e2e(args=None, mean_error_final=None,
                                            seconds_per_sim=None):

    _, path_to_hyperparams_of_this_config, _ = e2e_get_folder_paths(args)

    # have the rollouts been run?
    # check the master results file 
    status_path = os.path.join(args.meta_outputDir, 'experiment_status.csv')
    e2e_hyperparams = hash_e2e_hyperparams(args)
    status = pd.read_csv(status_path)
    row = status.query("e2e_hyperparams==@e2e_hyperparams"
                        ).query("e2e_ConfigIter==@args.e2e_ConfigIter").index
    assert len(row) == 1, 'somehow wrote this experiment to this file more than once'

    if 'STARCCM_final_step_IA_error' not in status.columns:
        status['STARCCM_final_step_IA_error'] = None
    if 'seconds_per_sim' not in status.columns:
        status['seconds_per_sim'] = None
    if mean_error_final is not None:
        status.at[row[0],'STARCCM_final_step_IA_error'] = mean_error_final
        status.to_csv(status_path, index=False)
    elif seconds_per_sim is not None:
        status.at[row[0],'seconds_per_sim'] = seconds_per_sim
        status.to_csv(status_path, index=False)
    return 0

def time_surrogate_simulation(model_path, args):
    '''
    adapted from write_out_surrogate_frames() of LIN.py
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bs = 4
   
    # leave memory at its value from training, 
    # as it won't affect the frame data when window=simLength-1
    # but will enable the LIN to use its window value from during training
    override_args = {'lin_window': get_simLength(args)-1, # ensures that the full rollout is run from step 0
                    'batch_size': bs, 'shuffle': False, 'drop_last': False, 
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
    model.eval()
    best_elapsed_time_per_sim = 1e6
    with torch.no_grad():
        for batch in model.trainDataLoader:
            assert len(batch[0]) == bs, f'{len(batch[0])} elements found but expected {bs} elements in batch'
            start = timer()
            rel_error, _, U_y_hat = model.frame_wise_L2_rel_error(batch, (mean, std))
            end = timer()
            elapsed_time_per_sim = (end - start)/len(batch[0]) # Time in seconds per simulation
            print(f'Training rel error {rel_error}', flush=True)
            print(f'Elapsed time per sim was {elapsed_time_per_sim}', flush=True)
            if elapsed_time_per_sim<best_elapsed_time_per_sim:
                best_elapsed_time_per_sim = elapsed_time_per_sim
        for batch in model.testDataLoader:
            start = timer()
            rel_error, _, U_y_hat = model.frame_wise_L2_rel_error(batch, (mean, std))
            end = timer()
            elapsed_time_per_sim = (end - start)/len(batch[0]) # Time in seconds per simulation
            print(f'Test rel error {rel_error}', flush=True)
            print(f'Elapsed time per sim was {elapsed_time_per_sim}', flush=True)
            if elapsed_time_per_sim<best_elapsed_time_per_sim:
                best_elapsed_time_per_sim = elapsed_time_per_sim
    return best_elapsed_time_per_sim
