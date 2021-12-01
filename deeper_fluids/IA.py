from .utils import pkl_load, lock_file_for_MP, pkl_save, zip_equal, get_simLength
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from pathlib import Path
import contextlib
from args import hash_lin_hyperparams, hash_e2e_hyperparams, hash_latent_vector_hyperparams
import pandas as pd
from .LIN import get_folder_paths as lin_get_folder_paths
from .End2End_Finetune import get_folder_paths as e2e_get_folder_paths
from .latent_vectors import get_folder_paths as lv_get_folder_paths
from torch.utils.data import DataLoader
from .models import EndToEndSurrogate
from .utils import get_Ny, get_grid_folder, get_sims

def create_from_args(args, e2e=False):
    if args.meta_data == 'PNNL':
        processed_folder = get_grid_folder(args)
        Path(processed_folder).mkdir(parents=True, exist_ok=True)
        processed_IA_path = os.path.join(processed_folder, 'IA.pkl')
        try:
            file_size = os.path.getsize(processed_IA_path)
        except os.error:
            file_size = 0
        if file_size < 1000:
            print('\nGet the IA for each sim for grid size {}!\n'.format(str(args.meta_gridSize) + 
                ('_' + str(get_Ny(args)) if args.meta_Ny_over_Nx is not None else '^2')))
            IA_dict = lock_file_for_MP(processed_IA_path, get_IA_for_all_sims, args=args)
            if IA_dict:
                pkl_save(IA_dict, processed_IA_path)

        try:
            file_size = os.path.getsize(processed_IA_path)
        except os.error:
            file_size = 0
        if file_size < 1000:
            print('\nIA data does not yet exist. It is possible that another process/job is finishing it.\n')
            return 0

        print('Successfully created baseline IA files!')
       
        if not e2e:
            _, _, iter_folder = lin_get_folder_paths(args)
        else:
            _, _, iter_folder = e2e_get_folder_paths(args)  
        rollout_IA_path = os.path.join(iter_folder,'rollout_IAs') 
        try:
            file_size = os.path.getsize(rollout_IA_path)
        except os.error:
            file_size = 0
        if file_size < 20 and not args.run_LVM_IA_only: #rollout IAs don't exist yet!
            # get rollout IAs and their errors!
            baseline_IAs = pkl_load(processed_IA_path)
            
            test_rollout_IA_data = lock_file_for_MP(rollout_IA_path, IA_error, test=True, 
                                                baseline_IAs=baseline_IAs, args=args, iter_folder=iter_folder)
            train_rollout_IA_data = lock_file_for_MP(rollout_IA_path, IA_error, test=False,
                                                baseline_IAs=baseline_IAs, args=args, iter_folder=iter_folder)
            if test_rollout_IA_data and train_rollout_IA_data:
                rollout_IA_data = {'train': train_rollout_IA_data[0], 
                                    'test': test_rollout_IA_data[0]}
                rollout_IA_errors = {'train': train_rollout_IA_data[1], 
                                    'test': test_rollout_IA_data[1]}
                if not e2e:
                    track_experiment_status(args=args, mean_error_final=test_rollout_IA_data[2])
                else:
                    track_experiment_status_e2e(args=args, mean_error_final=test_rollout_IA_data[2])
                pkl_save(rollout_IA_data, os.path.join(iter_folder,'rollout_IAs'))
                pkl_save(rollout_IA_errors, os.path.join(iter_folder,'rollout_IA_errors'))
            else:
                print('failed to get train and test rollout IA data')
                return 0

        if track_experiment_status_lv(args) == 0:
            print('\ngetting IA errors for LVM reconstructions')
            baseline_IAs = pkl_load(processed_IA_path)
            test_rollout_IA_data = lock_file_for_MP(rollout_IA_path, IA_error, test=True, 
                                        baseline_IAs=baseline_IAs, args=args, LVM=True)
            track_experiment_status_lv(args=args, mean_error_final=test_rollout_IA_data[2])

    else:
        return 0 # no other data set up yet, just PNNL

    return 1 

def track_experiment_status(args=None, mean_error_final=None):

    _, path_to_hyperparams_of_this_config, _ = lin_get_folder_paths(args)

    # have the rollouts been run?
    # check the master results file 
    status_path = os.path.join(args.meta_outputDir, 'experiment_status.csv')
    lin_hyperparams = hash_lin_hyperparams(args)
    num_rows_of_results_for_this_config = 0
    status = pd.read_csv(status_path)
    row = status.query("lin_hyperparams==@lin_hyperparams"
                        ).query("lin_ConfigIter==@args.lin_ConfigIter").index
    assert len(row) == 1, 'somehow wrote this experiment to this file more than once'

    if 'final_step_IA_error' not in status.columns:
        status['final_step_IA_error'] = None
    status.at[row[0],'final_step_IA_error'] = mean_error_final
    status.to_csv(status_path, index=False)
    return 0

def track_experiment_status_e2e(args=None, mean_error_final=None):

    _, path_to_hyperparams_of_this_config, _ = e2e_get_folder_paths(args)

    # have the rollouts been run?
    # check the master results file 
    status_path = os.path.join(args.meta_outputDir, 'experiment_status.csv')
    e2e_hyperparams = hash_e2e_hyperparams(args)
    num_rows_of_results_for_this_config = 0
    status = pd.read_csv(status_path)
    row = status.query("e2e_hyperparams==@e2e_hyperparams"
                        ).query("e2e_ConfigIter==@args.e2e_ConfigIter").index
    assert len(row) == 1, 'somehow wrote this experiment to this file more than once'

    if 'final_step_IA_error' not in status.columns:
        status['final_step_IA_error'] = None
    status.at[row[0],'final_step_IA_error'] = mean_error_final
    status.to_csv(status_path, index=False)
    return 0

def track_experiment_status_lv(args=None, mean_error_final=None):

    _, path_to_hyperparams_of_this_config, _ = lv_get_folder_paths(args)

    # get the master results file 
    status_path = os.path.join(args.meta_outputDir, 'experiment_status.csv')
    lv_hyperparams = hash_latent_vector_hyperparams(args)
    num_rows_of_results_for_this_config = 0
    status = pd.read_csv(status_path)
    row = status.query("lv_hyperparams==@lv_hyperparams"
                        ).query("lv_ConfigIter==@args.lv_ConfigIter"
                        ).query("latents_created==1").index
    assert len(row) == 1, f'somehow wrote this experiment to this file {len(row)} times'

    if 'LVM_recon_final_step_IA_error' in status.columns:
        if not pd.isna(status.at[row[0],'LVM_recon_final_step_IA_error']):
            return 1
    else:
        status['LVM_recon_final_step_IA_error'] = None
    if mean_error_final is not None:
        status.at[row[0],'LVM_recon_final_step_IA_error'] = mean_error_final
        print('\nwriting LVM_recon_final_step_IA_error to status tracker\n')
        status.to_csv(status_path, index=False)
        return 1
    return 0

def LVM_reconstructions(args):
    # decoder of trained LVM associated with these LVM hparams
    encoder, decoder = EndToEndSurrogate.get_lv_model_from_args(None, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decoder = decoder.to(device)
    if len(args.meta_gpuIDs.split(',')) > 1:
        decoder = torch.nn.DataParallel(decoder)
    decoder.eval()
    # dataloader of latent representations of the PNNL test data
    _, _, iter_folder = lv_get_folder_paths(args)
    vecs_path = os.path.join(iter_folder, "latent_vectors.pth")
    latents = torch.load(vecs_path)
    latents['test'] = torch.cat(latents['test'])
    assert len(latents['test']) == get_simLength(args)*int(get_sims(args)*args.meta_testSplit), (
            f'{len(latents["test"])} latents, '
            'args.meta_simLength*args.meta_sims*args.meta_testSplit expected' )
    testDataset = DataLoader(dataset=latents['test'], batch_size=128, num_workers=args.meta_workers)

    def decode(decoder, encoding, args):
        if args.lv_Model != "SVD":
            return decoder(encoding)
        else:
            return decoder(encoding[...,:-2])

    rollouts = []
    for _, encoding in enumerate(testDataset, start=1):  
        # forward, no gradient calculations
        with torch.no_grad():
            X_hat = decode(decoder, encoding.to(device), args)
            assert X_hat.shape == torch.Size([len(encoding), 1, args.meta_gridSize, get_Ny(args)])
            rollouts.append(X_hat)
    rollouts = torch.cat(rollouts)
    assert len(rollouts) == get_simLength(args)*int(get_sims(args)*args.meta_testSplit), (
            f'test data reconstructions have {len(rollouts)} frames, '
            'should be args.meta_simLength*args.meta_sims*args.meta_testSplit')
    # drop first frame for comparison to LIN rollout IAs, which start on 2nd frame
    rollouts = rollouts[np.mod(np.arange(get_simLength(args)*int(get_sims(args)*args.meta_testSplit)),
                                 get_simLength(args)) != 0]
    return torch.reshape(rollouts, [int(get_sims(args)*args.meta_testSplit), 
                get_simLength(args)-1, 1 , args.meta_gridSize, get_Ny(args)]).cpu()

def IA_error(test=None, baseline_IAs=None, args=None, iter_folder=None, LVM=False):
    idx = int(args.meta_testSplit*get_sims(args))
    # adding 1 to these inds because that makes the indices correspond to the sim # in the raw data,
    # which is consistent with the numbering in get_IA_for_all_sims()
    if test:
        indices = np.linspace(1,get_sims(args)-2,idx).astype('int') + 1
    else:
        testInds = np.linspace(1,get_sims(args)-2,idx).astype('int')
        indices = np.array(list(set(np.arange(0,get_sims(args))).difference(set(testInds)))) + 1

    if not LVM:
        rollout_loc = os.path.join(iter_folder, "rollout_frames.pth")
        if test:
            rollouts = torch.load(rollout_loc)['test']  
        else:
            rollouts = torch.load(rollout_loc)['train']
        rollouts = torch.cat(rollouts)
    else:
        rollouts = LVM_reconstructions(args)
    print(f'we got {len(rollouts)} rollouts with shape {rollouts[0].shape}')
    assert rollouts.shape == torch.Size([len(indices), get_simLength(args)-1, 
                                        1, args.meta_gridSize, get_Ny(args)])
    rollout_IAs, rollout_IA_errors = {}, {}
    for i, rollout in zip_equal(indices, rollouts):
        print(f'\nGetting rollout IAs for simulation {i}', flush=True)
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            rollout_IAs[i] = get_IA_for_sim(rollout, args, last_n = 500)
        rollout_IA_errors[i] = [np.abs(IA_hat-IA) / np.abs(IA) for IA_hat, IA in 
                                        zip_equal(rollout_IAs[i], baseline_IAs[i][1:])]
        print(f'Relative IA error of rollout at final timestep of simulation {i} was {rollout_IA_errors[i][-1]}')
    mean_error = np.mean([x for x in rollout_IA_errors.values()])
    mean_error_final = np.mean([x[-1] for x in rollout_IA_errors.values()])
    print(f'Mean rollout IA error was {mean_error}')
    print(f'Mean rollout IA error at final timestep was {mean_error_final}')
    return (rollout_IAs, rollout_IA_errors, mean_error_final)

def sumLineSegments(p):
    d = p[1:] - p[:-1]
    d = np.sum(np.sqrt(np.sum(d**2,axis=1)))
    return d

def computeInterfacialArea(grid_x,grid_y,Z):
    levels = np.array([0.5]) # assume alpha = 0.5
    _, ax = plt.subplots(figsize=(10,10))
    CS = ax.contour(grid_x,grid_y,Z, levels, origin='lower')
    IA = np.sum(list(map(lambda x: sumLineSegments(x), CS.allsegs[0])))
    plt.close()
    return IA

def get_IA_for_sim(sim, args, last_n = 500, grid_nodes = None):
    IA = []
    if not grid_nodes:
        gridsFile = os.path.join(get_grid_folder(args), 'grid_x_grid_y.pkl')
    else:
        gridsFile = grid_nodes
    grids = pkl_load(gridsFile)
    grid_x = grids['grid_x']
    grid_y = grids['grid_y']
    for Z in sim[-last_n:]:
        ia = computeInterfacialArea(grid_x,grid_y,Z.squeeze())
        IA.append(ia)
    return IA

def get_IA_for_all_sims(args=None):
    
    IA_dict = {} 
    processed_folder = get_grid_folder(args)
    Path(processed_folder).mkdir(parents=True, exist_ok=True)
    processed_file_names = ["{:03d}.pkl".format(x) for x in range(1,get_sims(args)+1)]
    for i, fn in enumerate(processed_file_names):
        processed_grid_path = os.path.join(processed_folder, fn)
        try:
            file_size = os.path.getsize(processed_grid_path)
        except os.error:
            file_size = 0
        assert file_size > 1000, 'you need to build the grid before computing IA!'
    for i, fn in enumerate(processed_file_names, start=1):
        processed_grid_path = os.path.join(processed_folder, fn)
        sim = pkl_load(processed_grid_path)
        assert sim.shape[0]==get_simLength(args)
        assert sim.shape[1]==args.meta_gridSize
        assert sim.shape[2]==get_Ny(args)
        IA_dict[i] = get_IA_for_sim(sim, args, last_n = 500)
    return IA_dict
