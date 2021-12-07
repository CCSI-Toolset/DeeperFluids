import os, argparse, torch
import numpy as np
from args import get_args_from_file
from deeper_fluids.models import EndToEndSurrogate
from deeper_fluids.grid import loadfile, get_Ny
from deeper_fluids.IA import get_IA_for_sim
from deeper_fluids.utils import get_simLength
from timeit import default_timer as timer

class DF:
    """DF is a surrogate model inspired by Deep Fluids (Kim et al., 2019)

    Args:
        LVM: path to a latent vector model (pytorch files)
        LIN: path to a latent integration network (pytorch file)

    Notes:
        The LVM's folder should also hold the latent vectors of the data the LVM was trained on,
            'latent_vectors.pth'.
        The LIN's folder should also hold the hyperparameters associated with the LIN+LVM,
            'hyperparameters.log'.
        This is a high-level interface that enables a trained surrogate to predict on user-provided
            data. To take advantage of all functionality and train new models, use experiments.py
            instead.
    """
    
    def __init__(self,
                LVM = 'data/lv_c49da55ff41fe509b00b35c8b28e172d/0/best_latent_vector_model.pth',
                LIN = 'data/lin_a91f54e91cc7cefd872a2cab572f79f4/0/best_lin_model.pth', 
                verbose=True):

        if verbose:
            print('\n*********************************************\n'
                    '          Welcome to Deeper Fluids           '
                  '\n*********************************************\n')

        start_load = timer() # start loading model
        self.model = self.get_model_from_args(LIN, LVM)
        end_load = timer() # end loading model
        if verbose:
            print(f'Loaded surrogate in {round(end_load - start_load,3)} seconds')

    def get_model_from_args(self, LIN, LVM):
        # get the args the model was trained with, we require this file to be in the LIN's folder.
        # this requirement could be relaxed later (TODO).
        hyperparams_of_this_config = os.path.join('/'.join(LIN.split('/')[:-1]),
                                                         'hyperparameters.log')
        training_args = get_args_from_file(hyperparams_of_this_config)

        # set the GPU.
        # you must have available the same GPU IDs that training occurred with (meta_gpuIDs).
        # this requirement could be relaxed later (TODO).
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]= training_args.meta_gpuIDs

        # set the device---we require that a GPU/CUDA environment is available.
        # this requirement could be relaxed later (TODO).
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # need to specify location of original latent vectors for normalization; 
        # e.g., EndToEndSurrogate.preprocess_x depends on those vectors' statistics.
        # note that we require this file to be in the same folder as the LVM.
        # this requirement could be relaxed later (TODO).
        latent_vec_location = os.path.join('/'.join(LVM.split('/')[:-1]), 'latent_vectors.pth')
        # set override args to allow EndToEndSurrogate class to do simulations on new data.
        # build_without_datasets=True allows creation of an EndToEndSurrogate without a
        # dataset (by default, this class would hold the training/test data for the LIN).
        override_args = {'latent_vec_location': latent_vec_location,
                        'build_without_datasets': True,
                        'LVM_location': LVM}   
        model = EndToEndSurrogate(training_args, override_args) # build model
        model = model.to(self.device )
        model.load_from_checkpoint(LIN) # load LIN checkpoint (LVM checkpoint loaded at init)
        model.eval()
        return model

    def predict(self,
                sim_parameter_dict = None,
                init_data = 'data/raw_data/001/XYZ_Internal_Table_table_10.csv', 
                timesteps = 500,
                store_frames = True,
                verbose=True):
        """ predict() performs surrogate simulations given raw point-cloud data at t=1

            'sim_parameter_dict' looks like this:
                {'packing type': 'Raschig rings',  
                'column configuration': 'PNNL 2D RCM', 
                'solvent inlet velocity [m/s]': <user given>, 
                'solvent viscosity [Pa s]': 0.00969, 
                'solvent surface tension [N/m]': 0.02041}

            'init_data' should give the location of the CSV file that holds the first timestep
                (initial condition) for the simulation to be run. 

                this CSV file must have the following format:
                    "column1 column2 volume_fraction 'X (m)' 'Y (m)' "
                    i.e., 5 columns, with the final 3 columns giving volume fraction, x position, 
                    and y position.

            when store_frames=True, predict() saves the surrogate-simulated frames as a .pth file 
                inside the folder that 'init_data' is in. 
                
            predict() saves the uniform grid nodes inside the folder that 'init_data' is in.

            returns:
                sim_IA_at_T_dict = {'2D interfacial area [m/m^2]': <result of simulation>,
                                 '3D interfacial area [m^2/m^3]': None}
        """
        # print warning if model was trained on fewer timesteps than requested
        if timesteps > get_simLength(self.model.args):
            print(f'*Model was trained on {get_simLength(self.model.args)} timesteps.')
            print(f'Recommended action to ensure interpolation: use T <'
                    f' {get_simLength(self.model.args)}.')
            print(f'Your simulation is {timesteps} timesteps, which requires NN to extrapolate.\n')
        # print warning if requested inlet velocity is outside the range the model was trained on
        if (sim_parameter_dict['solvent inlet velocity [m/s]'] > 2.18e-2 or
            sim_parameter_dict['solvent inlet velocity [m/s]'] < 2e-3):
            # TODO: make these max/min vels a property of the model (like simLength above)
            # TODO for PNNL: provide paragraph summarizing model details (2D RCM, varying inlet
            # velocity, constant viscosity, etc.)
            print(f'*Model was trained on max vel of 2.18e-2 and min vel of 2e-3.')
            print(f'Recommended action to ensure interpolation: use 2e-3 <= v <= 2.18e-2')
            print(f'Your simulation has {sim_parameter_dict["solvent inlet velocity [m/s]"]}'
                    ' m/s velocity, which requires NN to extrapolate.\n')

        # set up output dictionary
        sim_IA_at_T_dict = {'2D interfacial area [m/m^2]': None,
                                 '3D interfacial area [m^2/m^3]': None}

        start_data = timer() # start data preprocessing        
        # grid data is batch (1 for now) X memory/seq-length X channel-count (1 for now) X Nx x Ny
        U_x = torch.zeros((1, self.model.args.lin_memory, 1, 
                            self.model.args.meta_gridSize, 
                            get_Ny(self.model.args)))
        # simulation data folder
        init_data_folder = '/'.join(init_data.split('/')[:-1])
        # set path for grid node location file, which is needed for IA computation
        grid_nodes = os.path.join(init_data_folder, 'grid_nodes.pkl')
        # preprocess CSV at timestep 1 to get volume fraction grid (U_x) at timestep 1
        U_x[0,0,0] = torch.tensor(loadfile((init_data, self.model.args), save_loc = grid_nodes))
        # use user's timesteps and velocities to create p_x and p_y
        t = np.linspace(0, timesteps/get_simLength(self.model.args), timesteps).astype('float32')
        v = sim_parameter_dict['solvent inlet velocity [m/s]']
        p = np.vstack([[v]*timesteps, t]).T
        p_x = torch.zeros((1, 2, self.model.args.lin_memory))
        p_x[0, :, 0] = torch.tensor(p[0])
        p_y = torch.tensor(p[1:]).unsqueeze(0)
        # set start equal to value associated with memory of trained model
        start = 1 - self.model.args.lin_memory
        end_data = timer() # end data preprocessing   

        if verbose:
            print(f'Data preprocessed in {round(end_data - start_data,3)} seconds')
            
        start_sim = timer() # start surrogate simulation
        # put data onto GPU
        U_x, p_x, p_y = U_x.to(self.device ), p_x.to(self.device ), p_y.to(self.device )
        with torch.no_grad():
            # simulate to get the predicted volume fraction field at each timestep.
            _, U_y_hat = self.model(U_x, p_x=p_x, p_y=p_y, window=timesteps-1, start=start)
        end_sim = timer() # end surrogate simulation
        
        if verbose:
            print(f'Surrogate simulated {timesteps-1} new timesteps '
                f'in {round(end_sim - start_sim,3)} seconds')
            
        start_IA = timer() # start IA computation
        sim_IA_at_T_dict['2D interfacial area [m/m^2]'] = get_IA_for_sim(
                U_y_hat.squeeze().cpu(), self.model.args, last_n=1, grid_nodes=grid_nodes)[0]
        end_IA = timer() # end IA computation
        
        if verbose:
            print(f'IA computed in {round(end_IA - start_IA,3)} seconds')

        
        if verbose:
            print(f'\nInterfacial area of surrogate simulation with velocity {v} m/s at '
                    f'timestep {timesteps} was:'
                    f'\n*********************************************\n'
                  f'         {sim_IA_at_T_dict["2D interfacial area [m/m^2]"]}'
                   ' m/m^2          '
                    f'\n*********************************************\n')   
            if store_frames:
                torch.save({f'{v}': U_y_hat.cpu()}, 
                        os.path.join(init_data_folder, 'surrogate_simulation.pth'))
                print(f'Simulated frames stored in {init_data_folder}')
            print('Exiting...\n')

        return sim_IA_at_T_dict

if __name__ == '__main__':

    # get user args
    parser = argparse.ArgumentParser(description="Apply Deeper Fluids to CFD data")
    parser.add_argument(
        "--LVM", help="LVM path", 
        default=None
    )
    parser.add_argument(
        "--LIN", help="LIN path", 
        default=None
    )
    parser.add_argument(
        "--init_data", help="path to file with raw (point cloud) volume fraction data for t=1", 
        default=None
    )
    parser.add_argument(
        "--solvent_inlet_velocity", help="solvent inlet velocity [m/s] of desired simulation", 
        default=None, type=float
    )
    parser.add_argument(
        "--T", help="number of timesteps of desired surrogate simulation", 
        default=500, type=int
    )
    ARGS = parser.parse_args()
  
    # build model and load trained weights
    surrogate = DF(LVM = ARGS.LVM, LIN = ARGS.LIN)

    # set up simulation parameters
    sim_parameter_dict = {'packing type': 'Raschig rings',  
                'column configuration': 'PNNL 2D RCM', 
                'solvent inlet velocity [m/s]': ARGS.solvent_inlet_velocity, 
                'solvent viscosity [Pa*s]': 0.00969, 
                'solvent surface tension [N/m]': 0.02041}

    # predict interfacial area at final timestep T for each initial condition
    sim_IA_at_T_dict = surrogate.predict(
                        sim_parameter_dict = sim_parameter_dict,
                        init_data = ARGS.init_data,
                        timesteps = ARGS.T)