import argparse, sys, os, yaml, hashlib
from configs import parser as _parser


def hash_e2e_hyperparams(args):
    # get short, string representation of hyperparameter settings
    relevant_hyperparams = [arg_name+'=='+str(args.__dict__[arg_name]) for arg_name in sorted(args.__dict__)
                                                         if arg_affects_e2e_results(args, arg_name)]
    hash_str = hashlib.md5(';'.join(relevant_hyperparams).encode('utf-8')).hexdigest()
    return hash_str


def hash_lin_hyperparams(args):
    # get short, string representation of hyperparameter settings
    relevant_hyperparams = [arg_name+'=='+str(args.__dict__[arg_name]) for arg_name in sorted(args.__dict__)
                                                         if arg_affects_lin_results(args, arg_name)]
    hash_str = hashlib.md5(';'.join(relevant_hyperparams).encode('utf-8')).hexdigest()
    return hash_str


def hash_latent_vector_hyperparams(args):
    # get short, string representation of hyperparameter settings
    relevant_hyperparams = [arg_name+'=='+str(args.__dict__[arg_name]) for arg_name in sorted(args.__dict__)
                                                         if arg_affects_latent_vector_results(args, arg_name)]
    hash_str = hashlib.md5(';'.join(relevant_hyperparams).encode('utf-8')).hexdigest()
    return hash_str


def get_args_from_file(file):
    # load args from a file created by write_args_to_file
    with open(file, 'r') as f:
        d = f.readlines()
    parse_this = []
    for a in d:
        if a.split(':')[0].strip() == 'lin_hidden_layers':
            parse_this += ['--' + a.split(':')[0].strip(), a.split(':')[1].strip()[1:-1]]
        else:
            parse_this += ['--' + a.split(':')[0].strip(), a.split(':')[1].strip()]
    return get_parser().parse_args(parse_this)
    

def write_args_to_file(args, file):
    # write values of all args to a file
    with open(file, 'w') as f:
        for arg_name in sorted(args.__dict__):
            f.write(arg_name + ': ' + str(args.__dict__[arg_name]) + '\n')


def arg_affects_latent_vector_results(args, arg_name):
    '''
    check if arg will be used by hash_latent_vector_hyperparams to differentiate experiments
    '''
    # a new config will only affect results if the arguments it contains are different from another config's args,
    # so none of these should affect lv results
    if (arg_name in ['meta_gpuIDs', 'meta_workers', 'meta_config', 'meta_dataDir', 'meta_outputDir'] or
            arg_name[:3] == 'lin' or
            arg_name[:3] == 'e2e' or
            arg_name[:3] == 'run'): 
        return False

    # if a new arg is added after experiment X was run, we don't want experiment X's hash to change.
    # we ensure that here, by returning False for args that have a value of None, and by following these rules:
    # 1) ALWAYS USE "None" AS A DEFAULT FOR NEWLY ADDED ARGUMENTS (we also NEVER USE "None" AS A DEFAULT FOR THE ORIGINAL ARGS to ensure that hashes are based on all variables)
    # 2) user must integrate new args such that their default value (None) does not affect the operation of the program at all!

    # (note: "meta_config" is the only original arg with a default of None, but it has required=True, so it can't be None)
    if args.__dict__[arg_name] is None: # arg=None iff it's a new arg that isn't affecting results (see discussion above)
        return False 
    if arg_name == 'lv_ConfigIter': # replicates will be stored in their own subfolder of the hashed path
        return False # so we return False
    else:
        return True


def arg_affects_lin_results(args, arg_name):
    '''
    check if arg will be used by hash_LIN_hyperparams to differentiate experiments
    '''
    # a new config will only affect results if the arguments it contains are different from another config's args,
    # so none of these should affect LIN results
    if (arg_name in ['meta_gpuIDs', 'meta_workers', 'meta_config', 'meta_dataDir', 'meta_outputDir'] or
            arg_name[:3] == 'run' or
            arg_name[:3] == 'e2e'): 
        return False

    # see arg_affects_latent_vector_results for a discussion of this logic
    if args.__dict__[arg_name] is None:
        return False 
    if arg_name == 'lin_ConfigIter': # replicates will be stored in their own subfolder of the hashed path
        return False # so we return False
    else:
        return True


def arg_affects_e2e_results(args, arg_name):
    '''
    check if arg will be used by hash_e2e_hyperparams to differentiate experiments
    '''
    # a new config will only affect results if the arguments it contains are different from another config's args,
    # so none of these should affect e2e finetuning results
    if (arg_name in ['meta_gpuIDs', 'meta_workers', 'meta_config', 'meta_dataDir', 'meta_outputDir'] or
            arg_name[:3] == 'run'): 
        return False

    # see arg_affects_latent_vector_results for a discussion of this logic
    if args.__dict__[arg_name] is None:
        return False 
    if arg_name == 'e2e_ConfigIter': # replicates will be stored in their own subfolder of the hashed path
        return False # so we return False
    else:
        return True


def str2bool(v):
    # converts string args to bool type
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def strWithNone(v):
    # converts string args to None if string is 'None'
    if isinstance(v, str) and v!='None':
       return v
    elif v=='None':
        return None
    else:
        raise argparse.ArgumentTypeError('String value expected.')


def get_parser():
    # based on https://github.com/RAIVNLab/STR
    
    # creates a parser with hyperparameters set to particular defaults
    # DO NOT CHANGE DEFAULTS
    # doing so would cause config files that rely on the original defaults to run with different settings

    parser = argparse.ArgumentParser(description="PyTorch Experiments for Deep Fluids Ablation Study")
  
    # General Config
    parser.add_argument(
        "--meta_config", help="Config file to use (see configs dir)", default=None, required=True
    )
    parser.add_argument(
        "--meta_outputDir", help="directory for holding processed data, logs, etc.", 
                    default="/usr/workspace/bartolds/simsur/ablation_study/data/"
    )
    parser.add_argument(
        "--meta_data", default="PNNL"
    )
    parser.add_argument(
        "--meta_dataDir", help="path to raw dataset directory", default="/p/gscratchr/bartolds/PNNL_raw/"
    )
    parser.add_argument(
        "--meta_channel", default=2, type=int, metavar="N", help="data channel'",
    )
    parser.add_argument(
        "--meta_latent_dimension", default=1024, type=int
    )
    parser.add_argument(
        "--meta_testSplit", default=0.2, type=float
    )
    parser.add_argument(
        "--meta_workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 8)",
    )
    parser.add_argument(
        "--meta_gpuIDs", default='0,1,2,3', help="Which GPUs to use. Leave blank for none",
    )
    parser.add_argument(
        "--meta_simLength", default=None, type=strWithNone, metavar="N", 
                help="number of timesteps, e.g. 500",
    )
    parser.add_argument(
        "--meta_sims", default=None, type=strWithNone, metavar="N", 
                help="number of simulations, e.g. 50",
    )
    parser.add_argument(
        "--meta_STARCCM_IA", help="folder that holds STARCCM IA files; each file should be named 'NNN_interfaceareamonitor.csv'"
                                    " where NNN gives the simulation number; e.g. 010 for the tenth simulation.",
                             default=None, type=strWithNone
    )



    # grid creation 
    parser.add_argument(
        "--meta_gridSize", default=128, type=int, metavar="N", 
        help="128 will make 128*128 grid, 512 makes 512*512, etc."
            " use meta_Ny_over_Nx arg to make a rectangular grid, meta_gridSize becomes width",
    )
    parser.add_argument(
        "--meta_Ny_over_Nx", default=None, type=strWithNone, metavar="N", 
                help="desired height divided by width for grid",
    )



    # latent creation
    parser.add_argument(
        "--lv_ConfigIter", default=0, type=int, metavar="N", help="replicate number for latent model training",
    )
    parser.add_argument(
        "--lv_max_epochs", default=1000, type=int, metavar="N", help="max number of total epochs to run",
    )
    parser.add_argument(
        "--lv_Model", default="ConvDeconvFactor2", help="ConvDeconvFactor2, SVD, etc.",
    )
    parser.add_argument(
        "--lv_LR", default=0.0001, type=float, help="initial learning rate for LVM"
    )
    parser.add_argument(
        "--lv_batch_size", default=128, type=int, help="batch size for LVM"
    )
    parser.add_argument(
        "--lv_filters", default=128, type=int, help="number of channels in first layer of LVM"
    )
    parser.add_argument(
        "--lv_num_conv", default=4, type=int, help="LVM hyperparameter not currently used"
    )
    parser.add_argument(
        "--lv_repeat", default=3, type=int, metavar="N", help="number of conv layers in encoder - 1"
    )
    parser.add_argument(
        "--lv_patience", default=3, type=int, help="number of epochs of no improvement in loss before reducing LR"
    )
    parser.add_argument(
        "--lv_doJacobian", type=str2bool, default=False, help="adds to loss the L1 norm ||grad(vel) - grad(predicted vel)||"
    )
    parser.add_argument(
        "--lv_createStreamFcn", type=str2bool, default=False, help="make decoder approximate stream function, for incompressible flow predictions"
    )
    parser.add_argument(
        "--lv_skip_connection", type=str2bool, default=False, help="use skip connections in LVM"
    )
    parser.add_argument(
        "--lv_stack", type=str2bool, default=False, help="LVM hyperparameter not currently used"
    )
    parser.add_argument(
        "--lv_Act", default="nn.ELU()", help="activation to use in LVM, used as a key for act_dict in models.py"
    )
    parser.add_argument(
        "--lv_Loss", default="L2_relative_loss", help="loss function to use in LVM"
    )
    parser.add_argument(
        "--lv_use_sigmoid_output_layer", type=str2bool, default=True, help="force outputs to (0,1) with sigmoid layer"
    )
    parser.add_argument(
        "--lv_DEBUG", type=str2bool, default=False, help="only run with a subset of data"
    )
    parser.add_argument(
        "--lv_standardize", type=str2bool, default=False, help="standardizes the data for model training"
    )
    parser.add_argument(
        "--lv_update_SVD", type=strWithNone, default=None, help="allow LVM from SVD of training data to be tuned"
    )


    # LIN creation
    parser.add_argument(
        "--lin_ConfigIter", default=0, type=int, metavar="N", help="replicate number for LIN model training"
    )
    parser.add_argument(
        "--lin_max_epochs", default=1000, type=int, metavar="N", help="max number of total epochs to run",
    )
    parser.add_argument(
        "--lin_Model", default="ARC", help="ARC, MLP, etc.",
    )
    parser.add_argument(
        "--lin_LR", default=0.001, type=float, help="learning rate for LIN"
    )
    parser.add_argument(
        "--lin_window", default=499, type=int, help="number of steps to rollout for during training"
    )
    parser.add_argument(
        "--lin_memory", default=6, type=int, help="number of past timesteps to use as input to LIN"
    )
    parser.add_argument(
        "--lin_batch_size", default=1, type=int, help="LIN batch size"
    )
    parser.add_argument(
        "--lin_Loss", default="L2_relative_vector_loss", help="loss function for LIN"
    )
    parser.add_argument(
        "--lin_patience", default=3, type=int, help="number of epochs of no improvement in LIN loss before LR decrease"
    )
    parser.add_argument(
        "--lin_DEBUG", type=str2bool, default=False, help="run with reduced training set size"
    )
    parser.add_argument(
        "--lin_normalize", type=str2bool, default=True, help="normalizes the data for model training"
    )
    parser.add_argument(
        "--lin_Act", default="nn.ELU()", help="LIN model activation function, used as a key for act_dict in models.py"
    )
    parser.add_argument(
        "--lin_hidden_layers", default=[128, 128, 128], 
            type=lambda s: [int(item) for item in s.split(',')], help="number of hidden layers in LIN"
    )
    parser.add_argument(
        "--lin_end_to_end_training", type=str2bool, default=False, help="train LIN jointly with LVM"
    )
    parser.add_argument(
        "--lin_train_on_unnormalized", type=str2bool, default=False, help="train on unnormalized latent vectors"
    )



    # End2End finetuning of LIN
    parser.add_argument(
        "--e2e_finetune", type=str2bool, default=False, help="apply fine-tuning of LIN after it is trained"
    )
    parser.add_argument(
        "--e2e_ConfigIter", default=0, type=int, metavar="N", help="replicate number for e2e finetuning"
    )
    parser.add_argument(
        "--e2e_max_epochs", default=1000, type=int, metavar="N", help="max number of total epochs to run",
    )
    parser.add_argument(
        "--e2e_LR", default=1e-6, type=float, help="learning rate to use during fine tuning"
    )
    parser.add_argument(
        "--e2e_window", default=499, type=int, help="number of steps to roll out for during training"
    )
    parser.add_argument(
        "--e2e_patience", default=5, type=int, help="number of epochs of no improvement in loss before LR decrease"
    )

    parser.add_argument(
        "--e2e_batch_size", default=None, type=strWithNone, help="E2E model batch size"
    )


    # Run particular components of the code
    parser.add_argument(
        "--run_display_output_locations",  type=strWithNone, default=None, help="print the location/hashes of outputs associated with the given hyperparameters then exit"
    )
    parser.add_argument(
        "--run_ground_truth_IA",  type=strWithNone, default=None, help='compute and save the ground truth IAs'
    )
    parser.add_argument(
        "--run_LVM_IA_only", type=strWithNone, default=None, help="compute IA associated with LVM reconstructions"
    )
    parser.add_argument(
        "--run_from_hash", type=strWithNone, default=None, help="run experiments.py using arguments found in a specified (preivously started) experiment's output folder"
    )

    return parser 


def parse_arguments():
  
    parser = get_parser()    

    args = parser.parse_args()

    get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open('configs/'+args.meta_config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.meta_config}")
    args.__dict__.update(loaded_yaml)

    # confirm hash can be recreated from written args
    print(args.__dict__)
    hash_str = hash_lin_hyperparams(args) 
    write_args_to_file(args, hash_str + '.txt')
    test_args = get_args_from_file(hash_str + '.txt')
    assert hash_str == hash_lin_hyperparams(test_args), 'hash of args failed'
    os.remove(hash_str + '.txt')
