import os
import deeper_fluids.grid as grid
import deeper_fluids.latent_vectors as latent_vectors
import deeper_fluids.LIN as LIN
import deeper_fluids.IA as IA
import deeper_fluids.metric as metric
import deeper_fluids.End2End_Finetune as E2E
from args import (parse_arguments, hash_latent_vector_hyperparams, 
                    hash_lin_hyperparams, get_args_from_file)


def main():

    ARGS = parse_arguments()

    if ARGS.run_from_hash:
        config_folder = os.path.join(ARGS.meta_outputDir, ARGS.run_from_hash)
        iter_zero_folder = os.path.join(config_folder, str(0))
        path_to_hyperparams_of_this_config = os.path.join(iter_zero_folder, 'hyperparameters.log')
        ARGS = get_args_from_file(path_to_hyperparams_of_this_config)
        print(ARGS)
        print('\nSuccessfully loaded the above args from the given hparam hash')

    if ARGS.run_display_output_locations:
        print(ARGS.meta_outputDir + 'lv_' + hash_latent_vector_hyperparams(ARGS))
        print(ARGS.meta_outputDir + 'lin_' + hash_lin_hyperparams(ARGS))
        exit()

    if ARGS.run_ground_truth_IA:
        IA.create_from_args(ARGS)
        exit()

    #set the GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]= ARGS.meta_gpuIDs

    # build or select grid
    print('*************************\nBuilding grid (if not done yet)')
    grid_made = grid.create_from_args(ARGS) # build grid with specified args unless it already exists

    if grid_made:
        # learn or select latent vectors 
        print('*************************\nBuilding latents (if not done yet)')
        latents_made = latent_vectors.create_from_args(ARGS) # learns SVD/AE latent vectors if those with specified args don't already exist
        if ARGS.run_LVM_IA_only:
            IA.create_from_args(ARGS)
            return 1
    else:
        print('\nYou need to make the grid for this config!\n')
        return 0

    if latents_made:
        # learn or select LIN
        print('*************************\nBuilding LIN (if not done yet)')
        LIN_made = LIN.create_from_args(ARGS)
    else:
        print('\nYou need to make the latent vectors for this config!\n')
        return 0

    if LIN_made:
        # compute IA of LIN simulations
        print('*************************\nComputing IA (if not done yet)')
        IAs_made = IA.create_from_args(ARGS)
    else:
        print('\nYou need to make the LIN for this config!\n')
        return 0

    if IAs_made:
        # compare IAs to STAR-CCM IAs
        print('*************************\nComputing metrics (e.g. IA error w.r.t. STAR-CCM and speedup)')
        metric.create_from_args(ARGS)
    else:
        print('\nYou need to make the IAs for this config!\n')
        return 0

    if ARGS.e2e_finetune:
        print('*************************\nFinetuning LIN using e2e training (if not done yet)')
        e2e_LIN_finetuned = E2E.create_from_args(ARGS)

        if e2e_LIN_finetuned:
            # compute IA of LIN simulations
            print('*************************\nComputing IA (if not done yet)')
            IAs_made = IA.create_from_args(ARGS, e2e=True)
        else:
            print('\nYou need to finetune the LIN for this config!\n')
            return 0

        if IAs_made:
            # compare IAs to STAR-CCM IAs
            print('*************************\nGetting IA error w.r.t. STAR-CCM')
            metric.create_from_args(ARGS, e2e=True)
        else:
            print('\nYou need to make the IAs for this config!\n')
            return 0

    print('\nAll experiments completed successfully.\nExiting...')

if __name__ == '__main__':
    main()
