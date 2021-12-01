import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.init import xavier_uniform_
from torch.utils.data import Dataset, DataLoader
from .utils import (CCSI_Latent, get_SVD_vectors, rmse, get_PNNL_train_test_data,
                    L1_loss, L2_relative_loss, L2_relative_vector_loss)
from args import hash_latent_vector_hyperparams
from .utils import get_Ny
import os


act_dict = {'nn.ELU()': nn.ELU(),
            'nn.LeakyReLU()': nn.LeakyReLU()}
loss_dict = {'L1_loss':L1_loss, 
            'L2_relative_loss':L2_relative_loss,
            'L2_relative_vector_loss': L2_relative_vector_loss,
            'rmse':rmse}


# duplicating this function from latent_vectors.py, except assuming certain paths exist
# or will be created elsewhere
def lv_get_folder_paths(args):

    lv_hyperparams = hash_latent_vector_hyperparams(args)
    config_folder = os.path.join(args.meta_outputDir, 'lv_' + lv_hyperparams)
    iter_zero_folder = os.path.join(config_folder, str(0))
    #Path(iter_zero_folder).mkdir(parents=True, exist_ok=True)
    path_to_hyperparams_of_this_config = os.path.join(iter_zero_folder, 'hyperparameters.log')
    iter_folder = os.path.join(config_folder, str(args.lv_ConfigIter))
    #Path(iter_folder).mkdir(parents=True, exist_ok=True)

    return config_folder, path_to_hyperparams_of_this_config, iter_folder


#utility function for making models
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        s = [x.shape[0], *self.shape]
        return x.view(s)

#utility function for making models
def get_latent_vectors(args, override_args=None, return_mean_std=False):
    if override_args is None:
        override_args = {}

    if args.lin_DEBUG:
        numSamplesToKeep = 2
    else:
        numSamplesToKeep = np.infty #if not debugging

    # latent representations of the PNNL train and test data
    _, _, iter_folder = lv_get_folder_paths(args)
    vecs_path = os.path.join(iter_folder, "latent_vectors.pth")
    if override_args.setdefault('latent_vec_location',None):
        vecs_path = override_args['latent_vec_location']
    latents = torch.load(vecs_path)
    if return_mean_std:
        return latents['mean'], latents['std']
    
    # build dataset from these latents
    trainDataset = CCSI_Latent(latents['train'], numToKeep=numSamplesToKeep, args=args, override_args=override_args)
    testDataset = CCSI_Latent(latents['test'], numToKeep=numSamplesToKeep, args=args, train=False, override_args=override_args, 
                                abs_x_mx = trainDataset.abs_x_mx, abs_p_mx = trainDataset.abs_p_mx)

    return trainDataset, testDataset, trainDataset.preprocess_x, trainDataset.invPreprocess_x, trainDataset.preprocess_p


class ConvDeconvFactor2(nn.Module):
    def __init__(self, x_shape, filters=32, latentDim=16, num_conv=2, repeat=1,
                 skip_connection=False, stack=False, conv_k=3, last_k=3,
                 act=nn.LeakyReLU(), return_z=True, stream=False, device='cpu',
                 use_sigmoid_output_layer=False, norm = nn.BatchNorm2d):
        super(ConvDeconvFactor2,self).__init__()

        self.filters = filters
        self.latentDim = latentDim
        self.num_conv = num_conv
        self.repeat = repeat
        self.skip_connection = skip_connection
        self.stack = stack
        self.act = act
        self.conv_k = 3
        self.last_k = 3
        self.device = device
        self.return_z = return_z
        self.stream = stream
        self.use_sigmoid_output_layer = use_sigmoid_output_layer
        
        convLayers = [norm(x_shape[1]),nn.Conv2d(x_shape[1],filters, kernel_size=conv_k,stride=2,padding=1)]
        ch = filters
        for _ in range(0,repeat):
            convLayers.append(self.act)
            convLayers.append(norm(ch))
            convLayers.append(nn.Conv2d(ch, 2*ch, kernel_size=conv_k,stride=2,padding=1))
            ch = 2*ch

        self.ch = ch
        self.sz = x_shape[2]
        self.sz_y = x_shape[3]
        out_pad = []
        out_pad_y = []
        for i in range(repeat+1):
            out_pad.append(self.sz%2 ^ 1)
            out_pad_y.append(self.sz_y%2 ^ 1)
            self.sz = self.sz // 2 + self.sz%2
            self.sz_y = self.sz_y // 2 + self.sz_y%2
        out_pad.reverse()
        out_pad_y.reverse()
        convLayers.append(Reshape(-1))
        convLayers.append(nn.Linear(self.ch*self.sz*self.sz_y,self.latentDim))
        self.encoder = nn.Sequential(*convLayers).to(device)

        if stream:
            self.output_shape = torch.tensor([x_shape[1]-1, x_shape[2], x_shape[3]])
        else:
            self.output_shape = torch.tensor([x_shape[1], x_shape[2], x_shape[3]])

        #print(self.output_shape)
        deconvLayers = [nn.Linear(self.latentDim,self.ch*self.sz*self.sz_y),
                        Reshape(self.ch,self.sz,self.sz_y)]
        for _ in range(0,repeat):
            deconvLayers.append(norm(ch))
            deconvLayers.append(nn.ConvTranspose2d(ch,ch//2,
                                                   kernel_size=conv_k,stride=2,
                                                   padding=1, 
                                        output_padding=(out_pad[_], out_pad_y[_])))
            deconvLayers.append(self.act)
            ch = ch//2

        deconvLayers.append(norm(ch))
        deconvLayers.append(nn.ConvTranspose2d(ch,int(self.output_shape[0]),
                                               kernel_size=conv_k,stride=2,
                                               padding=1, 
                                        output_padding=(out_pad[-1], out_pad_y[-1])))

        if use_sigmoid_output_layer:
            deconvLayers.append(nn.Sigmoid())

        self.generator = nn.Sequential(*deconvLayers).to(device)
        # input data must be square

    def forward(self, x, p_x):
        z = self.encoder(x)
        z[:, -p_x.size(1):] = p_x
        x = self.generator(z)
        if self.return_z:
            return x, z
        else:
            return x

class SVD_Encoder(nn.Module):
    def __init__(self, U):
        super(SVD_Encoder,self).__init__()
        self.U = U

    def forward(self, frames):
        # u is from u,s,vh = svd(data)
        # frames = batch_size x channels x height x width
        assert len(frames.shape)==4
        x = frames.reshape(len(frames), -1)
        coeffs = x.matmul(self.U)
        # coeffs is now batch_size x numComp
        return coeffs    
    
class SVD_Decoder(nn.Module):
    def __init__(self, U, args=None):
        super(SVD_Decoder,self).__init__()
        self.U = U
        self.grid_size = args.meta_gridSize, get_Ny(args)
        self.n_channels = 1 if not hasattr(args.meta_channel,'__len__') else len(args.meta_channel)

    def forward(self, coeffs):
        # coeffs is now batch_size x numComp
        bs, nC = coeffs.shape
        R = self.U.matmul(coeffs.T)
        R = R.T.reshape(torch.Size((bs, self.n_channels, *self.grid_size)))
        return R
    
class SVD_Net(nn.Module):
    def __init__(self, args=None, return_z=True, allow_updates_to_U=False):
        super(SVD_Net,self).__init__()
        svd_vectors = get_SVD_vectors(args=args)
        self.return_z = return_z
        self.U = nn.Parameter(torch.tensor(svd_vectors[:, :args.meta_latent_dimension-2]),
                                                     requires_grad = allow_updates_to_U)
        self.encoder=SVD_Encoder(self.U)
        self.decoder=SVD_Decoder(self.U, args=args)
        self.args = args

    def forward(self, x, p_x):
        z = self.encoder(x)
        x = self.decoder(z)
        if self.return_z:
            z = torch.cat((z, p_x), dim=-1)
            assert z.shape == torch.Size((len(x), self.args.meta_latent_dimension)), z.shape
            return x, z
        else:
            return x


def get_lv_model(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.lv_Model == 'ConvDeconvFactor2':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_shape = torch.Size([args.lv_batch_size, 1 if not hasattr(args.meta_channel,'__len__') else len(args.meta_channel),
                                 args.meta_gridSize, get_Ny(args)])
        model = ConvDeconvFactor2(x_shape, args.lv_filters, args.meta_latent_dimension, args.lv_num_conv, args.lv_repeat,
                                args.lv_skip_connection, args.lv_stack, conv_k=3, last_k=3, 
                                act=act_dict[args.lv_Act], return_z=True, stream=args.lv_createStreamFcn, device=device, 
                                use_sigmoid_output_layer=args.lv_use_sigmoid_output_layer)

    if args.lv_Model == 'SVD':
        model = SVD_Net(args=args, return_z=True, allow_updates_to_U = args.lv_update_SVD is not None).to(device)

    return model


class MLP(nn.Module):
    def __init__(self, args=None):
        super(MLP,self).__init__()

        self.args = args
        self.activation = act_dict[args.lin_Act]
        hidden_layers = args.lin_hidden_layers.copy()
        hidden_layers.append(args.meta_latent_dimension)

        modules = []
        modules.append(nn.Linear(args.meta_latent_dimension * args.lin_memory, hidden_layers[0]))
        modules.append(self.activation)
        for idx,_ in enumerate(hidden_layers[:-1]):
            modules.append(nn.Linear(hidden_layers[idx],hidden_layers[idx+1]))
            if idx < len(hidden_layers) - 2:
                modules.append(self.activation)

        self.layers = nn.Sequential(*modules)

    def forward(self,x):
        x = x.reshape(len(x), -1)
        x = self.layers(x)
        return x


class ARC(nn.Module):
    '''
    Autoregressive Conv Net
    '''
    def __init__(self, args=None):
        super(ARC, self).__init__()
        
        self.memory = args.lin_memory
        self.activation = act_dict[args.lin_Act]
        hidden_layers = args.lin_hidden_layers.copy()
        hidden_layers.append(args.meta_latent_dimension)
        modules = []
        for idx, _ in enumerate(hidden_layers):
            if idx == 0:
                in_channels = args.meta_latent_dimension
            else:
                in_channels = hidden_layers[idx-1]
            # flip the channel_in-length order?
            modules.append(nn.Conv1d(in_channels, hidden_layers[idx], 3, padding=1)) 
            modules.append(self.activation)
        
        self.layers = nn.Sequential(*modules)

        self.output = nn.Conv1d(self.memory, 1, 1, padding=0)
        
        #self.output = nn.Linear(hiddenLayerSizes[-1]*memory, latentDim)
                                
        
    def forward(self, x):
        x = self.layers(x)
        if type(self.output) == nn.Conv1d:
            x = self.output(x.transpose(-1,-2)).squeeze(1)
        else:
            x = self.output(x.view(x.size(0), -1))
        return x
        
class LSTM(nn.Module):
    def __init__(self, args=None):
        super(LSTM, self).__init__()

        self.args = args
        hidden_layers = args.lin_hidden_layers.copy()
        hidden_layers.insert(0, args.meta_latent_dimension)

        LSTMS = []
        for idx,_ in enumerate(hidden_layers[:-1]):
            LSTMS.append(torch.nn.LSTM(input_size=hidden_layers[idx], hidden_size=hidden_layers[idx+1], batch_first=True))
        self.LSTMS = torch.nn.ModuleList(LSTMS)
        self.linear = torch.nn.Linear(hidden_layers[-1], args.meta_latent_dimension)

    def forward(self, x):
        for lstm in self.LSTMS:
            lstm.flatten_parameters()
            x = lstm(x)[0]
        x = self.linear(x)
        return x


# https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask
def generate_square_subsequent_mask(sz: int):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def get_pe_from_start(self, start, length):
        N = start.shape[0]
        # start is a tensor with different start for each batch element
        # stride an interva of length for each row (each batch item)
        x = torch.stack([torch.arange(N)] * length).T
        y = torch.stack([start + i for i in range(length)]).T
        return self.pe.expand(-1, N, -1)[y, x, :].transpose(0,
                                                            1)  # for some reason, need to transpose to keep [seq_length, N, dim]

    def forward(self, x, start = None):
        length = x.shape[0]
        x = x + self.get_pe_from_start(start, length)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, d_model: int = 1024, d_out: int = 1024, nhead: int = 8, num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.0, activation: str = "gelu",
                 # d_velo: int = 8,
                 lin_act: nn.Module = nn.ELU
                 ) -> None:
        super(Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model) 
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.out = nn.Linear(d_model, d_out) 
        #self.velocity_encoder = nn.Linear(1, d_velo)
        self._reset_parameters()
        self.d_model = d_model
        self.d_out = d_out
        self.nhead = nhead

    def forward(self, src, start #, velocity
                , src_mask = None,
                src_key_padding_mask = None):

        bs, seq, d_model = src.shape
        src = src.transpose(0, 1)  # transformer is not batch first
        src = self.pos_encoder(src, start=start)

        # Encode velocity and concat to src
        '''
        velo = self.velocity_encoder(velocity)  # bs, v_dim
        bs, v_dim = velo.shape
        velo = velo.unsqueeze(0).expand(src.size(0), -1, -1)
        src = torch.cat((src, velo), dim=-1)
        '''

        output = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = output.transpose(0, 1)  # transformer is not batch first
        output = self.out(output)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class EndToEndSurrogate(torch.nn.Module):
    
    def __init__(self, args, override_args=None):
        super(EndToEndSurrogate, self).__init__()
        if override_args is None:
            override_args = {}
        self.window = args.lin_window if 'lin_window' not in override_args else override_args['lin_window']
        self.memory_len = args.lin_memory if 'memory' not in override_args else override_args['memory']
        self.z_size = args.meta_latent_dimension - 2 # this does not include the size of p
        self.p_size = 2
        self.c_size = self.z_size + self.p_size # this does include the size of p, it's equal to args.meta_latent_dimension
        if args.lin_Model not in ['LSTM', 'Transformer']:
            model_dict = {'MLP': MLP, 'ARC': ARC}
            self.LIN = model_dict[args.lin_Model](args) 
            self.predict_next_w_encodings = self.base_predict_next_w_encodings
        elif args.lin_Model == 'LSTM':
            self.LIN = LSTM(args)
            self.predict_next_w_encodings = self.seq2seq_predict_next_w_encodings
        elif args.lin_Model == 'Transformer':
            self.LIN = Transformer(d_model=args.meta_latent_dimension, d_out=args.meta_latent_dimension, 
                                            nhead = 8, num_encoder_layers = 6, 
                                            dim_feedforward=args.meta_latent_dimension, lin_act=args.lin_Act) 
            self.predict_next_w_encodings = self.seq2seq_predict_next_w_encodings
        if 'e2e_LIN_initial_state' in override_args:
            best_LIN = torch.load(override_args['e2e_LIN_initial_state'])
            self.LIN.load_state_dict({(k if k[:7]!='module.' else k[7:]):v for k,v in best_LIN['LIN_state_dict'].items()})
            assert not args.lin_end_to_end_training, 'you are trying to use e2e finetuning, but e2e training was already used when training the LIN'
        if len(args.meta_gpuIDs.split(',')) > 1 and type(self.LIN)!='torch.nn.parallel.data_parallel.DataParallel':
            self.LIN = torch.nn.DataParallel(self.LIN)

        self.end_to_end = args.lin_end_to_end_training if 'end_to_end' not in override_args else override_args['end_to_end']
        self.loss_type = self.latent_vec_loss
        self.forward = self.forward_LIN
        self.args = args

        batch_size = args.lin_batch_size if 'batch_size' not in override_args else override_args['batch_size']
        shuffle = True if 'shuffle' not in override_args else override_args['shuffle']
        drop_last = True if 'drop_last' not in override_args else override_args['drop_last']

        if override_args.setdefault('build_without_datasets', False):
            (_, _, self.preprocess_x,
                self.invPreprocess_x,
                self.preprocess_p) = get_latent_vectors(args, override_args=override_args)
            self.encoder, self.decoder = self.get_lv_model_from_args(args,
                                               user_given_path=override_args['LVM_location'])
            self.forward = self.forward_end_to_end
            if len(args.meta_gpuIDs.split(',')) > 1:
                self.encoder, self.decoder = (torch.nn.DataParallel(self.encoder),
                                                 torch.nn.DataParallel(self.decoder))
            return
        if self.end_to_end:
            self.encoder, self.decoder = self.get_lv_model_from_args(args)
            self.loss_type = self.frame_loss
            self.forward = self.forward_end_to_end
            if len(args.meta_gpuIDs.split(',')) > 1:
                self.encoder, self.decoder = torch.nn.DataParallel(self.encoder), torch.nn.DataParallel(self.decoder) 
            self.trainDataset, self.testDataset = get_PNNL_train_test_data(args, window=self.window, memory=self.memory_len, return_start=True)
            _, _, self.preprocess_x, self.invPreprocess_x, self.preprocess_p = get_latent_vectors(args)
        else:
            self.trainDataset, self.testDataset, _, self.invPreprocess_x, _ = get_latent_vectors(args, override_args=override_args)
        self.trainDataLoader = DataLoader(dataset=self.trainDataset, batch_size=batch_size, pin_memory= True,
                                        shuffle=shuffle, drop_last=drop_last, num_workers=args.meta_workers)
        self.testDataLoader = DataLoader(dataset=self.testDataset, batch_size=batch_size, pin_memory= True, num_workers=args.meta_workers)

    def load_from_checkpoint(self, checkpoint):
        if type(checkpoint)==str:
            map_loc = None if torch.cuda.is_available() else torch.device('cpu')
            checkpoint = torch.load(checkpoint, map_location = map_loc)
        self.LIN.load_state_dict(checkpoint['LIN_state_dict'])
        if 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

    def save_checkpoint(self, checkpoint_path, epoch=None, opt=None, lr_scheduler=None, loss=None):
        d = {
            'last_epoch': epoch,
            'LIN_state_dict': self.LIN.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': loss
            }
        if self.end_to_end:
            assert ('e2e' in checkpoint_path or 
                    self.args.lin_end_to_end_training), 'you are trying to save the encoder and decoder without using end2end training/finetuning'
            lv = {
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict()
                }
            d.update(lv)
        torch.save(d, checkpoint_path)

    def get_lv_model_from_args(self, args=None, user_given_path=None):
        lv_model = get_lv_model(args)
        _, _, iter_folder = lv_get_folder_paths(args)
        model_path = os.path.join(iter_folder, "best_latent_vector_model.pth")
        map_loc = None if torch.cuda.is_available() else torch.device('cpu')
        if user_given_path:
            checkpoint = torch.load(user_given_path, map_location = map_loc)
        else:
            checkpoint = torch.load(model_path, map_location = map_loc)
        lv_model.load_state_dict({(k if k[:7]!='module.' else k[7:]):v for k,v in 
                                        checkpoint['model_state_dict'].items()})
        if hasattr(lv_model, 'generator'):
            assert not hasattr(lv_model, 'decoder')
            return lv_model.encoder, lv_model.generator
        else:
            assert hasattr(lv_model, 'decoder')
            return lv_model.encoder, lv_model.decoder
        
    def encode(self, U, p_x):
        if self.args.lv_Model != "SVD":
            c = self.encoder(U)
            c[:, -self.p_size:] = p_x
        else:
            #since the SVD encoding doesn't have the axes for the p-vector, concat here
            c = self.encoder(U)
            c = torch.cat((c, p_x), dim=-1)
        return c
        
    def decode(self, encoding):
        if self.args.lv_Model != "SVD":
            return self.decoder(encoding)
        else:
            return self.decoder(encoding[...,:-2])
        
    def seq2seq_predict_next_w_encodings(self, encoding, p_y, window, start):
        predicted_encodings = []

        seq = encoding.transpose(1,2)  # after transpose, seq.shape == [bs, seq_length, latent_dim]
        seq = seq.flip(1) # chronologically earliest sequence element is first now (at the 0 index)
        if self.args.lin_Model == 'Transformer':
            src_mask = generate_square_subsequent_mask(seq.size(1)).to(seq.device)  # seq_length
            n_heads = 8
            if len(self.args.meta_gpuIDs)>1:
                src_mask = src_mask.repeat(len(seq)*n_heads, 1, 1)
        # given a batch of encodings, advance each encoding window time steps.
        # save the result at each time step
        for j in range(window):
            # if you want to train on a window length smaller than simLength, you need to restrict seq to prevent attention between
            # test vectors that are more than window-length steps apart---i.e., to prevent attention spans not seen during training               
            memory_to_save = seq[:,1:].clone() # TODO: could try .detach().clone() here to get rid of recursive loop
            # use LIN to predict delta in encoding
            if self.args.lin_Model == 'Transformer':
                seq[:, -1] = self.LIN(src=seq.clone(), start=start, src_mask=src_mask)[:, -1] + seq[:, -1] 
            elif self.args.lin_Model == 'LSTM':
                seq[:, -1] = self.LIN(seq.clone())[:, -1] + seq[:, -1]
            else:
                assert False, 'only transformer and LSTM have seq2seq input-output-style'
            seq[:, -1, -self.p_size:] = p_y[:, j]
            # shift seq and add prediction to latest slot
            seq[:, :-1] = memory_to_save   
            start += 1             
            predicted_encodings.append(seq[:,-1].clone())

        #TODO: add option to return all predictions at all timesteps, not just last prediction from each timestep 
        # (this would require adjusting the loss function too)

        return torch.stack(predicted_encodings)

    def base_predict_next_w_encodings(self, encoding, p_y, window, start=None):
        '''
        use the LIN to predict the next w encodings for each encoded U in the batch.

        encoding: latent vectors with shape [batch_size, args.meta_latent_dimension, args.lin_memory]
        p_y: the ground-truth physics vars we apply to predictions, has shape [batch_size, window, 2]
        window: the number of future timesteps to predict

        this should be a method of the LIN, which will enable different
        prediction methods for transformers and ARC-nets
        '''
            
        predicted_encodings = []
            
        # given a batch of encodings, advance each encoding window time steps.
        # save the result at each time step
        for i in range(window):
            memory_to_save = encoding[:,:,0:-1].clone() # TODO: could try .detach().clone() here to get rid of recursive loop
            encoding[:,:,0] = self.LIN(encoding.clone()) + encoding[:,:,0] # use LIN to predict delta in encoding
            encoding[:,:,1:] = memory_to_save
            encoding[:,-self.p_size:,0] = p_y[:, i]
                    
            predicted_encodings.append(encoding[:,:,0].clone())
            
        return torch.stack(predicted_encodings)
    
    def forward_LIN(self, x, p_x=None, p_y=None, window = None, start=None):
        '''
        x.shape == [batch_size, c_size (args.meta_latent_dimension), memory_len]  (x is the latent vectors)
        p_x.shape == [batch_size, 2, memory_len]
        p_y.shape == [batch_size, window, 2]
        window: # of steps to rollout/simulate
        '''
        if window == None:
            window = self.window
        assert p_y.shape == torch.Size([len(p_y), window, 2]), f'{p_y.shape} is p_y; {window} is window'
        assert p_x.shape == torch.Size([len(p_x), 2, self.memory_len])
        assert x.shape == torch.Size([len(x), self.args.meta_latent_dimension, self.memory_len])
        assert (x[:,-self.p_size:] == p_x).all(), (x[:,-self.p_size:], p_x)

        encoding = x
        encoding_w = self.predict_next_w_encodings(encoding, p_y, window, start) # returns shape [window, batch_size, args.meta_latent_dimension]
        unnormalized_encoding_w = self.invPreprocess_x(encoding_w)
        # want to have this agree with targets, which are [batch_size, window_size, args.meta_latent_dimension]
        assert unnormalized_encoding_w.shape == torch.Size([window, len(p_y), self.c_size]), unnormalized_encoding_w.shape
        # so transpose dimensions 0 and 1
        return encoding_w.transpose(0,1), unnormalized_encoding_w.transpose(0,1)
    
    def forward_end_to_end(self, x, p_x=None, p_y=None, window = None, start=None):
        '''
        x.shape == [batch_size, memory, channels, nx, ny]  (x is the frames)
        p_x.shape == [batch_size, 2, memory_len]
        p_y.shape == [batch_size, window, 2]
        window: # of steps to rollout/simulate

        enables end-to-end training.
        '''
        U = x
        if window == None:
            window = self.window
        assert p_y.shape == torch.Size([len(p_y), window, 2]), p_y.shape
        assert p_x.shape == torch.Size([len(p_x), 2, self.memory_len]), p_x.shape
        assert U.shape == torch.Size([len(U), self.memory_len, 
                1 if not hasattr(self.args.meta_channel,'__len__') else len(self.args.meta_channel), 
                self.args.meta_gridSize, get_Ny(self.args)]), (f'{U.shape} is shape of U'
                    ' expected batch X memory/seq-length X n channels X Nx X Ny' )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # we normalized p if args.lin_normalize was True. 
        # but we haven't been normalizing p for the AE---we could add this later, TODO.
        # i.e., these p are unnormalized when we run encoding = self.encode(U[:,i], p_x[...,i])...
        # BUT when we self.preprocess_x(encoding) below, the encodings' p components 
        # won't match the unnormalized p... 
        # so we need to apply the preprocessing/normalization for p to the
        # p vectors after the encoding is done, so the p vectors and the p components of the encoding
        # match... matching is confirmed when the assertion in forward_LIN() passes.
        encoding_list = []
        for i in range(self.memory_len):
            if not (U[:,i] == 0).all(): # if this memory slot is filled
                encoding = self.encode(U[:,i], p_x[...,i])
                # normalize these encodings to enable prediction with the LIN
                normalized_encoding_part = self.preprocess_x(encoding)
            else:
                normalized_encoding_part = torch.zeros(U.shape[0], self.c_size, device=device)
            encoding_list.append(normalized_encoding_part[...,None])
        normalized_encodings = torch.cat(encoding_list, dim=-1)
        p_x = self.preprocess_p(p_x)
        p_y = self.preprocess_p(p_y)
        normalized_encoding_w, unnormalized_encoding_w = self.forward_LIN(normalized_encodings, p_x=p_x, p_y=p_y,
                                                                            window=window, start=start)
        U_hat = torch.stack([self.decode(encoding_i) for encoding_i in unnormalized_encoding_w])
        return unnormalized_encoding_w, U_hat

    def loss(self, batch, function=None):        
        return self.loss_type(batch, function=function)

    def frame_wise_L2_rel_error(self, batch, mean_std):
        return self.frame_loss(batch, function=L2_relative_loss, mean_std=mean_std)
    
    def latent_vec_loss(self, batch, function=None):
        if function is None:
            function = loss_dict[self.args.lin_Loss]

        x, y, p_x, p_y, start = [*batch]

        encodings_w, unnormalized_encodings_w = self.forward_LIN(x, p_x, p_y, start=start)
        target = y
        pred = encodings_w
        if self.args.lin_train_on_unnormalized:
            pred = unnormalized_encodings_w
            target = self.trainDataset.invPreprocess_x(target)

        return function(pred, target, args=self.args), rmse(pred, target), pred

    def frame_loss(self, batch, function=None, window=None, mean_std=(0,1)):
        if function is None:
            function = L2_relative_loss
        mean, std = mean_std 
        assert len(batch[0].shape) >= 4, f'reconstruction loss operates on frames, not elements with shape {batch[0].shape}'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        U_x, U_y, p_x, p_y, start = [v.to(device) for v in batch]
        _, U_y_hat = self.forward_end_to_end(U_x, p_x=p_x, p_y=p_y, window = window, start=start)
        U_y_hat = U_y_hat*std + mean
        U_y = U_y*std + mean
        return function(U_y_hat, U_y, args=self.args), rmse(U_y_hat, U_y), U_y_hat
