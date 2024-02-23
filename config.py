import argparse
import glob

from nn.utils.train_utils import add_flags_from_config
from lib.graph_utils import sup_power_of_two
config_args = {
    'training_config': {
        'lr': (5e-5, 'learning rate'),
        'dropout': (0.01, 'dropout probability'),
        'dropout_branch': (0.01, 'dropout probability in the branch net'),
        'dropout_trunk': (0.01, 'dropout probability in the trunk net'),
        'epochs': (100000, 'number of epochs to train for'),
        'optim': ('adamw', 'optax class name of optimizer'),
        'slaw': (False, 'whether to use scaled loss approximate weighting (SLAW)'),
        'b1': (.9, 'coefficient for first moment in adam'),
        'b2': (.999, 'coefficient for first moment in adam'),
        'weight_decay': (1e-3, 'l2 regularization strength'),
        'beta': (0.99, 'moving average coefficient for SLAW'),
        'log_freq': (50, 'how often to compute print train/val metrics (in epochs)'),
        'batch_freq': (10, 'how often to resample training graph'),
        'max_norm': (1.0, 'max norm for gradient clipping, or None for no gradient clipping'),
        'verbose': (True, 'print training data to console'),
        'opt_study': (False, 'whether to run a hyperparameter optimization study or not'),
        'num_col': (1, 'number of colocation points in the time domain'),
        'batch_size': (216, 'number of nodes in test and batch graphs'),
        'sampler_batch_size': (-1, 'factor to down sample training set.'),
        'batch_walk_len': (3, 'length of GraphSAINT sampler random walks.'),
        'lcc_train_set': (True, 'use LCC of graph after removing test set'),
        'batch_red': (2, 'factor of reduction for batch size'),
        'pool_red': (4, 'factor of reduction for each pooling step'),
    },
    'model_config': {

        # activation in all modules (enc,renorm,dec)
        'act': ('gelu', 'which activation function to use (or None for no activation)'),
        
        # init flags for neural nets
        'enc_init': (1, 'flag indicating whether the encoder remains to be init-ed or not.'),
        'dec_init': (1, 'flag indicating whether the decoder remains to be init-ed or not.'),
        'pde_init': (2, 'flag indicating number of pde functions which remain to be init-ed.'),
        'pool_init': (2, 'flag indicating number of pooling modules which remain to be init-ed.'),
        'embed_init': (2, 'flag indicating number of embedding modules which remain to be init-ed.'), 

        # loss weights
        'w_data': (1e+0, 'weight for data loss.'),
        'w_pde': (1e+0, 'weight for pde loss.'),
        'w_gpde': (1e+3, 'weight for gpde loss.'),
        'w_ent': (1e-1, 'weight for assignment matrix entropy loss.'),
        'F_max': (1., 'max value of convective term'),
        'v_max': (.0, 'max value of viscous term.'),
        'input_scaler': (1., 'rescaling of input'),
        'rep_scaler': (1., 'rescaling of graph features'),

        # which layers use time encodings and what dim should encodings be
        'pe_dim': (64, 'dimension of positional encoding'),
        'time_enc': ([0,1,1], 'whether to insert time encoding in encoder, decoder, and pde functions, respectively.'),
        'time_dim': (512, 'dimension of time embedding'), 
        'x_dim': (3, 'dimension of differentiable coordinates for PDE'),

        # positional encoding arguments
        'le_size': (0, 'size of laplacian eigenvector positional encoding'),
        'rw_size': (0, 'size of random walk (diffusion) positional encoding'),
        'n2v_size': (0, 'size of node2vec positional encoding'),
        'pe_norm': (True, 'apply norm (standard scaler) on each pe type'),
        'use_cached_pe': (True, 'whether to use previously computed embeddings or not'),

        # input/output sizes
        'fe': (0, 'encode features or not'),
        'kappa': (64, 'size of lookback window used as input to encoder'),
        'f_dim': (64, 'size of fourier feature encoding'), 
        'tau_max': (1, 'maximum steps ahead forecast'),
        
        # specify models. pde function layers are the same as the decoder layers by default.
        'encoder': ('HGCN', 'which encoder to use'),
        'decoder': ('DeepOnet', 'which decoder to use'),
        'pde': ('emergent', 'which pde to use for the pde loss'),
        'pool': ('HGCN', 'which model to compute coarsening matrices'),
        'func_space': ('GRF', 'function space for DeepOnet.'),
        'length_scale': (1., 'length scale for GRF'),
        'num_func': (32, 'number of functions to sample from func_space'),
        'num_spl': (1000, 'number of spline points for GRF'),
        'p_basis': (100, 'size of DeepOnet basis'),

        # dims of neural nets. -1 will be inferred based on args.skip and args.time_enc. 
        'enc_width': (64, 'dimensions of encoder layers'),
        'dec_width': (512,'dimensions of decoder layers'),
        'pde_width': (512, 'dimensions of each pde layers'),
        'pool_width': (512, 'dimensions of each pde layers'),
        'enc_depth': (3, 'dimensions of encoder layers'),
        'dec_depth': (5,'dimensions of decoder layers'),
        'pde_depth': (-1, 'dimensions of each pde layers'),
        'pool_depth': (4, 'dimensions of each pooling layer'),
        'enc_dims': ([-1,96,-1], 'dimensions of encoder layers'),
        'dec_dims': ([-1,256,256,-1],'dimensions of decoder layers'),
        'pde_dims': ([-1,256,256,1], 'dimensions of each pde layers'),
        'pool_dims': ([-1,96,-1], 'dimesions of pooling layers.'), 
        'embed_dims': ([-1,96,-1], 'dimensions of embedding layers.'),

        # DeepONet params
        'num_heads': (8, 'number of heads in transformer blocks.'),
        'trunk_res': (True, 'use residual connections in trunk net.'),
        'trunk_norm': (True, 'use layer norm in trunk net.'),
        'pos_emb_var': ([1/8, 1.], 'variance of transformer positional embedding at l=0 and l>0, respectively'),
        'level_emb_var': ([1.], 'variance of transformer level embedding'),
         
        # graph encoder params
        'res': (True, 'whether to use sum skip connections or not.'),
        'cat': (True, 'whether to concatenate all intermediate layers to final layer.'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': (.5, 'hyperbolic radius, set to None for trainable curvature'),
        'edge_conv': (True, 'use edge convolution or not'),
        'agg': ('sum', 'aggregation function to use'),
        'num_gat_heads': (6, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'use_att': (0, 'whether to use attention in next module to be inited.'),
        'use_att_enc': (0, 'whether to use attention in encoder layers or not'),
        'use_att_pool': (0, 'whether to use attention in graph pooling layers or not'),
        'use_layer_norm': (0, 'whether or not to use layernorm'),
        'use_bias': (1, 'whether to use a bias in linear layers'),
        'local_agg': (1, 'whether to local tangent space aggregation or not')
    },
    'data_config': {
        'path': ('t1707779950','from which to infer data path'),
        'log_path': (None, 'snippet from which to infer log/model path.'),
    }
}

def set_dims(args):
    # read cached pe if loading from path
    #if args.log_path != None: args.use_cached = True
    args.sampler_batch_size = args.batch_size//args.batch_walk_len + 10
        
    # size of renorm/pooling graphs
    args.pool_size = [args.batch_size//args.pool_red**i for i in range(1,args.pool_init+1)]
    args.num_nodes = args.batch_size + sum(args.pool_size)
    
    # pe dims
    args.le_size = args.pe_dim
    args.rw_size =0# args.pe_dim
    args.n2v_size = sup_power_of_two(2*args.pe_dim)
    
    # layer dims (enc,renorm,pde,dec)
    args.pde_depth = args.dec_depth
    args.enc_dims[0] = args.f_dim * 2 if args.fe else args.kappa
    args.enc_dims[0] += args.le_size + args.rw_size + args.n2v_size
    args.enc_dims[-1] = args.enc_width 
    args.dec_dims[-1] = args.x_dim
    args.enc_dims[1:-1] = (args.enc_depth-1) * [args.enc_width]
    args.dec_dims[1:-1] = (args.dec_depth-1) * [args.dec_width]
    args.pde_dims[1:-1] = (args.pde_depth-1) * [args.pde_width]
    args.pool_dims[1:-1] = (args.pool_depth-1) * [args.pool_width]
    args.embed_dims[1:-1] = (args.pool_depth-1) * [args.pool_width]
    if args.res: 
        enc_out = sum(args.enc_dims) + args.x_dim
        args.dec_dims[0] = enc_out + args.time_enc[1] * args.time_dim 
    else: 
        enc_out = args.enc_dims[-1] + args.x_dim
        args.dec_dims[0] = enc_out + args.time_enc[1] * args.time_dim 
    
    if args.pde=='emergent':
        args.pde_dims[0] = args.dec_dims[0] + 5 * args.x_dim
    else: 
        args.pde_dims[0] = args.dec_dims[0] 
        
    args.pool_dims[0] = enc_out - args.x_dim
    args.embed_dims[0] = enc_out - args.kappa - args.x_dim 
    args.pool_dims[-1] = args.batch_size
    args.embed_dims[-1] = args.embed_dims[0] 

    return args 

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
args = parser.parse_args()
args.manifold_pool = args.manifold
args = set_dims(args)
