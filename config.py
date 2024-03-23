import argparse
import glob

from nn.utils.train_utils import add_flags_from_config
from lib.graph_utils import sup_power_of_two
config_args = {
    'training_config': {
        'lr': (1e-5, 'learning rate'),
        'dropout': (0.01, 'dropout probability'),
        'dropout_branch': (0.01, 'dropout probability in the branch net'),
        'dropout_trunk': (0.01, 'dropout probability in the trunk net'),
        'epochs': (60000, 'number of epochs to train for'),
        'num_cycles': (1, 'number of warmup/cosine decay cycles'),
        'optim': ('adamw', 'optax class name of optimizer'),
        'slaw': (False, 'whether to use scaled loss approximate weighting (SLAW)'),
        'b1': (.9, 'coefficient for first moment in adam'),
        'b2': (.999, 'coefficient for second moment in adam'),
        'weight_decay': (2e-4, 'l2 regularization strength'),
        'epsilon': (1e-8, 'epsilon in adam denominator'),
        'beta': (.99, 'moving average coefficient for SLAW'),
        'log_freq': (100, 'how often to compute print train/val metrics (in epochs)'),
        'batch_freq': (1, 'how often to resample training graph'),
        'max_norm': (.05, 'max norm for gradient clipping, or None for no gradient clipping'),
        'max_norm_enc': (.05, 'max norm for graph network gradient clipping, or None for no gradient clipping'),
        'verbose': (True, 'print training data to console'),
        'opt_study': (False, 'whether to run a hyperparameter optimization study or not'),
        'num_col': (1, 'number of colocation points in the time domain'),
        'batch_size': (216, 'number of nodes in test and batch graphs'),
        'sampler_batch_size': (-1, 'factor to down sample training set.'),
        'batch_walk_len': (30, 'length of GraphSAINT sampler random walks.'),
        'min_subgraph_size': (100, 'minimum subgraph size for training graph sampler.'),
        'lcc_train_set': (True, 'use LCC of graph after removing test set'),
        'batch_red': (2, 'factor of reduction for batch size'),
        'pool_red': (4, 'factor of reduction for each pooling step'),
        'pool_steps': (2, 'number of pooling steps'),
        'eta_var': (1e-6, 'variance of multiplicative noise'),
    },
    'model_config': {

        # activation in all modules (enc,renorm,dec)
        'act': ('gelu', 'which activation function to use (or None for no activation)'),
        
        # loss weights
        'w_data': (1e+0, 'weight for data loss.'),
        'w_pde': (1e+0, 'weight for pde loss.'),
        'w_gpde': (1e+3, 'weight for gpde loss.'),
        'w_ms': (1e-4, 'weight for assignment matrix entropy loss.'),
        'w_pool': ([1e+0, 2e+0, 2e+0], 'weights for S entropy, A entropy, and LP respectively.'),
        'F_max': (1., 'max value of convective term'),
        'v_max': (.0, 'max value of viscous term.'),
        'input_scaler': (1., 'rescaling of input'),
        'rep_scaler': (1., 'rescaling of graph features'),

        # which layers use time encodings and what dim should encodings be
        'x_dim': (3, 'dimension of differentiable coordinates for PDE'),
        'coord_dim': (512, 'dimension of (t,x) embedding'), 
        't_var': (5e-2, 'variance of time embedding in trunk net'),
        'x_var': (1e-2, 'variance of space embedding in trunk net'),

        # positional encoding arguments
        'pe_dim': (128, 'dimension of positional encoding'),
        'le_size': (-1, 'size of laplacian eigenvector positional encoding'),
        'rw_size': (-1, 'size of random walk (diffusion) positional encoding'),
        'n2v_size': (-1, 'size of node2vec positional encoding'),
        'pe_size': (-1, 'size of composite pe'),
        'pe_norm': (True, 'apply norm (standard scaler) on each pe type'),
        'use_cached_pe': (True, 'whether to use previously computed embeddings or not'),

        # input/output sizes for dynamical data
        'kappa': (64, 'size of lookback window used as input to encoder'),
        'tau_max': (1, 'maximum steps ahead forecast'),
        
        # specify models. pde function layers are the same as the decoder layers by default.
        'encoder': ('HGCN', 'which encoder to use'),
        'decoder': ('DeepOnet', 'which decoder to use'),
        'pde': ('emergent', 'which pde to use for the pde loss'),
        'pool': ('HGCN', 'which model to compute coarsening matrices'),
        'func_space': ('GRF', 'function space for DeepOnet.'),
        'length_scale': (1., 'length scale for GRF'),
        'num_func': (128, 'number of functions to sample from func_space'),
        'num_spl': (100, 'number of spline points for GRF'),
        'p_basis': (64, 'size of DeepOnet basis'),

        # dims of neural nets. -1 will be inferred based on args.skip and args.time_enc. 
        'enc_width': (256, 'dimensions of encoder layers'),
        'dec_width': (640,'dimensions of decoder layers'),
        'pde_width': (640, 'dimensions of each pde layers'),
        'pool_width': (640, 'dimensions of each pde layers'),
        'enc_depth': (2, 'dimensions of encoder layers'),
        'dec_depth': (4,'dimensions of decoder layers'),
        'pde_depth': (-1, 'dimensions of each pde layers'),
        'pool_depth': (2, 'dimensions of each pooling layer'),
        'enc_dims': ([-1]*3, 'dimensions of encoder layers'),
        'dec_dims': ([-1]*3,'dimensions of decoder layers'),
        'pde_dims': ([-1,-1,1], 'dimensions of each pde layers'),
        'pool_dims': ([-1]*3, 'dimesions of pooling layers.'), 
        'embed_dims': ([-1]*3, 'dimensions of embedding layers.'),

        # DeepONet params
        'trunk_net': ('MLP', 'trunk network architecture.'),
        'branch_net': ('Transformer', 'branch net architecture.'),
        'num_heads': (8, 'number of heads in transformer blocks.'),
        'trunk_res': (True, 'use residual connections in trunk net.'),
        'trunk_norm': (True, 'use layer norm in trunk net.'),
        'pos_emb_var': (0, 'variance of transformer positional embedding at l=0 and l>0, respectively'),
        'level_emb_var': (0, 'variance of transformer level embedding'),
         
        # graph network params
        'res': (True, 'whether to use sum skip connections or not.'),
        'cat': (True, 'whether to concatenate all intermediate layers to final layer.'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': (1/2, 'hyperbolic radius, set to None for trainable curvature'),
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
        'path': ('t1708806913','from which to infer data path'),
        'log_path': (None, 'snippet from which to infer log/model path.'),
    }
}

def configure(args):

    if args.pos_emb_var == 1: 
        args.pos_emb_var = [1/2, 1.]
    elif args.pos_emb_var == 2: 
        args.pos_emb_var = [1/8, 1.]
    else: 
        args.pos_emb_var = [1/4, 1.]
    
    if args.level_emb_var == 1: 
        args.level_emb_var = [0.]
    else:
        args.level_emb_var = [1.]

    # read cached pe if loading from path
    #if args.log_path != None: args.use_cached = True
    args.sampler_batch_size = args.batch_size//args.batch_walk_len + 20
        
    # size of renorm/pooling graphs
    args.manifold_pool = args.manifold
    args.pool_size = [32//args.pool_red**i for i in range(0,args.pool_steps)]
    args.num_nodes = args.batch_size + sum(args.pool_size)
    
    # pe dims
    args.le_size = args.pe_dim
    args.rw_size = 0 # args.pe_dim
    args.n2v_size = args.pe_dim 
    args.pe_size = args.le_size + args.rw_size + args.n2v_size
    
    # layer dims (enc,renorm,pde,dec)
    args.pde_depth = args.dec_depth
    args.enc_dims[0] = args.kappa * 2
    #args.enc_dims[0] += args.le_size + args.rw_size + args.n2v_size
    args.enc_dims[-1] = args.enc_width 
    args.dec_dims[-1] = args.x_dim
    args.enc_dims[1:-1] = (args.enc_depth-1) * [args.enc_width]
    args.dec_dims[1:-1] = (args.dec_depth-1) * [args.dec_width]
    args.pde_dims[1:-1] = (args.pde_depth-1) * [args.pde_width]
    args.pool_dims[1:-1] = (args.pool_depth-1) * [args.pool_width]
    args.embed_dims[1:-1] = (args.pool_depth-1) * [args.pool_width]

    if args.res: 
        enc_out = sum(args.enc_dims) 
        args.dec_dims[0] = enc_out + (1 + args.x_dim) * args.coord_dim
    else: 
        enc_out = args.enc_dims[-1] 
        args.dec_dims[0] = enc_out + (1 + args.x_dim) * args.coord_dim
    
    if args.pde=='emergent':
        args.pde_dims[0] = args.dec_dims[0] + 5 * args.x_dim
    else: 
        args.pde_dims[0] = args.dec_dims[0] 
        
    args.pool_dims[0] = enc_out - args.kappa
    args.embed_dims[0] = enc_out - args.kappa 
    args.pool_dims[-1] = args.pool_size[0] * args.pool_red
    args.embed_dims[-1] = args.embed_dims[0] 

    return args 

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
args = parser.parse_args()
args = configure(args)
