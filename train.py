import warnings
warnings.filterwarnings('ignore')

import os
import sys
import copy
import time
import glob
import numpy as np
import pandas as pd
from typing import Any, Optional, Sequence, Tuple, Union, Dict

import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax

import torch
from torch_geometric.utils import add_self_loops, subgraph
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.transforms import LargestConnectedComponents as LCC

from config import parser, configure

# wait to import jax.jit functions. prevents jax preallocation while using torch to compute pe.
from nn.models.roma import ROMA, loss_train, loss_report, make_step
from lib import utils
from lib.graph_utils import get_next_batch, sup_power_of_two, pad_graph, threshold_subgraphs_by_size
from lib.positional_encoding import pe_path_from, pos_enc

#jax.config.update("jax_enable_x64", True)
prng = lambda i=0: jax.random.PRNGKey(i)
dim2str = lambda d: '[' + str(d[0])+', ' +str(len(d)-2)+'*'+str(d[1:2])+', ' + str(d[-1]) + ']'
read_file = lambda path: np.load(path) if path[-3:]=='npy' else pd.read_parquet(path).to_numpy()
read_file_T = lambda path: np.load(path).T if path[-3:]=='npy' else pd.read_parquet(path).to_numpy().T

if __name__ == '__main__': 
    stamp = str(int(time.time()))
    args = parser.parse_args()
    args = configure(args)    
    
    args.data_path = glob.glob(f'../data/x*{args.path}*')[0]
    args.adj_path = glob.glob(f'../data/edges*{args.path.split("_")[0]}*')[0]
    args.pe_path = pe_path_from(args)

    print(f'\n time_stamp = {stamp}')
    print(f'\n data_path: {args.data_path}\n adj_path: {args.adj_path}\n pe_path: {args.pe_path}\n')
    
    #torch.set_default_device('cpu')
    torch_device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    pe = pos_enc(args, le_size=args.le_size, rw_size=args.rw_size, n2v_size=args.n2v_size, norm=args.pe_norm, use_cached=args.use_cached_pe, device=torch_device) 
    torch.set_default_device('cpu')
    pe = torch.tensor(pe, dtype=torch.float32)

    print(' Reading graph from data_adj...', end='')    
    #A = pd.read_parquet(args.adj_path).T.to_numpy()
    A = read_file_T(args.adj_path)
    adj = A if A.shape[0]==2 else np.where(A)
    edge_index = torch.tensor(adj,dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index)
    time.sleep(0.2)
    print(' Done.\n')

    print(' Reading timeseries from data_path...', end='')
    #x = pd.read_parquet(args.data_path).to_numpy()
    x = read_file(args.data_path)
    time.sleep(0.2)
    print(' Done.\n')
    
    n,T = x.shape
    
    # split training and test sets via torch geometric data loaders:
    # i.e. dataloader hold torch.tensors on cpu. pad_graph converts batches to numpy.array on cpu -> jax.Arrays on gpu during training
    torch.manual_seed(args.torch_seed)
    
    # test set
    print(' Initializing loader and sampling test batch...',end='')
    idx=torch.arange(x.shape[0])#.reshape(-1,1) 
    data = Data(edge_index=edge_index, idx=idx)
    loader = GraphSAINTRandomWalkSampler(data, batch_size=args.sampler_batch_size, walk_length=args.batch_walk_len)
    batch_test, _ = get_next_batch(loader, args, data)
    x_test, adj_test, pe_test = pad_graph(x=x[batch_test.idx.numpy()], adj=batch_test.edge_index.numpy(), pe=pe[batch_test.idx.numpy()], x_size=args.batch_size)
    idx_test = batch_test.idx
    edge_index_test = batch_test.edge_index 
    time.sleep(0.2)
    print(' Done.\n')

    #print(f'batch_walk_len = {args.batch_walk_len}, sampler_batch_size = {args.sampler_batch_size}')    
    # training graph    
    print(' Initializing loader and sampling first training batch...',end='')
    mask_train = torch.ones(n, dtype=torch.int)
    mask_train[idx_test] = 0
    idx_train = torch.where(mask_train)[0]    
    edge_index_train, _ = subgraph(idx_train, edge_index, relabel_nodes=False)

    #idx_train = threshold_subgraphs_by_size(edge_index_train, min_size = args.min_subgraph_size)    
    #edge_index_train, _ = subgraph(idx_train, edge_index, relabel_nodes=False)
    # initialize batch loader from training graph
    data_train = Data(edge_index=edge_index_train, idx=idx)
    loader = GraphSAINTRandomWalkSampler(data_train, batch_size=args.sampler_batch_size, walk_length=args.batch_walk_len)

    batch, loader = get_next_batch(loader, args, data_train)
    
    x_batch, adj_batch, pe_batch = pad_graph(x=x[batch.idx.numpy()], adj=batch.edge_index.numpy(), pe=pe[batch.idx.numpy()], x_size=args.batch_size)
    time.sleep(0.2)
    print(' Done.\n')

    if args.log_path:
        model, args = utils.read_model(args)
    else:
        model = ROMA(args)
        model = utils.init_ortho(model, prng(123))

    if args.verbose: 
        print(f'\n MODULE: MODEL[DIMS](curv)')
        print(f'  encoder: {args.encoder}{args.enc_dims}{args.manifold[:3]}(c={args.c})')
        print(f'  pool:')
        for i in model.pool.pools.keys(): 
            pdims = args.pool_dims
            pdims[-1] = model.pool_dims[i] #sup_power_of_two(n // args.batch_red) // (args.pool_red)**(i+1)
            print(f'   pool_{i}: {args.pool}{pdims}')
        print(f'  embed:')
        for i in model.pool.pools.keys(): 
            print(f'   embed_{i}: {args.pool}{args.embed_dims}')
        if args.decoder == 'Operator':
            bdims = dim2str(model.decoder.branch_dims)
            print(f'  decoder: {args.decoder}{bdims} -> {args.dec_dims[-1:]}')
            print(f'  pde: {args.pde}/{args.decoder}{bdims} -> {args.pde_dims[-1:]}')
            print(f'    nonlinear[dec,pde]: {bool(args.nonlinear)},{bool(args.nonlinear_pde)}')
            print(f'    func_space: {args.func_space}(l={args.length_scale})')
            print(f'    branch/trunk nets: {model.decoder.branch.__class__.__name__}/{model.decoder.trunk.__class__.__name__}')
            print(f'    dual pe: {model.decoder.func_pe.__class__.__name__}') 
            print(f'    pos_emb_var = {args.pos_emb_var}, level_emb_var = {args.level_emb_var}')
            print(f'    time_enc: fourier[{args.coord_dim}][t_var={args.t_var},x_var={args.x_var}]\n')
        else:        
            print(f'  decoder: {args.decoder}{args.dec_dims}')
            print(f'  pde: {args.pde}/{args.decoder}{args.pde_dims}')

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))
    if args.w_pde*args.w_gpde < 1e-8: 
        param_count -= sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model.pde, eqx.is_inexact_array))) 
    pc = utils.round_to_nearest_thousands(param_count)
    print(f'  NUM_PARAMS = {pc}\n\n')
     
    log = {}
    log['args'] = vars(copy.copy(args))
    log['train_index'] = idx_train
    log['test_index'] = idx_test
    log['param_count'] = param_count

    if args.verbose:
        print(f' x[train] = {x[idx_train].shape}, edge_index[train] = {edge_index_train.shape}')
        print(f' x[test]  = {x[idx_test].shape},  edge_index[test]  = {edge_index_test.shape}\n')

        
        print(f' optim(lr,b1,b2,wd) = {args.optim}(lr={args.lr}, b1={args.b1}, b2={args.b2}, wd={args.weight_decay})')
        print(f' w[data,pde,gpde,ms] = ({args.w_data:.0e}, {args.w_pde:.0e}, {args.w_gpde:.0e}, {args.w_ms:.0e}*{args.w_pool})')
        print(f' dropout[enc,trunk,branch] = ({args.dropout}, {args.dropout_trunk}, {args.dropout_branch})\n')

    @jax.jit
    def _batch(x, pe, idx, sigma=args.eta_var, key=prng()):
        
        taus = jnp.arange(1, 1+args.tau_max, 1)
        bundles = idx + taus
        y = x[:,bundles].T
        y = jnp.swapaxes(y,0,1)
 
        win = jnp.arange(1 - args.kappa, 1, 1)
        x = x.at[:, idx + win].get()
        x = jnp.swapaxes(x,0,1)
        
        # add noise
        leta = jnp.sqrt(sigma) * jr.normal(key, x.shape)
        eta = jnp.exp(leta)
        x = eta * x
        
        leta = jnp.sqrt(sigma) * jr.normal(key, y.shape)
        eta = jnp.exp(leta)
        y = eta * y

        # add positional encoding
        pe = jnp.tile(pe, (idx.shape[0],1,1))
        x = jnp.concatenate([x, pe], axis=-1)

        return x, y
     
    #@jax.jit 
    def update(key, model, x, pe, adj, state, opt_state):
        key = jax.random.split(key)[0]
        ti = jax.random.randint(key, (args.num_col, 1), args.kappa, T-args.kappa).astype(jnp.float32)
        idx = ti.astype(int)
        xi,yi = _batch(x, pe, idx, key=key)
        
        (loss, state), grad = loss_train(model, xi, adj, ti, yi, key=key, mode='train', state=state)
        grad = jax.tree_map(lambda x: 0. if jnp.isnan(x).any() else x, grad) 
        if args.max_norm != args.max_norm_enc:
            grad = eqx.tree_at(lambda x: x.encoder, grad, utils.clip_tree(grad.encoder, args.max_norm_enc))
            grad = eqx.tree_at(lambda x: x.pool, grad, utils.clip_tree(grad.pool, args.max_norm_enc))        
        model, opt_state = make_step(grad, model, opt_state, optim)
        return model, opt_state, key

    #@jax.jit
    def loss_test(model, x, adj, pe):
        model = eqx.tree_inference(model, value=True) 
        ti = jnp.linspace(args.kappa, T - args.kappa , 20).reshape(-1,1)
        idx = ti.astype(int)
        xi,yi = _batch(x, pe, idx, sigma=0.)
        
        terms, _ = loss_report(model, xi, adj, ti, yi)
        loss = [term.mean() for term in terms]
        model = eqx.tree_inference(model, value=False) 
        return loss, model

    if args.steps == 0: sys.exit(0)

    lr = args.lr
    steps = args.steps
    num_cycles = args.num_cycles
    cycle_length = args.steps//num_cycles
    warmup_steps = min(10000, steps/2)
    decay_steps = steps - 2  * warmup_steps
    lr_min = 2e-7 #* (10000 / decay_steps)   
 
    schedule = optax.join_schedules(schedules=
    [
      optax.linear_schedule(0., lr, warmup_steps),
      optax.linear_schedule(lr, lr_min, decay_steps),
      optax.linear_schedule(lr_min, lr_min/10., warmup_steps),
    ] , boundaries=[warmup_steps, decay_steps+warmup_steps])

    params = {'learning_rate': schedule, 'weight_decay': args.weight_decay, 'b1': args.b1, 'b2': args.b2, 'eps': args.epsilon}
    optimizer = getattr(optax, args.optim)(**params)
    optim = optax.chain(optax.clip(args.max_norm), optimizer)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
 
    log['loss'] = {}
   
    tic = time.time()
    state = {'a': jnp.zeros(4), 'b': jnp.zeros(4)} if args.slaw else None
    key = jax.random.PRNGKey(0)
    model = eqx.tree_inference(model, value=False)
    for i in range(args.steps+1):
      
        if i % args.batch_freq == 0: 
            batch, loader = get_next_batch(loader, args, data_train)
       
        # if args.overfit_test: x_b, adj_b, pe_b = x_test, adj_test, pe_test
        x_b, adj_b, pe_b = pad_graph(x=x[batch.idx.numpy()], adj=batch.edge_index.numpy(), pe=pe[batch.idx.numpy()], x_size=args.batch_size)

        model, opt_state, key = update(key, model, x_b, pe_b, adj_b, state, opt_state)
        
        if i % args.log_freq == 0:
            
            x_b, adj_b, pe_b = x_test, adj_test, pe_test
            loss, model = loss_test(model, x_b, adj_b, pe_b)
            log['loss'][i] = [loss[0].item(), loss[1].item(), loss[2].item(), loss[3].item(), loss[4].item(), loss[5].item(), loss[6].item()]
            
            if args.verbose:
                print(f'{i:04d}/{args.steps} : l_data = ({loss[0]:.2e}, {loss[1]:.2e}),'
                      f' l_pde = ({loss[2]:.2e}, {loss[3]:.2e}),' 
                      f' l_ms = ({loss[4]:.2e}, {loss[5]:.2e}, {loss[6]:.2e}); ' 
                      f' lr = {schedule(i).item():.2e} (time: {time.time()-tic:.1f} s)')
            tic = time.time()
            
        if i % (args.log_freq * 10) == 0:
            utils.save_model(model, log, stamp=stamp)

    log['wall_time'] = int(time.time()) - int(stamp) 
    utils.save_model(model, log, stamp=stamp)
