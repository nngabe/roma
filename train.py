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
if jax.devices()[0].platform=='METAL': jax.config.update('jax_platform_name','cpu')

import jax.numpy as jnp
import equinox as eqx
import optax

import torch
from torch_geometric.utils import add_self_loops, subgraph
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.transforms import LargestConnectedComponents as LCC

from nn.models.renonet import RenONet, loss_train, loss_report, make_step
from config import parser, set_dims

from lib import utils
from lib.graph_utils import get_next_batch, sup_power_of_two, pad_graph
from lib.positional_encoding import pe_path_from, pos_enc

#jax.config.update("jax_enable_x64", True)
prng = lambda i=0: jax.random.PRNGKey(i)


if __name__ == '__main__':
    args = parser.parse_args()
    args.data_path = glob.glob(f'../data/x*{args.path}*')[-1]
    args.adj_path = glob.glob(f'../data/edges*{args.path.split("_")[0]}*')[0]
    args.pe_path = pe_path_from(args.adj_path)

    print(f'\n data path: {args.data_path}\n adj path: {args.adj_path}\n')
    
    args = set_dims(args)    
    pe = pos_enc(args, le_size=args.le_size, rw_size=args.rw_size, n2v_size=args.n2v_size, norm=args.pe_norm, use_cached=args.use_cached_pe)
    
    A = pd.read_csv(args.adj_path, index_col=0).to_numpy()
    adj = A if A.shape[0]==2 else np.where(A)
    edge_index = torch.tensor(adj,dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index)
    x = pd.read_csv(args.data_path, index_col=0).dropna().T
    x = torch.tensor(x.to_numpy())
    x = (x - x.min())/(x.max() - x.min())
    pe = torch.tensor(pe)
    n,T = x.shape


    # split training and test sets via torch geometric data loaders:
    # i.e. datatypes are torch.tensor -> numpy.array (-> jax.Array during training)
    torch.manual_seed(0)
    
    # test set
    idx=torch.arange(x.shape[0]).reshape(-1,1) 
    data = Data(edge_index=edge_index, idx=idx, x=x, pe=pe)
    loader = GraphSAINTRandomWalkSampler(data, batch_size=x.shape[0]//args.batch_down_sample, walk_length=args.batch_walk_len)
    batch_test, _ = get_next_batch(loader, args, data)
    x_test, adj_test, pe_test = pad_graph(x=batch_test.x.numpy(), adj=batch_test.edge_index.numpy(), pe=batch_test.pe.numpy(), x_size=args.batch_size)
    idx_test = batch_test.idx
    edge_index_test = batch_test.edge_index 

    # training graph
    mask_train = torch.ones(n, dtype=torch.int)
    mask_train[idx_test] = 0
    idx_train = torch.where(mask_train)[0]    
    edge_index_train, _ = subgraph(idx_train, edge_index, relabel_nodes=True)
    x_train, pe_train = x[idx_train], pe[idx_train]
    idx=torch.arange(x_train.shape[0]).reshape(-1,1) 
    data_train = Data(edge_index=edge_index_train, idx=idx, x=x[idx_train], pe=pe[idx_train])
    
    # initialize batch loader from training graph
    loader = GraphSAINTRandomWalkSampler(data_train, batch_size=data_train.idx.shape[0]//args.batch_down_sample, walk_length=args.batch_walk_len)
    batch, loader = get_next_batch(loader, args, data_train)
    x_batch, adj_batch, pe_batch = pad_graph(x=batch.x.numpy(), adj=batch.edge_index.numpy(), pe=batch.pe.numpy(), x_size=args.batch_size)

    if args.log_path:
        model, args = utils.read_model(args)
    else:
        model = RenONet(args)
        model = utils.init_he(model, prng(123))

    if args.verbose: 
        print(f'\n MODULE: MODEL[DIMS](curv)')
        print(f'  encoder: {args.encoder}{args.enc_dims}{args.manifold[:3]}(c={args.c})')
        print(f'  decoder: {args.decoder}{args.dec_dims}')
        print(f'  pde: {args.pde}/{args.decoder}{args.pde_dims}')
        print(f'  func_space: {args.func_space}(l={args.length_scale})')
        print(f'  branch/trunk nets: {model.decoder.branch.__class__.__name__}/{model.decoder.trunk.__class__.__name__}')
        print(f'  pool:')
        for i in model.pool.pools.keys(): 
            pdims = args.pool_dims
            pdims[-1] = model.pool_dims[i] #sup_power_of_two(n // args.batch_red) // (args.pool_red)**(i+1)
            print(f'   pool_{i}: {args.pool}{pdims}')
        print(f'  embed:')
        for i in model.pool.pools.keys(): 
            print(f'   embed_{i}: {args.pool}{args.embed_dims}')
        print(f' time_enc: fourier[{args.time_dim}]\n')
     
    log = {}
    log['args'] = vars(copy.copy(args))
    log['train_index'] = idx_train
    log['test_index'] = idx_test

    if args.verbose:
        print(f'\n x[train] = {x[idx_train].shape}, adj[train] = {edge_index_train.shape}')
        print(f' x[test]  = {x[idx_test].shape},  adj[test]  = {edge_index_test.shape}')
        print(f'\n w[data,pde,gpde,ent] = ({args.w_data:.0e}, {args.w_pde:.0e}, {args.w_gpde:.0e}, {args.w_ent:.0e})')
        print(f' dropout[enc,trunk,branch] = ({args.dropout}, {args.dropout_trunk}, {args.dropout_branch})')
    schedule = optax.warmup_exponential_decay_schedule(init_value=0., peak_value=args.lr, warmup_steps=args.epochs//10,
                                                        transition_steps=args.epochs, decay_rate=5e-3, end_value=args.lr/1e+3)

    params = {'learning_rate': schedule, 'weight_decay': args.weight_decay, 'b1': args.b1, 'b2': args.b2}
    optimizer = getattr(optax, args.optim)(**params)
    optim = optax.chain(optax.clip(args.max_norm), optimizer)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @jax.jit
    def _batch(x, pe, idx):
        win = jnp.arange(1 - args.kappa, 1, 1)
        x = x.at[:, idx + win].get()
        x = jnp.swapaxes(x,0,1)
        pe = jnp.tile(pe, (idx.shape[0],1,1))
        xb = jnp.concatenate([x, pe], axis=-1)
        return xb
   
    stamp = str(int(time.time()))
    log['loss'] = {}
   
    n = args.num_col # number of temporal colocation points
    tic = time.time()
    state = {'a': jnp.zeros(4), 'b': jnp.zeros(4)} if args.slaw else None
    key = jax.random.PRNGKey(0)
    model = eqx.tree_inference(model, value=False)
    for i in range(args.epochs):
      
        if i % args.batch_freq == 0: 
            batch, loader = get_next_batch(loader, args, data_train)
        
        x, adj, pe = pad_graph(batch.x.numpy(), batch.edge_index.numpy(), batch.pe.numpy(), x_size=args.batch_size)
        key = jax.random.split(key)[0]
        ti = jax.random.randint(key, (n, 1), args.kappa, T-args.kappa).astype(jnp.float32)
        idx = ti.astype(int)
        taus = jnp.arange(1, 1+args.tau_max, 1)
        bundles = idx + taus
        yi = x[:,bundles].T
        yi = jnp.swapaxes(yi,0,1)
        xi = _batch(x, pe, idx)
        (loss, state), grad = loss_train(model, xi, adj, ti, yi, key=key, mode='train', state=state)
        grad = jax.tree_map(lambda x: 0. if jnp.isnan(x).any() else x, grad) 
        
        model, opt_state = make_step(grad, model, opt_state, optim)
        if i % args.log_freq == 0:
            
            x, adj, pe = x_test, adj_test, pe_test
            model = eqx.tree_inference(model, value=True) 

            ti = jnp.linspace(args.kappa, T - args.kappa , 10).reshape(-1,1)
            idx = ti.astype(int)
            taus = jnp.arange(1, 1+args.tau_max, 1).astype(int)
            bundles = idx + taus
            yi = x[:,bundles].T
            yi = jnp.swapaxes(yi,0,1)
            xi = _batch(x, pe, idx)
            
            terms, _ = loss_report(model, xi, adj, ti, yi)
            loss = [term.mean() for term in terms]
            log['loss'][i] = [loss[0].item(), loss[1].item(), loss[2].item(), loss[3].item(), loss[4].item()]
            
            if args.verbose:
                print(f'{i:04d}/{args.epochs} : l_data = ({loss[0]:.2e}, {loss[1]:.2e}), l_pde = {loss[2]:.2e},' 
                      f' l_gpde = {loss[3]:.2e},  l_ent = {loss[4]:.2e};' 
                      f' lr = {schedule(i).item():.2e} (time: {time.time()-tic:.1f} s)')
            tic = time.time()
            
            
            model = eqx.tree_inference(model, value=False)
            
        if i % args.log_freq * 100 == 0:
            utils.save_model(model, log, stamp=stamp)

    log['wall_time'] = int(time.time()) - int(stamp) 
    utils.save_model(model, log, stamp=stamp)
