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
import jax.numpy as jnp
import equinox as eqx
import optax

import torch
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.transforms import LargestConnectedComponents as LCC

from nn.models.renonet import RenONet, loss_scan, loss_terms, make_step
from config import parser, set_dims

from lib import utils
from lib.graph_utils import subgraph, get_next_batch, add_self_loops, sup_power_of_two
from lib.positional_encoding import pe_path_from, pos_enc

#jax.config.update("jax_enable_x64", True)
prng = lambda i=0: jax.random.PRNGKey(i)

if __name__ == '__main__':
    args = parser.parse_args()
    args.data_path = glob.glob(f'../data/x*{args.path}*')[0]
    args.adj_path = glob.glob(f'../data/adj*{args.path.split("_")[0]}*')[0]
    args.pe_path = pe_path_from(args.adj_path)

    print(f'\n w[data,pde,gpde,ent] = {args.w_data:.0e},{args.w_pde:.0e},{args.w_gpde:.0e},{args.w_ent:.0e}')
    print(f'\n data path: {args.data_path}\n adj path: {args.adj_path}\n\n')
    
    #pe = pe/pe.max()
    #args.pe_dim = pe.shape[1]
    args = set_dims(args)    
    pe = pos_enc(args, le_size=args.le_size, rw_size=args.rw_size, n2v_size=args.n2v_size, norm=args.pe_norm, use_cached=args.use_cached)
    
    A = pd.read_csv(args.adj_path, index_col=0).to_numpy()
    adj = A if A.shape[0]==2 else np.where(A)
    edge_index = torch.tensor(adj,dtype=torch.long)
    x = pd.read_csv(args.data_path, index_col=0).dropna()
    #for i in range(4): x = x.T.diff().rolling(20,center=True, win_type='gaussian').mean(std=40).dropna().cumsum().T 
    x = jnp.array(x.to_numpy())
    x = (x - x.min())/(x.max() - x.min())
    n,T = x.shape

    torch.manual_seed(0)
    data = Data(edge_index=edge_index, idx=torch.arange(x.shape[0]).reshape(-1,1))
    sampler = GraphSAINTRandomWalkSampler(data, batch_size=x.shape[0]//50, walk_length=3)
    batch_test = next(iter(sampler))
    idx_test = np.array(batch_test.idx).flatten()
    x_test, adj_test, pe_test = subgraph(index=idx_test, x=x, adj=adj, pe=pe)
    mask_train = np.ones(n, dtype=np.int32)
    mask_train[idx_test] = 0
    idx_train = np.where(mask_train)[0]
        
    x_train, adj_train, pe_train = subgraph(index=idx_train, x=x, adj=adj, pe=pe, pad=False)
    data_train = Data(edge_index=torch.tensor(adj_train.tolist()), idx=torch.arange(x_train.shape[0]).reshape(-1,1)) 
    #if args.lcc_train_set:
    #    lcc = LCC().forward(data_train)
    #    idx_lcc = lcc.idx.flatten().numpy()
    #    x_train, adj_train, pe_train = subgraph(index=idx_lcc, x=x_train, adj=adj_train, pe=pe_train, pad=False)
    data_load = Data(edge_index=torch.tensor(adj_train.tolist()), idx=torch.arange(x_train.shape[0]).reshape(-1,1))
    loader = GraphSAINTRandomWalkSampler(data_load, batch_size=data_load.idx.shape[0]//args.batch_down_sample, walk_length=args.batch_walk_len)
    batch, loader = get_next_batch(loader, args)
    idx_batch = batch.idx.flatten().numpy()
    x_batch, adj_batch, pe_batch = subgraph(index=idx_batch, x=x_train, adj=adj_train, pe=pe_train, pad=True)
    sys.exit(0)

    args.pool_dims[-1] = 128 #sup_power_of_two(2 * n//args.pool_red)
    if args.log_path:
        model, args = utils.read_model(args)
    else:
        model = RenONet(args)
        model = utils.init_he(model, prng(123))

    if args.verbose: 
        print(f'\nMODULE: MODEL[DIMS](curv)')
        print(f' encoder: {args.encoder}{args.enc_dims}({args.c})')
        print(f' decoder: {args.decoder}{args.dec_dims}')
        print(f' pde: {args.pde}/{args.decoder}{args.pde_dims}')
        print(' pool:')
        for i in model.pool.pools.keys(): 
            pdims = args.pool_dims
            pdims[-1] = model.pool_dims[i] #sup_power_of_two(n // args.batch_red) // (args.pool_red)**(i+1)
            print(f'   pool_{i}: {args.pool}{pdims}')
        print(' embed:')
        for i in model.pool.pools.keys(): 
            print(f'   embed_{i}: {args.pool}{args.embed_dims}')
        print(f' time_enc: linlog[{args.time_dim}]\n')
    
    log = {}
    log['args'] = vars(copy.copy(args))
    log['train_index'] = idx_train
    log['test_index'] = idx_test

    if args.verbose:
        print(f'\nx[train] = {x[idx_train].shape}, adj[train] = {adj_train.shape}')
        print(f'x[test]  = {x[idx_test].shape},  adj[test]  = {adj_test.shape}')
    
    schedule = optax.warmup_exponential_decay_schedule(init_value=0., peak_value=args.lr, warmup_steps=args.epochs//100,
                                                        transition_steps=args.epochs, decay_rate=5e-3, end_value=args.lr/1e+3)
    optim = optax.chain(optax.clip(args.max_norm), optax.adamw(learning_rate=schedule)) 
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
    #x, adj, pe, _   = random_subgraph(x_train, adj_train, pe_train, batch_size=args.batch_size, key=prng(0))
   
    print(f'\nSLAW: {args.slaw}, dropout: p={args.dropout}, num_col={args.num_col}, \n')
    
    n = args.num_col # number of temporal colocation points
    tic = time.time()
    state = {'a': jnp.zeros(4), 'b': jnp.zeros(4)} if args.slaw else None
    key = jax.random.PRNGKey(0)
    model = eqx.tree_inference(model, value=False)
    for i in range(args.epochs):
       
        x, adj, pe = x_batch, adj_batch, pe_batch
        key = jax.random.split(key)[0]
        ti = jax.random.randint(key, (n, 1), args.kappa, T-args.kappa).astype(jnp.float32)
        idx = ti.astype(int)
        taus = jnp.arange(1, 1+args.tau_max, 1)
        bundles = idx + taus
        yi = x[:,bundles].T
        yi = jnp.swapaxes(yi,0,1)
        xi = _batch(x, pe, idx)
        (loss, state), grad = loss_scan(model, xi, adj, ti, yi, key=key, mode='train', state=state)
        grad = jax.tree_map(lambda x: 0. if jnp.isnan(x).any() else x, grad) 
        
        model, opt_state = make_step(grad, model, opt_state, optim)
        if i % args.log_freq == 0:
            
            x, adj, pe = x_test, adj_test, pe_test
            model = eqx.tree_inference(model, value=True) 

            ti = jnp.linspace(args.kappa, T - args.tau_max , 10).reshape(-1,1)
            idx = ti.astype(int)
            taus = jnp.arange(1, 1+args.tau_max, 1).astype(int)
            bundles = idx + taus
            yi = x[:,bundles].T
            yi = jnp.swapaxes(yi,0,1)
            xi = _batch(x, pe, idx)
            
            terms, _ = loss_terms(model, xi, adj, ti, yi)
            loss = [term.mean() for term in terms]
            log['loss'][i] = [loss[0].item(), loss[1].item(), loss[2].item(), loss[3].item(), loss[4].item()]
            
            if args.verbose:
                print(f'{i:04d}/{args.epochs} : l_data = ({loss[0]:.2e}, {loss[1]:.2e}), l_pde = {loss[2]:.2e},' 
                      f' l_gpde = {loss[3]:.2e},  l_ent = {loss[4]:.2e};' 
                      f' lr = {schedule(i).item():.2e} (time: {time.time()-tic:.1f} s)')
            tic = time.time()
            
            #bsize,gsize,j = 0,0,0
            #while bsize<args.batch_size or gsize!=adj_test.shape[1]:
            #    x, adj, pe, _   = random_subgraph(x_train, adj_train, pe_train, batch_size=args.batch_size, key=prng(i+j))
            #    bsize = x.shape[0]
            #    gsize = adj.shape[1]
            #    j +=1
            batch = next(iter(loader))
            idx_batch = batch.idx.flatten().numpy()
            x_batch, adj_batch, pe_batch = subgraph(index=idx_batch, x=x_train, adj=adj_train, pe=pe_train, pad=True)
            
            model = eqx.tree_inference(model, value=False)
            
        if i % args.log_freq * 100 == 0:
            utils.save_model(model, log, stamp=stamp)

    log['wall_time'] = int(time.time()) - int(stamp) 
    utils.save_model(model, log, stamp=stamp)
