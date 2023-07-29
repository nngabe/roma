#import warnings
#warnings.filterwarnings('ignore')

import os
import sys
import copy
import time
import glob
import numpy as onp
import pandas as pd
from typing import Any, Optional, Sequence, Tuple, Union, Dict

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from nn.models.cosynn import COSYNN, loss_bundle, compute_bundle_terms, make_step
from config import parser, set_dims

from lib import utils
from lib.graph_utils import subgraph, random_subgraph, louvain_subgraph, add_self_loops, sup_power_of_two, pad_graph


prng = lambda i=0: jax.random.PRNGKey(i)


if __name__ == '__main__':
    args = parser.parse_args()
    args.data_path = glob.glob(f'../data_cosynn/gels*{args.path}*')[0]
    args.adj_path = glob.glob(f'../data_cosynn/adj*{args.path.split("_")[:-1]}*')[0]

    A = pd.read_csv(args.adj_path, index_col=0).to_numpy()
    adj = jnp.array(jnp.where(A))
    x = jnp.array(pd.read_csv(args.data_path, index_col=0).dropna().to_numpy().T)
    n,T = x.shape

    adj = add_self_loops(adj)
    x_test, adj_test, idx_test = louvain_subgraph(x, adj, batch_size=n//10)
    idx_train = jnp.where(jnp.ones(n, dtype=jnp.int32).at[idx_test].set(0))[0]    
    x_train, adj_train, idx_train = subgraph(idx_train, x, adj)

    args.batch_size = sup_power_of_two(n//args.batch_red)
    args.pool_dims[-1] = sup_power_of_two(2 * n//args.pool_red)
    if args.log_path:
        model, args = utils.read_model(args)
    else:
        model = COSYNN(args)
    
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
    
    schedule = optax.warmup_exponential_decay_schedule(args.lr, peak_value=args.lr, warmup_steps=args.epochs//10,
                                                        transition_steps=args.epochs, decay_rate=1e-2, end_value=args.lr/1e+3)
    optim = optax.chain(optax.clip(args.max_norm), optax.adamw(learning_rate=schedule)) 
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @jax.jit
    def _batch(x,idx):
        win = jnp.arange(1 - args.kappa, 1, 1)
        xb = x.at[:, idx + win].get()
        xb = jnp.swapaxes(xb,0,1)
        return xb

    def _taus(i, size=args.tau_num, tau_max=args.tau_max):
        taus = jax.random.randint(prng(i), (size,), 1, tau_max) 
        taus = jnp.clip(taus, 1, tau_max)
        return taus
   
    stamp = str(int(time.time()))
    log['loss'] = {}
    x, adj, _   = random_subgraph(x_train, adj_train, batch_size=args.batch_size, seed=0)
   
    sched_i = 0, 0 
    for i in range(args.epochs):
        ti = jax.random.randint(prng(i), (10, 1), args.kappa, T - args.tau_max).astype(jnp.float32)
        idx = ti.astype(int)
        taus = _taus(i)
        bundles = idx + taus
        yi = x[:,bundles].T
        xi = _batch(x, idx)
        mode = -1 if i<sched_i[0] else 0
        loss, grad = loss_bundle(model, xi, adj, ti, taus, yi, key=prng(i), mode=mode)
        grad = jax.tree_map(lambda x: 0. if jnp.isnan(x).any() else x, grad) 
        
        model, opt_state = make_step(grad, model, opt_state, optim)
        if i % args.log_freq == 0:
            x, adj = x_test, adj_test
            model = eqx.tree_inference(model, value=True) 

            ti = jnp.linspace(args.kappa, T - args.tau_max , 100).reshape(-1,1)
            idx = ti.astype(int)
            taus = jnp.arange(1, args.tau_max, 10).astype(int)
            bundles = idx + taus
            yi = x[:,bundles].T
            xi = _batch(x, idx)
            
            terms = compute_bundle_terms(model, xi, adj, ti, taus, yi)
            loss = [term.mean() for term in terms]
            log['loss'][i] = [loss[0].item(), loss[1].item(), loss[2].item(), loss[3].item()]
            if args.verbose:
                print(f'{i:04d}/{args.epochs}: loss_data = {loss[0]:.2e}, loss_pde = {loss[1]:.2e}, loss_gpde = {loss[2]:.2e}, loss_ent = {loss[3]:.2e}  lr = {schedule(i).item():.4e}')
            x, adj, _   = random_subgraph(x_train, adj_train, batch_size=args.batch_size, seed=i)
            if i<sched_i[1]:
                model = eqx.tree_inference(model, value=False)
        if i % args.log_freq * 10 == 0:
            utils.save_model(model, log, stamp=stamp)

    log['wall_time'] = int(time.time()) - int(stamp) 
    utils.save_model(model, log, stamp=stamp)
