import warnings
warnings.filterwarnings('ignore')

## plotting preferences
import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize'] = [8,6]; plt.rcParams['font.size'] = 24; plt.rcParams['xtick.major.size'] = 8
#plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'; plt.rcParams['text.usetex'] = True

import argparse
import time
import pandas as pd
import networkx as nx
import numpy as np
import glob
import os,sys
import time
from collections import defaultdict

from community import community_louvain
import jax
import jax.numpy as jnp
import jraph
import scipy

import graph

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', nargs='?', default=1e-16, type=float)
    parser.add_argument('--N', nargs='?', default=10000, type=int)
    parser.add_argument('--c', nargs='?', default=1., type=float)
    parser.add_argument('--K', nargs='?', default=5., type=float)
    parser.add_argument('--a', nargs='?', default=.5, type=float)
    parser.add_argument('--b', nargs='?', default=10., type=float)
    parser.add_argument('--g', nargs='?', default=10., type=float)
    parser.add_argument('--d', nargs='?', default=.5, type=float)
    parser.add_argument('--steps', nargs='?', default=20000, type=int)
    parser.add_argument('--dt', nargs='?', default=1e-3, type=float)
    parser.add_argument('--save', nargs='?', default=0, type=int)
    parser.add_argument('--seed', nargs='?', default=123, type=int)
    parser.add_argument('--plot', nargs='?', default=False, type=bool)
    parser.add_argument('--gplot', nargs='?', default=False, type=bool)
    parser.add_argument('--periods', nargs='?', default=1., type=float)
    parser.add_argument('--edge_path', nargs='?', default=None, type=str)
    parser.add_argument('--downsample', nargs='?', default=1, type=int) 
    args = parser.parse_args()


    np.random.seed(123)
    PATH = 'data/' #
    PATH = os.path.expanduser('~') + '/data/' 
    time_run = time.time()
    tic = time.time()
    avg_degree = 3.
    c=args.c
    N=args.N
    seed=71

    tic = time.time()
    print(f'\n Getting graph...',end='')
    if args.edge_path:
        path = glob.glob(f'{PATH}edges*{args.edge_path}*parquet')[0]
        G = pd.read_parquet(path).T.to_numpy()
        edge_index = np.array(G)
    else:
        print(f' computing graph with {N} nodes and curvature {c}...')
        G, comms = graph.gen_graph(N, avg_degree, c, seed=seed, plot=args.gplot)
        louvain_dict = defaultdict(list)
        for key, value in comms.items():
            louvain_dict[value].append(key)
        edge_index = np.array(list(G.edges)).T

    print(f' Done. (time: {time.time()-tic:.1f} s)\n')
    N = np.unique(edge_index).shape[0] 
    print(f'  N = {N}')
    print(f' |E| = {edge_index.shape[1]}')
    
    if edge_index.max()>N:
        nodes = jnp.unique(edge_index).tolist()
        re = {old:new for new,old in enumerate(nodes)}
        edge_index = graph.reindex(edge_index, re) 

    edge_index = np.concatenate([edge_index,edge_index[::-1]], axis=-1)
    edge_index = np.unique(edge_index, axis=1)
    s,r = edge_index


    edge_file = f'{PATH}edges_N{N}_c{c}_t{int(time_run)}.parquet'
    pd.DataFrame(edge_index, index=np.arange(edge_index.shape[0]).astype(str)).T.to_parquet(edge_file)
    print(f'\n edge file: {edge_file} written...')

    ####
    tic = time.time()
    print(f' Getting pure gels...',end='')
    GPATH = os.path.expanduser('~') + '/tmp/G-N1k-F1/*'
    paths = glob.glob(GPATH)
    n = len(paths)
    gel = [None for _ in range(n)]
    for i in range(n):
        gel[i] = pd.read_csv(paths[i],sep=' ',index_col=0)
        k = paths[i].split('-')[-1][:-4]
        gel[i].index.name = None
        gel[i].columns = [k]

    print(f' Done. (time: {time.time()-tic:.1f} s)\n')
    tic = time.time()
    print(f' Computing linear combinations of gels...',end='')
    
    gels = pd.concat(gel,axis=1)
    T,_ = gels.shape
    
    i = np.random.randint(0,n,(N,3))
    x = gels.to_numpy()[:,i].sum(2)/3.

    print(f' Done. (time: {time.time()-tic:.1f} s)\n')
    tic = time.time()

    print(f' Computing normalized weights and preprocessing...',end='')
    edge_ones = jnp.ones_like(edge_index[0])
    w = jraph.segment_sum(edge_ones, r, N) / N
    norm = jraph.segment_sum(w[s], r, N)
    w_ij = w[s]/norm[r]

    def f(x,t):
        m_ij = jnp.square(x[s])
        msg_i = jnp.sqrt(jraph.segment_sum(m_ij*w_ij, r, N))
        return omega_i + msg_i

    # set weights with conditions: 
    # (1) sender weights are proportional to degree
    # (2) receiver weights add to 1.0
    edge_ones = jnp.ones_like(edge_index[0])
    w = jraph.segment_sum(edge_ones, r, N) / N
    norm = jraph.segment_sum(w[s], r, N)
    w_ij = w[s]/norm[r] 

    xb = pd.DataFrame(x).fillna(0.).rolling(64, win_type='gaussian').sum(std=8).diff().dropna().T.to_numpy()
    xb = jnp.array(xb)
    

    print(f' Done. (time: {time.time()-tic:.1f} s)\n')

    tic=time.time()
    print(f' Computing message passing updates...', end='') 
    
    def update(xb, red=4., iters=3):
        for i in range(iters):
            def f(t):
                m_ij = (xb[s,t])
                msg_i = jnp.sqrt(jraph.segment_mean( jnp.square(m_ij*w_ij), r, N)) / red
                return msg_i
            
            t = jnp.arange(0,xb.shape[1]-1,1)
            msg = jax.vmap(f)(t).T
            xb = xb.at[:,1:].add(msg) 
        return xb

    def to_df(x):
        df = pd.DataFrame(x).cumsum(1)
        df = df/df.max().max()
        return df

    _update = jax.jit(lambda x: update(x, red=4., iters=6))

    xb3 = _update(xb)

    
    print(f' Done. (time: {time.time()-tic:.1f} s)\n')
    print(f' xb.mean = {xb.mean()/xb.mean()}, xb3.mean = {xb3.mean()/xb.mean():.3f}')

    df3 = to_df(xb3)

    k = 3
    data = eval('xb' + str(k))
    x_path = f'{PATH}x_burg_k{k}_N{N}_c{c}_t{int(time_run)}.parquet'
    print(f'\n writing {x_path} to file...')
    print(f'\n x.shape = {data.shape}\n\n')
    df2 = pd.DataFrame(data).cumsum(1)
    df2 = df2/df2.max().max()
    df2.columns = df2.columns.astype(str)
    df2.to_parquet(x_path, compression=None)
