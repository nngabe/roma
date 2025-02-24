from typing import Union, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch_geometric.loader import GraphSAINTRandomWalkSampler
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt

prng = lambda: jax.random.PRNGKey(0)

def get_next_batch(loader, args, data):
    bsize = 0
    patience = 100
    for i in range(patience):
        batch = next(iter(loader))
        bsize=batch.num_nodes
        if bsize == args.batch_size:
            break
    #print(f'{bsize}: walk_length = {args.batch_walk_len}, sampler_batch_size = {args.sampler_batch_size}')
    if bsize > args.batch_size: 
        args.sampler_batch_size = max(1,args.sampler_batch_size-1)
        loader = GraphSAINTRandomWalkSampler(data, batch_size=args.sampler_batch_size, walk_length=args.batch_walk_len)
        batch, loader = get_next_batch(loader, args, data)
    if bsize < args.batch_size:
        args.sampler_batch_size += 1
        loader = GraphSAINTRandomWalkSampler(data, batch_size=args.sampler_batch_size, walk_length=args.batch_walk_len)
        batch, loader = get_next_batch(loader, args, data)
    return batch, loader

def sup_power_of_two(x: int) -> int:
    y = 2
    while y < x:
        y *= 2
    return y

def pad_graph(x: np.ndarray, 
              adj: np.ndarray, 
              pe: np.ndarray,
              x_size: int = None, 
              adj_size: int = None,
              pad_x : bool = True) -> Tuple[np.ndarray, ...]:
    x_size = sup_power_of_two(x.shape[0]) if not x_size else x_size
    adj_size = sup_power_of_two(adj.shape[1]) if not adj_size else adj_size
    x_pad = 0.*np.ones((x_size-x.shape[0], x.shape[1]))
    pe_pad = 0.*np.ones((x_size-x.shape[0], pe.shape[1]))
    adj_pad = -1*np.ones((adj.shape[0], adj_size-adj.shape[1]), dtype=np.int32)
    if pad_x:
        return np.concatenate([x, x_pad], axis=0), np.concatenate([adj, adj_pad], axis=1), np.concatenate([pe, pe_pad], axis=0)
    else:
        return x, jnp.concatenate([adj, adj_pad],axis=1), pe

def threshold_subgraphs_by_size(edge_index, min_size=200):
    import networkx as nx
    adj = edge_index
    adj = [(i[0].item(),i[1].item()) for i in adj.T]

    G = nx.Graph()
    G.add_edges_from(adj)
    Gs = [g for g in nx.connected_components(G)]

    subgraph_sizes = np.array([len(g) for g in Gs])
    subgraphs = [list(g) for g in Gs]

    idx = np.where(subgraph_sizes > min_size)[0]
    subs = [subgraphs[i] for i in idx]   #; print(f' subraph_sizes = {[len(s) for s in subs]}')
    subgraph = np.concatenate(subs)      #; print(f' subgraph = {subgraph} (dim = {len(subgraph)})')
    return torch.tensor(subgraph)

def pad_adj(adj: jnp.ndarray,
            adj_size: int = None,
            fill_value: int = -1) -> jnp.ndarray:
    adj_size = sup_power_of_two(adj.shape[1]) if not adj_size else adj_size
    adj_pad = fill_value * jnp.ones((adj.shape[0], adj_size-adj.shape[1]), dtype=jnp.int32)
    return jnp.concatenate([adj, adj_pad],axis=1)

def index_to_mask(index, size):
    
    if not isinstance(index, jnp.ndarray): index = jnp.array(index)

    mask = jnp.zeros((size), dtype=bool)
    mask = mask.at[index].set(True)
    return mask

def subgraph(
    index: Union[jnp.ndarray, List[int]],
    x: jnp.ndarray,
    adj: jnp.ndarray,
    pe: jnp.ndarray,
    relabel_nodes: bool = True,
    pad: bool = True,
    pad_size: List[int] = [None,None],
    min_degree: int = 3
    ):
    """ get the subraph indexed by subset. """
    if not isinstance(index, jnp.ndarray): index = jnp.array(index)

    num_nodes = index.shape[0] #jnp.unique(jnp.concatenate(adj)).size
    node_mask = index_to_mask(index, size=num_nodes)
    edge_mask = node_mask[adj[0]] & node_mask[adj[1]]
    adj = adj[:, edge_mask]
    index, degree = jnp.unique(adj, return_counts=True)
    index = index[degree>=min_degree]
    
    x = x[index]
    pe = pe[index]
    num_nodes = index.shape[0] #jnp.unique(jnp.concatenate(adj)).size
    node_mask = index_to_mask(index, size=num_nodes)
    edge_mask = node_mask[adj[0]] & node_mask[adj[1]]
    adj = adj[:, edge_mask]

    if relabel_nodes:
        node_idx = jnp.zeros(node_mask.size, dtype=jnp.int32)
        node_idx = node_idx.at[index].set( jnp.arange(index.shape[0]) )
        adj = node_idx[adj]

    if pad: 
        x, adj, pe = pad_graph(x, adj, pe, x_size=pad_size[0], adj_size=pad_size[1])

    return x, adj, pe 


def dense_to_coo(A: jnp.ndarray) -> jnp.ndarray:

    adj = jnp.mask_indices(A.shape[0], lambda x,k: x)
    #adj = jnp.indices((size,size))
    #adj = jnp.array([adj[0].flatten(),adj[1].flatten()])
    adj = jnp.array([adj[0],adj[1]])
    w = A[adj[0],adj[1]]
    w = pad_adj(w.reshape(-1,1).T, fill_value=0)
    adj = pad_adj(adj)
    return adj, w[0]

def mask_pad(n: int, n_pad: int, flip: bool = False):
    mask = jnp.arange(0, n_pad, 1)<(n - 1)
    return mask.astype(jnp.int32)^flip

def random_subgraph( 
    x: jnp.array,
    adj: jnp.array,
    pe: jnp.array,
    batch_size: int = 100,
    key: jax.random.PRNGKey = prng(),
    init: jnp.int32 = 5, 
    relabel_nodes: bool = True,
    pad: bool = True,
    pad_size: List[int] = [None,None]
    ):
    """ obtain batch graph by hopping from initial nodes until desired batch_size is obtained.""" 
    num_nodes = jnp.unique(jnp.concatenate(adj)).size
    index = jax.random.randint(key, (1,), 0, num_nodes) 
    node_mask = index_to_mask(index, num_nodes)
    assert num_nodes > batch_size

    for i in range(100):
        if index.size > batch_size:
            break
        edge_mask = node_mask[adj[0]]
        _adj = jax.random.permutation(key, adj[:,edge_mask], axis=1)
        index = jnp.unique(jnp.concatenate(_adj))
        node_mask = node_mask.at[index].set(True)
    index = index[:batch_size]
    node_mask = index_to_mask(index, num_nodes)
    edge_mask = node_mask[adj[0]] & node_mask[adj[1]]
    adj = adj[:, edge_mask]
    
    if relabel_nodes:
        node_idx = jnp.zeros(node_mask.size, dtype=jnp.int32)
        node_idx = node_idx.at[index].set( jnp.arange(index.shape[0]) )
        adj = node_idx[adj]
    
    x, adj = pad_graph(x[index], adj, x_size=pad_size[0], adj_size=pad_size[1])
    pe = pe[index]
    return x, adj, pe, index

def louvain_subgraph(
    x: jnp.array,
    adj: jnp.array, 
    batch_size: int = 100,
    relabel_nodes: bool = True,
    pad: bool = True,
    pad_size: List[int] = [None,None]
    ):
    """ obtain batch graph by hopping from initial nodes until desired batch_size is obtained."""
    
    num_nodes = jnp.unique(jnp.concatenate(adj)).size
    graph = nx.Graph()
    graph.add_edges_from(onp.array(adj.T))
    comms = nx.community.louvain_communities(graph, resolution=1.)
    s = onp.array([len(c) for c in comms])
    i_min = onp.argmin(onp.abs(s-batch_size))
    index = jnp.array(list(comms[i_min]))
    x, adj, _ = subgraph(index, x, adj)
    
    x, adj = pad_graph(x, adj, x_size=pad_size[0], adj_size=pad_size[1])
 
    return x, adj, index

def to_undirected( 
    adj: jnp.ndarray
    ):

    tmp = jnp.concatenate([adj, adj[::-1]],axis=1)
    _, idx = jnp.unique(tmp.T, return_index=True, axis=0)
    tmp = tmp[:,idx]

    return tmp

def add_self_loops(
    adj: jnp.ndarray
    ):

    idx = jnp.unique(jnp.concatenate(adj))
    self_loops = jnp.array([idx,idx])
    tmp = jnp.concatenate([adj,self_loops], axis=1)
    _, reidx = jnp.unique(tmp.T, return_index=True) 
    tmp = tmp[:,reidx]

    return tmp

def gen_hyp(n, v, c, seed=131):    
    R = 2.*jnp.log(n/v)
    idx = jnp.array(jnp.triu_indices(n,1)).T
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key,10)
    
    idr = lambda x: np.arcsinh((x*(np.cosh(c*R)-1))/c)/c
    r,phi = idr(jax.random.uniform(keys[0], (n,))), 2*np.pi*jax.random.uniform(keys[1], (n,))
    r,phi = jnp.array(r), jnp.array(phi)
    r,phi = jax.lax.stop_gradient(r), jax.lax.stop_gradient(phi)
    def dist(i,j):
        dphi = jnp.pi - jnp.abs(jnp.pi - jnp.abs(phi[i]-phi[j]))
        return jnp.arccosh(jnp.cosh(r[i])*jnp.cosh(r[j]) - jnp.sinh(r[i])*jnp.sinh(r[j])*jnp.cos(dphi))
    
    @jax.jit
    def mask(e):
        i,j = e[0],e[1]
        return dist(i,j)<R
    
    chunk = int(1e+6)    
    edges = idx[:chunk][jax.vmap(mask)(idx[:chunk])]
    for j in range(1,idx.shape[0]//chunk): 
        _idx = idx[j*chunk:(j+1)*chunk]
        _e = _idx[jax.vmap(mask)(_idx)]
        edges = jnp.concatenate([edges, _e], axis=0)
        
    pos = jnp.array([r*np.cos(phi), r*np.sin(phi)]).T.tolist()
    
    return edges, pos

def draw(G, pos, part=None):
    cmap = 'jet'
    if part:
        nc = [p**.45 for p in part.values()]
        nx.draw(G, pos=pos, node_size=50, node_color=nc, width=.5, edgecolors='k',cmap=cmap)
    else:
        nx.draw(G, pos=pos, node_size=50, node_color='dodgerblue', width=.5, edgecolors='k',cmap=cmap)

def gen_graph(n, v, c, seed=123, plot=True):
    edges, pos = gen_hyp(n,v,c,seed)
    edges = [(e[0],e[1]) for e in edges.tolist()]
    G = nx.Graph()
    G.add_edges_from(edges)
    lcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc)
    re = {old:new for new,old in enumerate(lcc)}
    print(f'c = {c}, N = {G.number_of_nodes()}')
    partition = community_louvain.best_partition(G)
    if plot: 
        draw(G,pos,partition)
        plt.savefig(f'G_N{n}_c{c}_v{v}_seed{seed}.pdf')
    return G, partition, re

