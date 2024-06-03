from typing import Any, Optional, Sequence, Tuple, Union, Callable, List
import math

import numpy as np

import jax
import jax.numpy as jnp
from jax.numpy import concatenate as cat
import jax.tree_util as tree

import equinox as eqx
import equinox.nn as nn
from equinox.nn import Dropout as dropout

import jraph

prng_key = jax.random.PRNGKey(0)
act_dict = {
            'relu': jax.nn.relu, 
            'silu': jax.nn.silu, 
            'lrelu': jax.nn.leaky_relu, 
            'gelu': jax.nn.gelu, 
            'tanh': jax.nn.tanh,
            'sigmoid': jax.nn.sigmoid,
            }

def get_dim_act(args, module):
    """
    Helper function to get dimension and activation for each module (enc, dec, renorm, pde).
    """
    act = act_dict[args.act] 
    if module == 'enc':
        args.num_layers = len(args.enc_dims)
        dims = args.enc_dims
        args.skip = 0
    elif module == 'dec': 
        args.num_layers = len(args.dec_dims)
        dims = args.dec_dims
    elif module == 'pde': 
        args.num_layers = len(args.pde_dims)
        dims = args.pde_dims
    elif module == 'pool': 
        args.res = 1
        args.cat = 0
        args.num_layers = len(args.pool_dims)
        dims = args.pool_dims
        args.pool_dims[-1] = max(args.pool_dims[-1]//args.pool_red, 1)
    elif module == 'embed': 
        args.res = 1
        args.cat = 0
        dims = args.embed_dims
    else:
        print('All layers already init-ed! Define additional layers if needed.')
        raise
    
    # for now curvatures are static, change list -> jax.ndarray to make them learnable
    if args.c is None:
        curvatures = [1. for _ in range(args.num_layers)]
    else:
        curvatures = [args.c for _ in range(args.num_layers)]

    return args, dims, act, curvatures

class Linear(eqx.Module): 
    linear: eqx.nn.Linear
    act: Callable
    dropout: Callable
    ln: eqx.nn.LayerNorm
    norm: bool    
    def __init__(self, in_features, out_features, dropout_rate=0., act=jax.nn.gelu, key=prng_key, norm=True):
        super(Linear, self).__init__()
        self.linear = eqx.nn.Linear(in_features, out_features,  key=key)
        self.act = act
        self.dropout = dropout(dropout_rate)
        self.ln = eqx.nn.LayerNorm(out_features)
        self.norm = norm
    def __call__(self, x, key=prng_key):
        x = self.dropout(x, key=key)
        x = self.linear(x)
        x = self.act(x)
        x = self.ln(x) if self.norm else x
        return x

class GCNConv(eqx.Module):
    dropout_rate: float
    linear: eqx.nn.Linear
    act: Callable
    dropout: Callable

    def __init__(self, in_features, out_features, p=0., act=jax.nn.gelu, use_bias=True):
        super(GCNConv, self).__init__()
        self.dropout_rate = p
        self.linear = nn.Linear(in_features, out_features, use_bias, key=prng_key)
        self.act = act
        self.dropout = dropout(self.dropout_rate)

    def __call__(self, x, adj):
        n = x.shape[0]
        s, r = adj[0], adj[1]
        count_edges = lambda x: jax.ops.segment_sum(jnp.ones_like(s), x, n)
        sender_degree = count_edges(s)
        receiver_degree = count_edges(r)    

        h = jax.vmap(self.linear)(x)
        h = jax.vmap(self.dropout)(h) 
        h = tree.tree_map(lambda x: x * jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, None], h)
        h = tree.tree_map(lambda x: jax.ops.segment_sum(x[s], r, n), h)
        h = tree.tree_map(lambda x: x * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))[:, None], h)
        h = self.act(h)

        output = h, adj
        return output




class GATConv(eqx.Module):
    linear: eqx.nn.Linear
    a: eqx.nn.Linear
    W: eqx.nn.Linear
    act: Callable
    dropout: Callable

    def __init__(self, in_features, out_features, p=0., act=jax.nn.gelu, use_bias=True, num_heads=3, query_dim=8):
        super(GATConv, self).__init__()
        self.dropout = dropout(p)
        self.W = nn.Linear(in_features, query_dim * num_heads, key=prng_key) 
        self.a = nn.Linear( 2 * query_dim * num_heads, num_heads, key=prng_key) 
        self.linear = nn.Linear(in_features, out_features,  key=prng_key)
        self.act = act

    def __call__(self, x, key=prng_key):
        x, adj = input
        n = x.shape[0]
        s = r = jnp.arange(0,n)
        attr = jax.vmap(self.W)(x)
        sender_attr = attr[s]
        receiver_attr = attr[r]

        e = jnp.concatenate((sender_attr,receiver_attr), axis=1)
        alpha = jax.vmap(self.a)(e)
        
        h = dropout(h, key=key)
        h = tree.tree_map(lambda x: jax.segment_sum(x[s] * alpha[s], r, n), h)
        h = self.act(h)

        output = h, adj
        return output


