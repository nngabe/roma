from typing import Any, Optional, Sequence, Tuple, Union, Callable, List
import math

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.numpy import concatenate as cat
import jax.tree_util as tree

import equinox as eqx
import equinox.nn as nn
from equinox.nn import Dropout as dropout

import jraph

prng_key = jax.random.PRNGKey(0)
prng = lambda i=0: jr.PRNGKey(i)
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

from jax import jit
from functools import partial

@partial(jit, static_argnums=(2,))
def get_spline_basis(x_ext, grid, k=3):

    grid = jnp.expand_dims(grid, axis=2)
    x = jnp.expand_dims(x_ext, axis=1)
    basis_splines = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(float)
    
    for K in range(1, k+1):
        left_term = (x - grid[:, :-(K + 1)]) / (grid[:, K:-1] - grid[:, :-(K + 1)])
        right_term = (grid[:, K + 1:] - x) / (grid[:, K + 1:] - grid[:, 1:(-K)])
        basis_splines = left_term * basis_splines[:, :-1] + right_term * basis_splines[:, 1:]

    return basis_splines

@jax.jit
def solve_full_lstsq(A_full, B_full):
    solve_full = jax.vmap(solve_single_lstsq, in_axes=(0,0))
    full_solution = solve_full(A_full, B_full)

    return full_solution

class KANLayer(eqx.Module):
    """
    KANLayer class.
    Args:
        k (int): the order of the spline basis functions. Default: 3
        const_spl (float/bool): coefficient of the spline function in the overall activation. If set to False, then it is trainable per activation. Default: False
        const_res (float/bool): coefficient of the residual function in the overall activation. If set to False, then it is trainable per activation. Default: False
        residual (nn.Module): function that is applied on samples to calculate residual activation. Default: nn.swish
        noise_std (float): noise for the initialization of spline coefficients. Default: 0.1
        grid_e (float): parameter that defines if the grids are uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). 
                        Intermediate values correspond to a linear mixing of the two cases. Default: 0.05
    """
    
    in_dim: int
    out_dim: int
    k: int
    c_spl: float or bool
    c_res: float or bool
    residual: eqx.Module
    noise_std: float
    grid_e: float
    grid: jnp.array = eqx.field(static=True)
    c_basis: jnp.array
    
    def __init__(self, in_dim, out_dim, k=3, const_spl = False, const_res = False, residual = jax.nn.swish, noise_std = 0.1, grid_e = 0.15, key=prng(0)):
        super(KANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        #self.const_spl = const_spl
        #self.const_res = const_res
        self.residual = residual
        self.noise_std = noise_std
        self.grid_e = grid_e

        init_G = 3
        init_knot = (-1, 1)

        h = (init_knot[1] - init_knot[0]) / init_G

        grid = jnp.arange(-self.k, init_G + self.k + 1, dtype=jnp.float32) * h + init_knot[0]

        grid = jnp.expand_dims(grid, axis=0)
        grid = jnp.tile(grid, (self.in_dim*self.out_dim, 1))

        self.grid = grid #self.variable('state', 'grid', lambda: grid)
        #self.c_basis = self.param('c_basis', initializers.normal(stddev=self.noise_std), (self.in_dim * self.out_dim, self.grid.value.shape[1]-1-self.k))
        self.c_basis = self.noise_std * jr.normal(key, (self.in_dim * self.out_dim, self.grid.shape[1]-1-self.k))
        
        if isinstance(const_spl, float):
            self.c_spl = jnp.ones(self.in_dim*self.out_dim) * self.const_spl
        elif const_spl is False:
            #self.c_spl = self.param('c_spl', initializers.constant(1.0), (self.in_dim * self.out_dim,))
            self.c_spl = jnp.ones(self.in_dim * self.out_dim)

        if isinstance(const_res, float):
            self.c_res = jnp.ones(self.in_dim * self.out_dim) * self.const_res
        elif const_res is False:
            #self.c_res = self.param('c_res', initializers.constant(1.0), (self.in_dim * self.out_dim,))
            self.c_res = jnp.ones(self.in_dim * self.out_dim) #* self.const_res


    def basis(self, x):
        batch = x.shape[0]
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.out_dim,)).reshape((batch, self.in_dim * self.out_dim))
        x_ext = jnp.transpose(x_ext, (1, 0))
        
        grid = self.grid
        k = self.k
        bases = get_spline_basis(x_ext, grid, k)
        
        return bases

    
    def new_coeffs(self, x, ciBi):
        
        A = self.basis(x) # shape (in_dim*out_dim, G'+k, batch)
        Bj = jnp.transpose(A, (0, 2, 1)) # shape (in_dim*out_dim, batch, G'+k)
        ciBi = jnp.expand_dims(ciBi, axis=-1)
        cj = solve_full_lstsq(Bj, ciBi)
        cj = jnp.squeeze(cj, axis=-1)
        
        return cj


    def update_grid(self, x, G_new):
        
        Bi = self.basis(x) # (in_dim*out_dim, G+k, batch)
        ci = self.c_basis # (in_dim*out_dim, G+k)
        ciBi = jnp.einsum('ij,ijk->ik', ci, Bi) # (in_dim*out_dim, batch)

        batch = x.shape[0]
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.out_dim,)).reshape((batch, self.in_dim * self.out_dim))
        x_ext = jnp.transpose(x_ext, (1, 0))
        x_sorted = jnp.sort(x_ext, axis=1)

        ids = jnp.concatenate((jnp.floor(batch / G_new * jnp.arange(G_new)).astype(int), jnp.array([-1])))
        grid_adaptive = x_sorted[:, ids]
        
        margin = 0.01
        uniform_step = (x_sorted[:, -1] - x_sorted[:, 0] + 2 * margin) / G_new
        grid_uniform = (
            jnp.arange(G_new + 1, dtype=jnp.float32)
            * uniform_step[:, None]
            + x_sorted[:, 0][:, None]
            - margin
        )

        grid = self.grid_e * grid_uniform + (1.0 - self.grid_e) * grid_adaptive

        h = (grid[:, [-1]] - grid[:, [0]]) / G_new
        left = jnp.squeeze((jnp.arange(self.k, 0, -1)*h[:,None]), axis=1) 
        right = jnp.squeeze((jnp.arange(1, self.k+1)*h[:,None]), axis=1) 
        grid = jnp.concatenate(
            [
                grid[:, [0]] - left,
                grid,
                grid[:, [-1]] + right
            ],
            axis=1,
        )

        self.grid.value = grid
        cj = self.new_coeffs(x, ciBi)

        return cj


    def __call__(self, x, key=prng(0)):
        """
        Args:
            x (jnp.array): inputs:  shape (batch, in_dim)
        Returns:
            y (jnp.array): output of the forward pass, corresponding to the weighted sum of the B-spline activation and the residual activation: shape (batch, out_dim)
            spl_reg (jnp.array): the array relevant to the B-spline activation, to be used for the calculation of the loss function: shape (out_dim, in_dim)
        """
        batch = x.shape[0]
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.out_dim,)).reshape((batch, self.in_dim * self.out_dim))
        x_ext = jnp.transpose(x_ext, (1, 0))
        
        res = jnp.transpose(self.residual(x_ext), (1,0))

        Bi = self.basis(x) # (in_dim*out_dim, G+k, batch)
        ci = self.c_basis # (in_dim*out_dim, G+k)
        spl = jnp.einsum('ij,ijk->ik', ci, Bi) # (in_dim*out_dim, batch)
        spl = jnp.transpose(spl, (1,0))

        cnst_spl = jnp.expand_dims(self.c_spl, axis=0)
        cnst_res = jnp.expand_dims(self.c_res, axis=0)
        y = (cnst_spl * spl) + (cnst_res * res) # (batch, in_dim*out_dim)
        y_reshaped = jnp.reshape(y, (batch, self.out_dim, self.in_dim))
        y = (1.0/self.in_dim)*jnp.sum(y_reshaped, axis=2)

        grid_reshaped = self.grid.reshape(self.out_dim, self.in_dim, -1)
        input_norm = grid_reshaped[:, :, -1] - grid_reshaped[:, :, 0] + 1e-5
        spl_reshaped = jnp.reshape(spl, (batch, self.out_dim, self.in_dim))
        spl_reg = (jnp.mean(jnp.abs(spl_reshaped), axis=0))/input_norm

        return y, spl_reg



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


