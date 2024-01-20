from typing import Any, Optional, Sequence, Tuple, Union, List
import copy

import manifolds
import layers.hyp_layers as hyp_layers
from layers.layers import GCNConv, GATConv, Linear, get_dim_act
import utils.math_utils as pmath
import lib.function_spaces

import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from equinox import Module, static_field
from jaxtyping import Array, Float, PRNGKeyArray

prng = lambda i=0: jr.PRNGKey(i)

class null(eqx.Module):
    def __init__(self, args):
        super(null, self).__init__()
    def __call__(self, x=None, adj=None):
        return 0.

class GraphNet(eqx.Module):
    c: float
    skip: bool
    layers: eqx.nn.Sequential
    lin: eqx.nn.Sequential
    encode_graph: bool
    res: bool
    manifold: Optional[manifolds.base.Manifold] = None
    pe_dim: int
    u_dim: int

    def __init__(self, args):
        super(GraphNet, self).__init__()
        self.c = args.c
        self.skip = args.skip
        self.res = args.res
        self.pe_dim = args.pe_dim
        self.u_dim = args.kappa

    def __call__(self, x, adj=None, w=None, key=prng(0)):
        if self.res:
            return self._res(x, adj, w, key)
        elif self.skip:
            return self._cat(x, adj, w, key)
        else:
            return self._forward(x, adj, w, key)

    def exp(self, x):
        x = self.manifold.proj_tan0(x, c=self.c)
        x = self.manifold.expmap0(x, c=self.c)
        x = self.manifold.proj(x, c=self.c)
        return x

    def log(self, y):
        y = self.manifold.logmap0(y, self.c)
        y = y * jnp.sqrt(self.c) * 1.4763057
        return y

    def _forward(self, x, adj, w, key):
        for layer in self.layers:
            if self.encode_graph:
                x,_ = layer(x, adj, w, key)
            else:
                x = layer(x)
            key = jax.random.split(key)[0]
        return x 

    def _cat(self, x, adj, w, key):
        x = self.exp(x)
        if self.pe_dim>0: x_i = [x[:,:self.u_dim], x[:,self.u_dim:]]
        else: x_i = [x]
        for layer in self.layers:
            x,_ = layer(x, adj, w, key)
            x_i.append(x)
            key = jax.random.split(key)[0]
        return jnp.concatenate(x_i, axis=1)
    
    def _res(self, x, adj, w, key):
        x = self.exp(x)
        if self.pe_dim>0: x_i = [x[:,:self.u_dim], x[:,self.u_dim:]]
        for conv,lin in zip(self.layers,self.lin):
            h,_ = conv(x, adj, w, key)
            x = jax.vmap(lin)(self.log(x)) + self.log(h)
            x = self.exp(x)
            key = jax.random.split(key)[0]
        return x

class AttentionBlock(eqx.Module):
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    def __init__(
        self,
        input_shape: int,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
        key: PRNGKeyArray,
    ):
        key1, key2, key3 = jr.split(key, 3)

        self.layer_norm1 = eqx.nn.LayerNorm(input_shape)
        self.layer_norm2 = eqx.nn.LayerNorm(input_shape)
        self.attention = eqx.nn.MultiheadAttention(num_heads, input_shape, key=key1)

        self.linear1 = eqx.nn.Linear(input_shape, hidden_dim, key=key2)
        self.linear2 = eqx.nn.Linear(hidden_dim, input_shape, key=key3)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(self, x, key): 
        input_x = jax.vmap(self.layer_norm1)(x)
        x = x + self.attention(input_x, input_x, input_x)

        input_x = jax.vmap(self.layer_norm2)(x)
        input_x = jax.vmap(self.linear1)(input_x)
        input_x = jax.nn.gelu(input_x)

        key1, key2 = jr.split(key, num=2)

        input_x = self.dropout1(input_x, key=key1)
        input_x = jax.vmap(self.linear2)(input_x)
        input_x = self.dropout2(input_x, key=key2)

        x = x + input_x

        return x

class Transformer(eqx.Module):
    positional_embedding: jnp.ndarray
    attention_blocks: list[AttentionBlock]
    dropout: eqx.nn.Dropout
    ln1: eqx.nn.LayerNorm
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    num_layers: int

    def __init__(self, in_dim, out_dim, args, key):
        key1, key2, key3, key4, key5 = jr.split(key, 5)
        keys = jr.split(key,10)
        hidden_dim = args.dec_width
        num_heads = args.num_heads
        dropout_rate = args.dropout_branch
        self.num_layers = args.dec_depth 

        self.positional_embedding = jr.normal(key2, (args.num_nodes + 1, in_dim))
        self.attention_blocks = [ AttentionBlock(hidden_dim, hidden_dim, num_heads, dropout_rate, keys[i]) for i in range(self.num_layers)]

        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.ln1 = eqx.nn.LayerNorm(hidden_dim)
        self.lin1 = eqx.nn.Linear(in_dim, hidden_dim, key=key4)
        self.lin2 = eqx.nn.Linear(hidden_dim, out_dim, key=key5)

    def __call__(self, x, key):
        x += self.positional_embedding[:x.shape[0]]  # Slice to the same length as x, as the positional embedding may be longer.
        dropout_key, *attention_keys = jr.split(key, num=self.num_layers + 1)
        x = jax.vmap(self.lin1)(x)
        x = self.dropout(x, key=dropout_key)
        for block, attention_key in zip(self.attention_blocks, attention_keys):
            x = block(x, key=attention_key)

        x = jax.vmap(self.ln1)(x)
        x = jax.vmap(self.lin2)(x)

        return x

class MLP(eqx.Module):
    num_layers: int
    layers: List[eqx.Module]
    resnet: bool

    def __init__(self, in_dim, out_dim, args, key, ln=True, resnet=True):
        keys = jr.split(key, args.dec_depth + 2 )
        hidden_dim = args.dec_width
        self.num_layers = args.dec_depth
        self.resnet = resnet
        dropout_rate = args.dropout_trunk
        dims = [in_dim] + self.num_layers * [args.dec_width] + [out_dim]
        self.layers = [Linear(dims[i], dims[i+1], dropout_rate=dropout_rate, key=keys[i]) for i in range(self.num_layers+1)]

    def __call__(self, x, key):

        for layer in self.layers:
            if len(x.shape)==1:
                x = layer(x, key)
            if len(x.shape)==2:
                x = jax.vmap(lambda x: layer(x,key))(x)
            key = jr.split(key)[0]
        return x


class DeepOnet(eqx.Module):
    
    trunk: eqx.Module
    branch: eqx.Module
    func_space: eqx.Module
    drop_fn: eqx.nn.Dropout
    norm: List[eqx.nn.LayerNorm]
    x_dim: int
    tx_dim: int
    u_dim: int
    p_dim: int
    trunk_dims: List[int]
    branch_dims: List[int]

    def __init__(self, args, module): 
        super(DeepOnet, self).__init__()
        self.func_space = getattr(lib.function_spaces, args.func_space)()
        dims, act, _ = get_dim_act(args,module)
        self.drop_fn = eqx.nn.Dropout(args.dropout)
        self.norm = []
        self.x_dim = args.x_dim
        self.tx_dim = args.time_dim + args.x_dim 
        self.u_dim = args.kappa
        self.p_dim = args.p_basis

        key1, key2, key3, key4, key5 = jr.split(prng(), 5)
        
        # set dimensions of branch net        
        self.branch_dims = copy.copy(dims)
        self.branch_dims[0] = self.u_dim
        self.branch_dims[-1] = self.p_dim
        self.branch_dims[-1] *= args.x_dim

        # set dimensions of trunk net
        self.trunk_dims = copy.copy(dims)
        self.trunk_dims[0] = dims[0] - self.u_dim 
        self.trunk_dims[-1] = self.p_dim

        self.branch = Transformer(self.branch_dims[0], self.branch_dims[-1], args, key1)
        self.trunk = MLP(self.trunk_dims[0], self.trunk_dims[-1], args, key2)

    def __call__(self, x, adj=None, w=None, key=prng(0)):
        keys = jax.random.split(key,10)
        tx, uz = x[:self.tx_dim], x[self.tx_dim:]
        u, z = uz[:self.u_dim], uz[self.u_dim:]
        
        #u = self.func_space(u)
        b = self.branch(u, keys[1])
        
        txz = jnp.concatenate([tx,z],axis=-1)
        t = self.trunk(txz, keys[0])
        
        b = b.reshape(-1, self.p_dim, self.x_dim)
        t = t.reshape(-1, self.p_dim)
        G = jnp.einsum('ijk,ij -> ik', b, t) / self.p_dim
        
        return G


class HGCN(GraphNet):
    
    curvatures: jax.numpy.ndarray 
    
    def __init__(self, args, module):
        super(HGCN, self).__init__(args)
        self.manifold = getattr(manifolds, args.manifold)()
        dims, act, self.curvatures = get_dim_act(args,module)
        self.curvatures.append(args.c)
        hgc_layers = []
        lin_layers = []
        key, subkey = jax.random.split(prng(0))
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            hgc_layers.append( hyp_layers.HGCNLayer( key, self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att))
            if in_dim==out_dim:
                lin_layers.append( lambda x: x)
            else:
                lin_layers.append( eqx.nn.Linear(in_dim, out_dim, key=key))
            key = jax.random.split(key)[0]
        self.layers = nn.Sequential(hgc_layers)
        self.lin = nn.Sequential(lin_layers)
        self.encode_graph = True

    def __call__(self, x, adj, w=None, key=prng(0)):
        return super(HGCN, self).__call__(x, adj, w, key)

