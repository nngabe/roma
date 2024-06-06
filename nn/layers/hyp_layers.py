from typing import Any, Optional, Sequence, Tuple, Union, Callable
import math

import manifolds

import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy import concatenate as cat
import jax.tree_util as tree
import jraph

import equinox as eqx
import equinox.nn as nn
from equinox.nn import Dropout as dropout

prng = lambda i=0: jax.random.PRNGKey(i)
act_dict = {'relu': jax.nn.relu, 'silu': jax.nn.silu, 'lrelu': jax.nn.leaky_relu}
segment_softmax = lambda t: lambda vals, ids, n: jraph.segment_sum(vals * jraph.segment_softmax(t * vals, ids, n), ids, n)

class Attn(eqx.Module):
    
    mlp: eqx.nn.MLP
    heads: int
    
    def __init__(self, in_features, heads=6):
        super(DenseAtt, self).__init__()
        self.mlp = eqx.nn.MLP( 3 * in_features, heads, width_size= 6 * in_features, depth=2, activation=jax.nn.gelu, key = prng())
        self.heads = heads

    def __call__(self, x, adj = None):
        n = x.shape[0]
        if adj:
            s,r = adj
        else: # dense attention
            u1,u2 = jnp.triu_indices(n,1)
            l1,l2 = jnp.tril_indices(n)
            s,r = jnp.concatenate([u1,l1]), jnp.concatenate([u2,l2])

        x_cat = jnp.concatenate([x[s], x[r], x[r]-x[s]], axis=-1)
        attn = jax.vmap(self.mlp)(x_cat).reshape(self.heads,n,n)
        attn = jax.nn.softmax(attn)
        return attn


class HypLinear(eqx.Module):

    dropout: eqx.nn.Dropout
    manifold: manifolds.base.Manifold
    c: float
    linear: eqx.nn.Linear
    layer_norm: eqx.nn.LayerNorm

    def __init__(self, in_features, out_features, manifold, key, args):
        super(HypLinear, self).__init__()
        self.dropout = eqx.nn.Dropout(args.dropout) 
        self.manifold = manifold
        self.c = args.c
        self.linear = eqx.nn.Linear(in_features, out_features, key=key)
        self.layer_norm = eqx.nn.LayerNorm(out_features) if args.use_layer_norm else (lambda x: x)

    def __call__(self, x, key):
        mv = self.manifold.mobius_matvec(self.linear.weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        bias = self.manifold.proj_tan0(self.linear.bias.reshape(1, -1), self.c)
        hyp_bias = self.manifold.expmap0(bias, self.c)
        hyp_bias = self.manifold.proj(hyp_bias, self.c)
        res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
        res = self.manifold.proj(res, self.c)
        return res


class HypAgg(eqx.Module):

    manifold: manifolds.base.Manifold
    c: float
    use_att: bool
    agg: str
    attn: Attn
    edge_conv: eqx.Module
    agg_embed: eqx.Module
    dropout: Callable

    def __init__(self, in_features, manifold, args):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = args.c
        self.agg = args.agg
        self.use_att = args.use_att
        self.attn = DenseAtt(in_features, heads=args.num_gat_heads) if self.use_att else None
        self.edge_conv = eqx.nn.MLP(3 * in_features, in_features, width_size=6*in_features, depth=2, activation=jax.nn.gelu, key=prng(0)) if args.edge_conv=='mlp' else None
        self.edge_conv = eqx.nn.Linear(3 * in_features, in_features, key=prng(0)) if args.edge_conv=='linear' else self.edge_conv
        self.agg_embed = eqx.nn.MLP(4 * in_features, in_features, width_size=6*in_features, depth=2, activation=jax.nn.gelu, key=prng(1)) if args.agg=='multi' else None
        #self.agg_embed = eqx.nn.Linear(4 * in_features, in_features, width_size=4*in_features, depth=2, activation=jax.nn.gelu, key=prng(1)) if args.agg=='multi' else None
        self.dropout = eqx.nn.Dropout(args.dropout)

    def __call__(self, x, adj, key, w=None):
        s,r = adj[0],adj[1]
        n = x.shape[0]
        x = self.manifold.logmap0(x, c=self.c)
        x = self.dropout(x, key=key)
        if self.use_att:
            attn = self.attn(x)
            x_agg = jnp.einsum('hij,jk -> ik', attn, x) 
        else:
            # edge covolution or use sender features directly
            if self.edge_conv:
                x_s = jnp.concatenate([x[r], x[s], x[s]-x[r]], axis=-1)
                x_s = jax.vmap(self.edge_conv)(x_s)
            else:
                x_s = x[s]
            # aggregation function. use mean if weighted else use self.agg
            if isinstance(w,jnp.ndarray): 
                x_s = jnp.einsum('ij,i -> ij', x_s, w)
            #    if self.agg=='sum': 
            #        x_agg = jraph.segment_sum(x_s, r, n)
            #    else: 
            #        x_agg = jraph.segment_sum(x_s, r, n)
            if self.agg == 'multi':
                temp = jnp.array([1.])
                x_softmax = [segment_softmax(t)(x_s, r, n) for t in temp]
                x_sum = jraph.segment_sum(x_s, r, n)
                x_mean = jraph.segment_mean(x_s, r, n)
                x_var = jraph.segment_variance(x_s, r, n)
                x_agg = jnp.concatenate([x_sum, x_mean, x_var] + x_softmax, axis=-1)
                x_agg = jax.vmap(self.agg_embed)(x_agg)
            elif self.agg == 'mean':
                x_agg = jraph.segment_mean(x_s, r, n)
            elif self.agg == 'sum':
                x_agg = jraph.segment_sum(x_s, r, n)
            elif self.agg == 'softmax':
                temp = jnp.array([0.1, 1., 10., 100.])
                x_softmax = [segment_softmax(t)(x_s, r, n) for t in temp]
                x_agg = jnp.concatenate(x_softmax, axis=-1)

        x_agg = self.manifold.proj(self.manifold.expmap0(x_agg, c=self.c), c=self.c)
        return x_agg


class HypAct(eqx.Module):
    manifold: manifolds.base.Manifold
    c_in: float
    c_out: float
    act: Callable
    dropout: Callable

    def __init__(self, manifold, c_in, c_out, act, dropout_rate):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act
        self.dropout = eqx.nn.Dropout(dropout_rate) 

    def __call__(self, x, key):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        #xt = self.dropout(xt, key=key)
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

class HGCNLayer(eqx.Module):
    linear: HypLinear
    agg: HypAgg
    hyp_act: HypAct

    def __init__(self, in_features, out_features,  key, args, manifold, c_in=None, c_out=None): 
        super(HGCNLayer, self).__init__()
        c_in, c_out = args.c, args.c
        self.linear = HypLinear(in_features, out_features, manifold, key, args) 
        self.agg = HypAgg(out_features, manifold, args) 
        self.hyp_act = HypAct(manifold, c_in, c_out, act=jax.nn.gelu, dropout_rate=args.dropout)

    def __call__(self, x, adj, key, w=None):
        h = self.linear(x, key)
        h = self.agg(h, adj, key, w)
        h = self.hyp_act(h, key)
        output = h, adj
        return output


class HNNLayer(eqx.Module):
    
    linear: HypLinear
    hyp_act: HypAct

    def __init__(self, in_features, out_features, key, manifold, c_in, c_out, dropout_rate, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(in_features, out_features, key, manifold, c_in, dropout_rate, use_bias)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def __call__(self, x, key):
        h = self.linear(x, key)
        h = self.hyp_act(h, key)
        return h

