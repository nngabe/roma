import sys
from typing import List, Dict

from nn.models import models

import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy import concatenate as cat
import equinox as eqx
import equinox.nn as nn
from equinox import Module, static_field

prng = lambda i=0: jax.random.PRNGKey(i)


class neural_burgers(eqx.Module):
    
    F: eqx.Module
    v: eqx.Module
    F_max: jnp.float32
    v_max: jnp.float32
    x_dim: int

    def __init__(self, args, module='pde'):
        super(neural_burgers, self).__init__()
        self.F = getattr(models, args.decoder)(args, module='pde')
        self.v = getattr(models, args.decoder)(args, module='pde')
        self.F_max = args.F_max
        self.v_max = args.v_max
        self.x_dim = args.x_dim

    def residual(self, tx, z, u, grad, lap_x, key):
        grad_t = grad[:,0]
        grad_x = grad[:,1:]
        txz = jnp.concatenate([tx, z], axis=-1)
        F = self.F_max * jax.nn.sigmoid(self.F(txz,key))
        v = self.v_max * jax.nn.sigmoid(self.v(txz,key))

        f0 = grad_t
        f1 = -F * jnp.einsum('j,ij -> i', u, grad_x)
        f2 = v * lap_x
        res = f0 - f1 - f2
        return res, res

    def reduction(self, u):
        return jnp.sqrt(jnp.square(u).sum())

class emergent(eqx.Module):
    
    F: eqx.Module
    F_max: jnp.float32 
    x_dim: int
    lin_red: eqx.nn.Linear

    def __init__(self, args, module='pde'):
        super(emergent, self).__init__()
        self.F = getattr(models, args.decoder)(args, module='pde')
        self.F_max = args.F_max
        self.x_dim = args.x_dim 
        self.lin_red = eqx.nn.Linear(self.x_dim,1,key=prng())
        #self.lin_red = eqx.nn.MLP(self.x_dim, 1, 4 * self.x_dim, 2, key = prng())

    def residual(self, tx, z, u, grad, lap_x, key):
        grad_t = grad[:,0]
        grad_x = grad[:,1:].ravel()
        txzugl = jnp.concatenate([tx, z, u, grad_x, lap_x], axis=-1)
        F = self.F_max * self.F(txzugl,key)

        res = grad_t - F
        return res, res

    def reduction(self, u):
        return jnp.sqrt(jnp.square(u).sum())
        #return jax.nn.sigmoid(self.lin_red(u))
        

class pooling(eqx.Module):
    
    pools: Dict[int,eqx.Module]
    embed: Dict[int,eqx.Module]
    
    def __init__(self, args, module='pool'):
        super(pooling, self).__init__()
        self.pools = {}
        self.embed = {}
        for i in range(args.pool_init):
            self.pools[i] = getattr(models, args.pool)(args, module='pool')
        for i in range(args.embed_init):
            self.embed[i] = getattr(models, args.pool)(args, module='embed')

    def __getitem__(self, i):
        return self.pools[i]

    def keys(self):
        return self.pools.keys()
