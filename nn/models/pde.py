import sys
from typing import List, Dict, Callable

from nn.models import models

import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy import concatenate as cat
import equinox as eqx
import equinox.nn as nn
from equinox import Module, static_field

prng = lambda i=0: jax.random.PRNGKey(i)

class burgers(eqx.Module):
    
    F: eqx.Module
    v: eqx.Module
    x_dim: int
    p_dim: int
    lin_red: eqx.nn.Linear
    time_encode: Callable
    coord_encode: Callable
    nonlinear: bool

    def __init__(self, args, module='pde', parent=None):
        super(burgers, self).__init__()
        self.F = getattr(models, args.decoder)(args, module='pde')
        self.v = getattr(models, args.decoder)(args, module='pde')
        self.x_dim = args.x_dim
        self.p_dim = args.p_basis
        self.lin_red = eqx.nn.Linear(self.x_dim,1,key=prng())
        self.time_encode = lambda x: parent.time_encode(x)
        self.coord_encode = lambda x: parent.coord_encode(x)
        self.nonlinear = args.nonlinear_pde
    
    def branch(b, key):
        b1 = self.F.branch(b, key)
        b2 = self.v.branch(b, key)
        b = jnp.concatenate([b1,b2], axis=-1)
        return b

    def residual(self, tx, z, u, grad, lap_x, key):
        grad_t = grad[:,0]
        grad_x = grad[:,1:].ravel()
        tx = self.coord_encode(tx)
        if hasattr(self.F, 'branch'):
            p_dim, x_dim = self.p_dim, self.x_dim
            b_dim = p_dim * x_dim
            b = z[:b_dim] 
            z = z if self.nonlinear else z[b_dim:]            
            txzugl = jnp.concatenate([tx, z, u, grad_x, lap_x], axis=-1)
            trunk_F = self.F.trunk(txzugl, key)
            trunk_v = self.v.trunk(txzugl, key)
            if self.nonlinear:
                F = trunk_F
                v = trunk_v
            else:
                trunk_F = trunk_F.reshape(p_dim)
                trunk_v = trunk_v.reshape(p_dim)
                branch = b.reshape(p_dim, x_dim)
                F = jnp.einsum('ij,i -> ', branch, trunk_F) / b_dim
                v = jnp.einsum('ij,i -> ', branch, trunk_v) / b_dim
        else:
            txzugl = jnp.concatenate([tx, z, u, grad_x, lap_x], axis=-1)
            F = self.F(txzugl, key)
            v = self.v(txzugl, key)

        f0 = grad_t
        f1 = -F * jnp.einsum('j,ij -> i', u, grad_x.reshape(x_dim,x_dim))
        f2 = v * lap_x

        res = f0 - f1 - f2
        return res, res

    def reduction(self, u):
        return jnp.sqrt(jnp.square(u).sum())
 

class emergent(eqx.Module):
    
    F: eqx.Module
    F_max: jnp.float32 
    x_dim: int
    p_dim: int
    lin_red: eqx.nn.Linear
    time_encode: Callable
    coord_encode: Callable
    nonlinear: bool

    def __init__(self, args, module='pde', parent=None):
        super(emergent, self).__init__()
        self.F = getattr(models, args.decoder)(args, module='pde', shared=args.shared_branch)
        self.F_max = args.F_max
        self.x_dim = args.x_dim
        self.p_dim = args.p_basis
        self.lin_red = eqx.nn.Linear(self.x_dim,1,key=prng())
        self.time_encode = lambda x: parent.time_encode(x)
        self.coord_encode = lambda x: parent.coord_encode(x)
        self.nonlinear = args.nonlinear_pde

    def branch(b, key):
        b = self.F.branch(b, key)
        return b

    def residual(self, tx, z, u, grad, lap_x, key):
        grad_t = grad[:,0]
        grad_x = grad[:,1:].ravel()
        tx = self.coord_encode(tx)
        if hasattr(self.F, 'trunk'):
            p_dim, x_dim = self.p_dim, self.x_dim
            b_dim = p_dim * x_dim
            b = z[:b_dim] 
            z = z if self.nonlinear else z[b_dim:]            
            txzugl = jnp.concatenate([tx, z, u, grad_x, lap_x], axis=-1)
            trunk = self.F.trunk(txzugl, key)
            if self.nonlinear:
                F = trunk 
            else:
                branch = b.reshape(p_dim, x_dim)
                trunk = trunk.reshape(p_dim)
                F = jnp.einsum('ij,i -> ', branch, trunk) / b_dim
        else:
            txzugl = jnp.concatenate([tx, z, u, grad_x, lap_x], axis=-1)
            F = self.F(txzugl, key)

        res = grad_t - F
        return res, res

    def reduction(self, u):
        return jnp.sqrt(jnp.square(u).sum())
