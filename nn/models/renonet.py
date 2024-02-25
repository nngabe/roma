from typing import Any, Optional, Sequence, Tuple, Union, Dict, List

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx

from nn import manifolds
from nn.models import models
from aux import aux
from lib.graph_utils import dense_to_coo

prng = lambda i: jr.PRNGKey(i)

class RenONet(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    pde: eqx.Module
    pool: eqx.Module
    manifold: manifolds.base.Manifold
    c: jnp.float32 = eqx.field(static=True)
    w_data: jnp.float32 = eqx.field(static=True)
    w_pde: jnp.float32 = eqx.field(static=True)
    w_gpde: jnp.float32 = eqx.field(static=True)
    w_ent: jnp.float32 = eqx.field(static=True)
    F_max: jnp.float32 = eqx.field(static=True)
    v_max: jnp.float32 = eqx.field(static=True)
    x_dim: int
    t_dim: int
    pool_dims: List[int]
    ln_pool: List[eqx.nn.LayerNorm]
    kappa: int
    batch_size: int
    scalers: Dict[str, jnp.ndarray] = eqx.field(static=True)
    beta: np.float32
    B: jnp.ndarray = eqx.field(static=True)
    fe: bool 
    eta: jnp.float32 = eqx.field(static=True)
    euclidean: bool

    def __init__(self, args):
        super(RenONet, self).__init__()
        self.kappa = args.kappa 
        self.batch_size = args.batch_size
        self.encoder = getattr(models, args.encoder)(args, module='enc')
        self.decoder = getattr(models, args.decoder)(args, module='dec')
        self.pde = getattr(aux, args.pde)(args, module='pde', parent=self)
        self.pool = getattr(aux, 'pooling')(args, module='pool')
        self.manifold = getattr(manifolds, args.manifold)()
        self.c = args.c
        self.w_data = args.w_data
        self.w_pde = args.w_pde
        self.w_gpde = args.w_gpde
        self.w_ent = args.w_ent
        self.F_max = args.F_max
        self.v_max = args.v_max
        self.x_dim = args.x_dim
        self.t_dim = args.time_dim
        self.pool_dims = args.pool_size #[self.pool.pools[i].layers[-1].linear.linear.bias.shape[0] for i in self.pool.pools]
        self.ln_pool = [eqx.nn.LayerNorm(dim) for dim in self.pool_dims]
        self.scalers = {'t': .01}
        self.beta = args.beta 
        self.B = self.scalers['t'] * jr.normal(prng(0), (1+self.x_dim, self.t_dim//2))
        self.fe = args.fe
        self.eta = .01
        self.euclidean = True if args.manifold=='Euclidean' else False 

    def time_encode(self, t):
        if len(t.shape)>1:
            return jax.vmap(self.time_encode)(t)        

        if self.t_dim==1: 
            return t/3000.
        
        assert self.t_dim % 2 == 0
        
        Bt = t * self.B  
        t_cos, t_sin = jnp.sin(Bt), jnp.cos(Bt)
        t = jnp.concatenate([t_cos, t_sin], axis=-1).flatten()
        return t

    def coord_encode(self, tx):
        if len(tx.shape)>1:
            return jax.vmap(self.coord_encode)(tx)        

        if self.t_dim==1: 
            return tx/3000.
        
        assert self.t_dim % 2 == 0
        
        Btx= jnp.einsum('i,ij -> ij', tx, self.B).reshape(-1,1)
        tx_cos, tx_sin = jnp.sin(Btx), jnp.cos(Btx)
        t = jnp.concatenate([tx_cos, tx_sin], axis=-1).flatten()
        return t


    def exp(self, x):
        if self.euclidean:
            return x
        x = self.manifold.proj_tan0(x, c=self.c)
        x = self.manifold.expmap0(x, c=self.c)
        x = self.manifold.proj(x, c=self.c)
        return x

    def log(self, y):
        if self.euclidean:
            return y
        y = self.manifold.logmap0(y, self.c)
        y = y * jnp.sqrt(self.c) * 1.4763057
        return y

    def encode(self, x, adj, key):
        z = self.encoder(x, adj, key)
        return z

    def embed_pool(self, z, adj, w, i, key):
        keys = jr.split(key,4)
        z0 = z[:,:self.kappa]
        zi = z[:,self.kappa:]
        ze = self.pool.embed[i](zi, adj, keys[0], w)
        s = self.pool[i](z, adj, keys[1], w)
        z = jnp.concatenate([z0,ze], axis=-1)
        return z,s

    def renorm(self, x, adj, y, key, mode='train'):
        w = None
        loss_ent = 0.
        S = {}
        A = {}
        z_r = x
        y_r = y
        A[0] = jnp.zeros(x.shape[:1]*2).at[adj[0],adj[1]].set(1.)
        for i in self.pool.keys():
            z,s = self.embed_pool(x, adj, w, i, key)
            s = jax.vmap(self.ln_pool[i])(self.log(s))
            S[i] = jax.nn.softmax(s, axis=0)
            m,n = S[i].shape
            x = jnp.einsum('ij,ik -> jk', S[i], z) * (n/m)
            y = jnp.einsum('ij,ki -> kj', S[i], y) * (n/m)
            A[i+1] = jnp.einsum('ji,jk,kl -> il', S[i], A[i], S[i])
            adj, w = dense_to_coo(A[i])
            z_r = jnp.concatenate([z_r, x], axis=0)
            y_r = jnp.concatenate([y_r, y], axis=-1)
            loss_ent += jax.scipy.special.entr(S[i]).mean()
            key = jr.split(key)[0]

        if mode == 'train':
            return z_r, y_r, loss_ent, None, None
        elif mode == 'fwd' or mode == 'report':
            return z_r, y_r, loss_ent, S, A
        elif mode == 'fwd_no_renorm' or mode == 'test':
            return x, y, loss_ent, None, None
        else:
            return z_r, y_r, loss_ent, S, A

    def decode(self, tx, z, key):
        #t = self.time_encode(tx[:1])
        #tx = jnp.concatenate([t,tx[1:]], axis=-1)
        tx = self.coord_encode(tx) 
        if hasattr(self.decoder, 'branch'):
            p_dim, x_dim = self.decoder.p_dim, self.decoder.x_dim
            b_dim = p_dim * x_dim
            branch,z = z[:b_dim], z[b_dim:]
            txz = jnp.concatenate([tx, z], axis=-1)
            trunk = self.decoder.trunk(txz, key)
            branch = branch.reshape(p_dim, x_dim)
            trunk = trunk.reshape(p_dim)
            u = jnp.einsum('ij,i -> j', branch, trunk) / p_dim
        else:    
            txz = jnp.concatenate([tx,z], axis=-1)
            u = self.decoder(txz, key=key)
        return u, (u, txz)

    def val_grad(self, tx, z, key):
        f = lambda tx: self.decode(tx, z, key)
        grad, val = jax.jacfwd(f, has_aux=True)(tx)
        return grad, (grad, val)

    def val_grad_lap(self, tx, z, key):
        vg = lambda tx,z: self.val_grad(tx, z, key)
        grad2, (grad, (u, txz)) = jax.jacfwd(vg, has_aux=True)(tx,z)
        hess = jax.vmap(jnp.diag)(grad2)
        lap_x = hess[:,1:].sum(1)
        return (u.flatten(), txz), grad, lap_x

    def pde_res_grad(self, tx, z, u, grad, lap_x, key):
        gpde, res = jax.jacfwd(self.pde.residual, has_aux=True)(tx, z, u, grad, lap_x, key)
        return res, gpde

    def branch(self, z, key):
        keys = jr.split(key, 2)
        if hasattr(self.decoder, 'branch'):
            b,z = z[:,:self.kappa],z[:,self.kappa:]
            b = self.decoder.func_space(b, key)
            b_dec = self.decoder.branch(b, keys[0])
            b_pde = self.decoder.branch(b, keys[1])
            z_dec = jnp.concatenate([b_dec,z], axis=-1)
            z_pde = jnp.concatenate([b_pde,z], axis=-1)
            return z_dec, z_pde
        else:
            return z,z

    def forward(self, x0, adj, t, y, key, mode='train'):
        keys = jr.split(key,5)
        z = self.encode(x0, adj, key=keys[0])
        z, y, loss_ent, S, A = self.renorm(z, adj, y, key=keys[1], mode=mode) 
        x = jnp.zeros((z.shape[0], self.x_dim))
        t_ = t * jnp.ones((z.shape[0],1))
        tx = jnp.concatenate([t_,x], axis=-1)
        loss_data = loss_pde = loss_gpde = 0.
        vgl = lambda tx,z: self.val_grad_lap(tx, z, keys[i])
        pde_rg = lambda tx, z, u, grad, lap_x: self.pde_res_grad(tx, z, u, grad, lap_x, keys[i])
        for i in range(y.shape[0]):
            z_dec, z_pde = self.branch(z, keys[2])
            (u, txz), grad, lap_x = jax.vmap(vgl)(tx, z_dec)
            red = jax.vmap(self.pde.reduction)(u) 
            loss_data += jnp.square(red - y[i]).mean()
            resid, gpde = jax.vmap(pde_rg)(tx, z_pde, u, grad, lap_x)
            loss_pde += jnp.square(resid).mean()
            loss_gpde += jnp.square(gpde).mean()
            t_ += 1 
            tx = jnp.concatenate([t_,x], axis=-1)
            x = jnp.concatenate([z[:,1:self.kappa], red.reshape(-1,1)], axis=-1)
            z = z.at[:,:self.kappa].set(x)

        if mode=='train':
            loss = self.w_data * loss_data + self.w_pde * loss_pde + self.w_gpde * loss_gpde + self.w_ent * loss_ent
            return loss        
        if mode=='report' or mode=='test': 
            loss_data = jnp.square(red - y[0])
            loss_data = loss_data[:self.batch_size].mean(), loss_data[self.batch_size:].mean()
            return jnp.array([loss_data[0], loss_data[1], loss_pde, loss_gpde, loss_ent])
        elif mode=='inference': 
            return u, red, z, y, grad, S, A 
    
    def slaw_update(self, loss, state):
        assert state != None
        a,b = state['a'], state['b']
        a = self.beta * a + (1. - self.beta) * loss**2
        b = self.beta * b + (1. - self.beta) * loss
        s = jnp.sqrt(a - b**2)
        w = loss.shape[0] / s / (1./s).sum()
        w = w/w.min()
        loss = w * loss
        loss = self.w_data * loss[0] + self.w_pde * loss[1] + self.w_gpde * loss[2] + self.w_ent * loss[3] 
        state['a'], state['b'] = a,b
        return loss, state 
  
    def loss_vmap(self, xb, adj, tb, yb, key=prng(0), mode='train', state=None):
        n = xb.shape[0]
        kb = jr.split(key, n) 
        loss_vec = lambda x,t,y,k: self.forward(x, adj, t, y, k, mode=mode)
        loss = jax.vmap(loss_vec)(xb, tb, yb, kb)
        if mode=='slaw':
           loss, state = self.slaw_update(loss, state)
           return loss.mean(), state
        else:
            return loss.mean(0), state

    def loss_scan(self, xb, adj, tb, yb, key=prng(0), mode='train', state=None):
        n = xb.shape[0] # batch size
        kb = jr.split(key,n) 
        body_fun = lambda i,val: val + self.forward(xb[i], adj, tb[i], yb[i], kb[i], mode=mode)
        loss = 0. if mode=='train' else jnp.zeros(5)
        loss = jax.lax.fori_loop(0, n, body_fun, loss)
        if mode=='slaw':
            loss, state = self.slaw_update(loss, state)
            return loss.mean(), state
        else:
            return loss, state

    def div(self, grad_x):
        return jax.vmap(jnp.trace)(grad_x)

    def curl(self, grad_x):
        omega = lambda grad_x: jnp.abs(grad_x - grad_x.T).sum()/2.
        return jax.vmap(omega)(grad_x)

    def enstrophy(self, grad_x):
        return jax.vmap(jnp.sum)(jnp.abs(grad_x))


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def loss_train(model, xb, adj, tb, yb, key=prng(0), mode='train', state=None):
    return model.loss_vmap(xb, adj, tb, yb, key=key, mode=mode, state=state)

@eqx.filter_jit
def loss_report(model, xb, adj, tb, yb):
    return model.loss_vmap(xb, adj, tb, yb, mode='report')

# updating model parameters and optimizer state
@eqx.filter_jit
def make_step(grads, model, opt_state, optim):
    updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model,updates)
    return model, opt_state
