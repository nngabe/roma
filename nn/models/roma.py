from typing import Any, Optional, Sequence, Tuple, Union, Dict, List, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx

from nn import manifolds
from nn.models import models
from nn.models import pde
from nn.models import pooling
from lib.graph_utils import dense_to_coo

prng = lambda i=0: jr.PRNGKey(i)

class ROMA(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    pde: eqx.Module
    pool: eqx.Module
    manifold: manifolds.base.Manifold
    c: jnp.float32 = eqx.field(static=True)
    w_data: jnp.float32 = eqx.field(static=True)
    w_pde: jnp.float32 = eqx.field(static=True)
    w_gpde: jnp.float32 = eqx.field(static=True)
    w_ms: jnp.float32 = eqx.field(static=True)
    w_pool: jnp.ndarray = eqx.field(static=True)
    x_dim: int
    coord_dim: int
    pool_dims: List[int]
    ln_pool: List[eqx.nn.LayerNorm]
    kappa: int
    batch_size: int
    scalers: np.ndarray = eqx.field(static=True)
    beta: np.float32 = eqx.field(static=True)
    B: jnp.ndarray = eqx.field(static=True)
    embed_pe: eqx.nn.Linear
    embed_s: eqx.nn.Linear
    euclidean: bool
    nonlinear: bool
    eps: jnp.float32 = eqx.field(static=True)
    eps_loss: jnp.float32 = eqx.field(static=True)
    w_l: jnp.float32 = eqx.field(static=True)
    func_pos_emb: bool
    entr: Dict[int,Callable] = eqx.field(static=True)
    alpha: jnp.float32 = eqx.field(static=True)

    def __init__(self, args):
        super(ROMA, self).__init__()
        self.kappa = args.kappa 
        self.batch_size = args.batch_size
        self.encoder = getattr(models, args.encoder)(args, module='enc')
        self.decoder = getattr(models, args.decoder)(args, module='dec')
        self.pde = getattr(pde, args.pde)(args, module='pde', parent=self)
        self.pool = getattr(pooling, 'pooling')(args, module='pool')
        self.manifold = getattr(manifolds, args.manifold)()
        self.c = args.c
        self.w_data = args.w_data
        self.w_pde = args.w_pde
        self.w_gpde = args.w_gpde
        self.w_ms = args.w_ms
        self.w_pool = args.w_pool
        self.x_dim = args.x_dim
        self.coord_dim = args.coord_dim
        self.pool_dims = args.pool_size 
        self.ln_pool = [eqx.nn.LayerNorm(dim) for dim in self.pool_dims]
        self.scalers = np.concatenate([[args.t_var], self.x_dim * [args.x_var]], axis=0).reshape(-1,1)
        self.beta = args.beta 
        self.B = self.scalers * jr.normal(prng(), (1 + self.x_dim, args.coord_dim//2))
        self.embed_s = eqx.nn.Linear(args.kappa, args.pe_embed_dim*2, key = prng(1))
        self.embed_pe = eqx.nn.Linear(args.pe_size, args.pe_embed_dim*2, key = prng(2)) 
        self.euclidean = True if args.manifold=='Euclidean' else False 
        self.nonlinear = args.nonlinear
        self.eps = 1e-15
        self.eps_loss = 1e-6
        self.w_l = 1.
        self.func_pos_emb = args.func_pos_emb
        self.entr = {}
        self.alpha = 2.
        for i,d in enumerate(self.pool_dims):
            zeta = 2. / jnp.log(d) if args.zeta<1e-6 else args.zeta
            _clip = lambda x: jax.numpy.clip(x, 1e-10, 1.)
            self.entr[i] = lambda x: (-1. * jnp.e * _clip(x)**zeta * jnp.log( _clip(x)**zeta + 1e-10))**self.alpha
            

    def coord_encode(self, tx):
        if len(tx.shape)>1:
            return jax.vmap(self.coord_encode)(tx)        

        if self.coord_dim==1: 
            return tx * 1e-3
        
        assert self.coord_dim % 2 == 0
        
        #Btx= jnp.einsum('i,ij -> ij', tx, self.B).reshape(-1,1)
        Btx= jnp.einsum('i,ij -> j', tx, self.B).reshape(-1,1)
        tx_cos, tx_sin = jnp.sin(Btx), jnp.cos(Btx)
        tx = jnp.concatenate([tx_cos, tx_sin], axis=-1).flatten()
        return tx

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
        return y

    def encode(self, x, adj, key):
        s,pe = x[:,:self.kappa], x[:,self.kappa:]
        pe = jax.vmap(self.embed_pe)(pe)
        x = jnp.concatenate([s,pe], axis=-1)
        x = self.encoder(x, adj, key)
        return x

    def embed_pool(self, z, adj, w, i, key):
        keys = jr.split(key,4)
        s = z[:,:self.kappa]
        zi = z[:,self.kappa:]
        ze = self.pool.embed[i](zi, adj, keys[0], w)
        S = self.pool[i](zi, adj, keys[1], w)
        z = jnp.concatenate([s,ze], axis=-1)
        return z,S

    def renorm(self, x, adj, y, key, mode='train'):
        w = None
        loss_ent = 0.
        S = {}
        A = {}
        z_r = x
        y_r = y
        A[0] = jnp.zeros(x.shape[:1]*2).at[adj[0],adj[1]].set(1.)
        loss_pool = jnp.array([0.,0.,0.])
        for i in self.pool.keys():
            z,s = self.embed_pool(x, adj, w, i, key)
            s = self.log(s)
            s = jax.vmap(self.ln_pool[i])(s)
            #S[i] = jax.nn.softmax(s, axis=0)
            #sr = jnp.sqrt(S[i])
            sr = s/jnp.linalg.norm(s, axis=0, keepdims=True)
            S[i] = sr**2
            m,n = S[i].shape
            x = jnp.einsum('ij,ik -> jk', S[i], z) * (n/m)
            y = jnp.einsum('ij,ki -> kj', S[i], y) * (n/m)
            A[i+1] = jnp.einsum('ji,jk,kl -> il', S[i], A[i], S[i])
            adj, w = dense_to_coo(A[i])
            z_r = jnp.concatenate([z_r, x], axis=0)
            y_r = jnp.concatenate([y_r, y], axis=-1)
            _entr = lambda x: (jax.scipy.special.entr(x)*jnp.e)
            loss_pool = loss_pool.at[0].add( self.w_pool[0] * _entr(S[i]).mean())
            loss_pool = loss_pool.at[1].add( self.w_pool[1] * self.entr[0](A[i+1]).mean())
            loss_pool = loss_pool.at[2].add( self.w_pool[2] * jnp.square(A[i] - jnp.einsum('ij,kj -> ik', sr, sr)).mean())
            key = jr.split(key)[0]

        if mode == 'train':
            return z_r, y_r, loss_pool.sum(), None, None
        elif mode == 'report':
            return z_r, y_r, loss_pool, S, A
        else:
            return z_r, y_r, loss_pool, S, A

    def decode(self, tx, z, key):
        tx = self.coord_encode(tx) 
        if hasattr(self.decoder, 'branch'):
            p_dim, x_dim = self.decoder.p_dim, self.decoder.x_dim
            b_dim = p_dim * x_dim
            b = z[:b_dim]
            z = z if self.nonlinear else z[b_dim:]
            txz = jnp.concatenate([tx, z], axis=-1)
            if self.nonlinear:
                u = self.decoder.trunk(txz, key) / x_dim
            elif not self.nonlinear:
                trunk = self.decoder.trunk(txz, key)
                branch = b.reshape(p_dim, x_dim)
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
        keys = jr.split(key, 5)
        if hasattr(self.decoder, 'branch'):
            b,z = z[:,:self.kappa],z[:,self.kappa:]
            b = self.decoder.func_space(b, keys[0])
            pe = jax.vmap(self.decoder.func_pe)(z) if self.func_pos_emb else None
            b_dec = self.decoder.branch(b, keys[2], pe=pe)
            b_pde = self.decoder.branch(b, keys[3], pe=pe)
            #b_pde = self.pde.branch(b, keys[2])
            z_dec = jnp.concatenate([b_dec,z], axis=-1)
            z_pde = jnp.concatenate([b_pde,z], axis=-1)
            return z_dec, z_pde
        else:
            return z,z

    def sMAPE(self, y, yh):
        loss_fn =  lambda y, yh: jnp.abs(y-yh)/(jnp.abs(y) + jnp.abs(yh) + 1e-16)
        if len(y.shape)>1:
            res = jax.vmap(loss_fn)(y,yh)
        else:
            res = loss_fn(y,yh) 
        return res.mean() 

    def sMSPE(self, y, yh):
        loss_fn =  lambda y, yh: jnp.abs(y-yh)/(jnp.abs(y) + jnp.abs(yh) + 1e-4)
        if len(y.shape)>1:
            res = jax.vmap(loss_fn)(y,yh)
        else:
            res = loss_fn(y,yh) 
        return jnp.square(res).mean() 

    def forward(self, sb, adj, t, y, key, mode='train'):
        keys = jr.split(key,5)
        z = self.encode(sb, adj, key=keys[0])
        z, y, loss_pool, S, A = self.renorm(z, adj, y, key=keys[1], mode=mode) 
        x = jnp.zeros((z.shape[0], self.x_dim))
        t_ = t * jnp.ones((z.shape[0],1))
        tx = jnp.concatenate([t_,x], axis=-1)
        loss_data = loss_pde = loss_gpde = 0.
        vgl = lambda tx,z: self.val_grad_lap(tx, z, keys[i])
        pde_rg = lambda tx, z, u, grad, lap_x: self.pde_res_grad(tx, z, u, grad, lap_x, keys[i])
        
        m = y.shape[0]
        for i in range(m):
            z_dec, z_pde = self.branch(z, keys[2])
            (u, txz), grad, lap_x = jax.vmap(vgl)(tx, z_dec)
            red = jax.vmap(self.pde.reduction)(u) 
            loss_data += jax.vmap(self.sMSPE)(red,y[i])
            resid, gpde = jax.vmap(pde_rg)(tx, z_pde, u, grad, lap_x)
            loss_pde += jnp.square(resid[:self.batch_size]).mean()
            loss_gpde += jnp.square(gpde[:self.batch_size]).mean()
            if i < (m - 1): 
                t_ += 1.
                tx = jnp.concatenate([t_,x], axis=-1)
                x = jnp.concatenate([z[:,1:self.kappa], red.reshape(-1,1)], axis=-1)
                z = z.at[:,:self.kappa].set(x)

        #mask = jnp.abs(z[:,:10].sum(1)) > self.eps
        #loss_data = loss_data.at[:].set(loss_data * mask)
        
        if mode=='train':
            if loss_data.shape[0] != self.batch_size:
                rescaling = 0. #jnp.sqrt( self.batch_size / (loss_data.shape[0] - self.batch_size)) 
                loss_data = loss_data.at[self.batch_size:].multiply(rescaling) 
            loss_data = loss_data.mean()
            loss = self.w_data * loss_data + self.w_pde * loss_pde + self.w_gpde * loss_gpde + self.w_ms * loss_pool
            return loss        
        if mode=='report': 
            loss_data = jax.vmap(self.sMSPE)(red,y[i])
            loss_data = loss_data[:self.batch_size].mean(), loss_data[self.batch_size:].mean()
            return jnp.array([loss_data[0], loss_data[1], loss_pde, loss_gpde, loss_pool[0]/self.w_pool[0], loss_pool[1]/self.w_pool[1], loss_pool[2]/self.w_pool[2]])
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
        loss = self.w_data * loss[0] + self.w_pde * loss[1] + self.w_gpde * loss[2] + self.w_ms * loss[3] 
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
