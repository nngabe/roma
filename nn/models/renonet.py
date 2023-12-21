from typing import Any, Optional, Sequence, Tuple, Union, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from nn import manifolds
from nn.models import models
from aux import aux
from lib.graph_utils import dense_to_coo

prng = lambda i: jax.random.PRNGKey(i)

class RenONet(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    pde: eqx.Module
    pool: eqx.Module
    manifold: manifolds.base.Manifold
    c: jnp.float32
    w_data: jnp.float32
    w_pde: jnp.float32
    w_gpde: jnp.float32
    w_ent: jnp.float32
    F_max: jnp.float32
    v_max: jnp.float32
    x_dim: int
    t_dim: int
    pool_dims: List[int]
    kappa: int
    batch_size: int
    scalers: Dict[str, jnp.ndarray]
    k_lin: Tuple[int,int]
    k_log: Tuple[int,int]
    beta: np.float32
    B: jnp.ndarray
    fe: bool
    eta: np.float32

    def __init__(self, args):
        super(RenONet, self).__init__()
        self.kappa = 0 if args.res else args.kappa
        self.batch_size = args.batch_size
        self.encoder = getattr(models, args.encoder)(args, module='enc')
        self.decoder = getattr(models, args.decoder)(args, module='dec')
        self.pde = getattr(aux, args.pde)(args, module='pde')
        self.pool = aux.pooling(args, module='pool')
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
        self.pool_dims = [self.pool.pools[i].layers[-1].linear.linear.bias.shape[0] for i in self.pool.pools]
        self.scalers = {'t_lin': 10. ** jnp.arange(2, 2*self.t_dim, 1, dtype=jnp.float32),
                        't_log': 10. ** jnp.arange(-2, 2*self.t_dim, 1, dtype=jnp.float32),
                        't_cos': 10. **jnp.arange(-4,2*self.t_dim, 1, dtype=jnp.float32),
                       }
        self.k_lin = self.t_dim//2
        self.k_log = self.t_dim - self.k_lin
        self.beta = args.beta 
        self.B = 1. * jax.random.normal(prng(0), (args.kappa, args.f_dim))
        self.fe = args.fe
        self.eta = .01

    def time_encode(self, t):
        if self.t_dim==1: 
            return t/3000.#jnp.log( t * 100. + 1. )
        t_lin = t / jnp.clip(self.scalers['t_lin'], 1e-7)
        t_log = t / jnp.clip(self.scalers['t_log'], 1e-7)
        t_cos = t / jnp.clip(self.scalers['t_cos'], 1e-7)
        t_lin = t_lin[:self.k_lin]
        t_log = jnp.log( t_log[:self.k_log] + 1. )
        t_cos = t_cos[:self.k_lin]
        t = jnp.concatenate([t_lin, t_log])
        return t

    def exp(self, x):
        x = self.manifold.proj_tan0(x, c=self.c)
        x = self.manifold.expmap0(x, c=self.c)
        x = self.manifold.proj(x, c=self.c)
        return x

    def log(self, y):
        y = self.manifold.logmap0(y, self.c)
        y = y * jnp.sqrt(self.c) * 1.4763057
        return y

    def fourier_enc(self, x, key=prng(1)):
        x = x * jnp.exp(self.eta * jax.random.normal(key, x.shape))
        Bx = jnp.einsum('ij,kj -> ik', x, self.B)
        y = jnp.concatenate([jnp.cos(Bx), jnp.sin(Bx)], axis=-1)
        return y

    def adv_enc(self, x, key=prng(0)):
        key = jax.random.split(key,2)
        y = x * jnp.exp(self.eta * jax.random.normal(key[0], x.shape))
        x = x * jnp.exp(self.eta * jax.random.normal(key[1], x.shape))
        return jnp.concatenate([x,y], axis=-1)
 
    def encode(self, x0, adj, t, eps=0.,key=prng(0)):
        x = self.fourier_enc(x0, key=key) if self.fe else x0
        z_x = self.encoder(x, adj, key=key)
        z = z_x[:,:-self.x_dim]
        x = eps * z_x[:,-self.x_dim:]
        t = self.time_encode(t)
        t = t*jnp.ones((x.shape[0],1))
        tx = jnp.concatenate([t, x], axis=-1)
        z = self.log(z)
        return tx, z

    def embed_pool(self, z, adj, w, i, key):
        keys = jax.random.split(key,4)
        z0 = z[:,:self.kappa]
        zi = z[:,self.kappa:]
        ze = self.pool.embed[i](zi, adj, w, keys[0])
        s = self.pool[i](z, adj, w, keys[1])
        z = jnp.concatenate([z0,ze], axis=-1)
        return z,s

    def decode(self, tx, z, key=prng(0)):
        txz = jnp.concatenate([tx,z], axis=-1)
        u = self.decoder(txz, key=key)
        return u, (u, txz)

    def renorm(self, t, x, adj, y, key=prng(0), mode='train'):
        w = None
        loss_ent = 0.
        S = {}
        A = {}
        z_r = x
        y_r = y
        A[0] = jnp.zeros(x.shape[:1]*2).at[adj[0],adj[1]].set(1.)
        for i in self.pool.keys():
            key = jax.random.split(key)[0]
            v = jnp.ones((x.shape[0], 1))
            tx = jnp.concatenate([t*v, x], axis=-1)
            z,s = self.embed_pool(x, adj, w, i, key)
            S[i] = jax.nn.softmax(self.log(s) * 1e+0, axis=0)
            m,n = S[i].shape
            x = jnp.einsum('ij,ik -> jk', S[i], z) * (n/m)
            y = jnp.einsum('ij,ki -> kj', S[i], y) * (n/m)
            A[i+1] = jnp.einsum('ji,jk,kl -> il', S[i], A[i], S[i])
            adj, w = dense_to_coo(A[i])
            z_r = jnp.concatenate([z_r, x], axis=0)
            y_r = jnp.concatenate([y_r, y], axis=-1)
            loss_ent += jax.scipy.special.entr(S[i]).mean()

        if mode == 'train':
            return z_r, y_r, loss_ent, None, None
        elif mode == 'fwd' or mode == 'report':
            return z_r, y_r, loss_ent, S, A
        elif mode == 'fwd_no_renorm' or mode == 'test':
            return x, y, loss_ent, None, None
        else:
            return z_r, y_r, loss_ent, S, A

    def val_grad(self, tx, z, key):
        f = lambda tx: self.decode(tx,z,key)
        grad, val = jax.jacfwd(f, has_aux=True)(tx)
        return grad, (grad, val)

    def val_grad_lap(self, tx, z, key):
        vg = lambda tx,z: self.val_grad(tx,z,key)
        grad2, (grad,(u,txz)) = jax.jacfwd(vg, has_aux=True)(tx,z)
        hess = jax.vmap(jnp.diag)(grad2)
        lap_x = hess[:,1:].sum(1)
        return (u.flatten(), txz), grad, lap_x

    def pde_res_grad(self, tx, z, u, grad, lap_x, key):
        gpde, res = jax.jacfwd(self.pde.residual, has_aux=True)(tx, z, u, grad, lap_x, key)
        return res, gpde

    def pde_bc(self, ):
        
        return 0.

    def forward(self, x0, adj, t, y, key, mode='train'):
        keys = jax.random.split(key,5)
        tx,z = self.encode(x0, adj, t, key=keys[0])
        t = tx[0,0]
        z, y, loss_ent, S, A = self.renorm(t, z, adj, y, key=keys[1], mode=mode)
        tx = tx[0] * jnp.ones((z.shape[0],1))
        loss_data = loss_pde = loss_gpde = 0.
        for i in range(y.shape[0]):
            vgl = lambda tx,z: self.val_grad_lap(tx, z, keys[i])
            (u, txz), grad, lap_x = jax.vmap(vgl)(tx, z)
            red = jax.vmap(self.pde.reduction)(u) 
            loss_data += jnp.square(red - y[i]).mean()
            pde_rg = lambda tx, z, u, grad, lap_x: self.pde_res_grad(tx, z, u, grad, lap_x, keys[i])
            resid, gpde = jax.vmap(pde_rg)(tx, z, u, grad, lap_x)
            loss_pde += jnp.square(resid).mean()
            loss_gpde += jnp.square(gpde).mean()      
            tx = tx.at[:,0].set(tx[:,0]+1.)
            x = jnp.concatenate([z[:,1:self.kappa], red.reshape(-1,1)], axis=-1)
            z = z.at[:,:self.kappa].set(x)

        if mode=='train':
            loss = self.w_data * loss_data + self.w_pde * loss_pde + self.w_gpde * loss_gpde + self.w_ent * loss_ent
            return loss        
        if mode=='report' or mode=='test': #reporting
            loss_data = jnp.square(red - y[0])
            loss_data = loss_data[:self.batch_size].mean(), loss_data[self.batch_size:].mean()
            return jnp.array([loss_data[0], loss_data[1], loss_pde, loss_gpde, loss_ent])
        elif mode=='fwd': 
            return u, red, y, grad, A, S 
    
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
        kb = jax.random.split(key, n) 
        loss_vec = lambda x,t,y,k: self.forward(x, adj, t, y, k, mode=mode)
        loss = jax.vmap(loss_vec)(xb, tb, yb, kb)
        if mode=='slaw':
           loss, state = self.slaw_update(loss, state)
           return loss.sum()/n, state
        else:
            return loss/n, state

    def loss_scan(self, xb, adj, tb, yb, key=prng(0), mode='train', state=None):
        n = xb.shape[0] # batch size
        kb = jax.random.split(key,n) 
        body_fun = lambda i,val: val + self.forward(xb[i], adj, tb[i], yb[i], kb[i], mode=mode)
        loss = 0. if mode=='train' else jnp.zeros(5)
        loss = jax.lax.fori_loop(0, n, body_fun, loss)
        if mode=='slaw':
            loss, state = self.slaw_update(loss, state)
            return loss.sum()/n, state
        else:
            return loss/n, state

    def div(self, grad_x):
        return jax.vmap(jnp.trace)(grad_x)

    def curl(self, grad_x):
        omega = lambda grad_x: jnp.abs(grad_x - grad_x.T).sum()/2.
        return jax.vmap(omega)(grad_x)

    def enstrophy(self, grad_x):
        return jax.vmap(jnp.sum)(jnp.abs(grad_x))


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def loss_scan(model, xb, adj, tb, yb, key=prng(0), mode='train', state=None):
    return model.loss_scan(xb, adj, tb, yb, key=key, mode=mode, state=state)

@eqx.filter_jit
def loss_terms(model, xb, adj, tb, yb):
    return model.loss_scan(xb, adj, tb, yb, mode='report')

# updating model parameters and optimizer state
@eqx.filter_jit
def make_step(grads, model, opt_state, optim):
    updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model,updates)
    return model, opt_state
