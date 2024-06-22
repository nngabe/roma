import os
import time
import math
import glob
import pickle
import jax
from jax.tree_util import tree_map
import equinox as eqx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as onp
import jax.numpy as jnp
from argparse import Namespace

from nn.models.roma import ROMA

def clip_tree(tree, max_norm, spec=eqx.is_inexact_array):
  clip_fn = lambda x: jnp.clip(x, -max_norm, max_norm) if spec(x) else x
  return tree_map(clip_fn, tree)

def trunc_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
  out, in_ = weight.shape
  stddev = math.sqrt(1 / in_)
  return stddev * jax.random.truncated_normal(key, lower=-2, upper=2, shape=weight.shape)

def ortho_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
  out_dim, in_dim = weight.shape
  if in_dim >= out_dim:
    return jax.random.orthogonal(key, in_dim)[:,:out_dim].T
  else:
    return jnp.power(out_dim/in_dim, 1/2) * jax.random.orthogonal(key, out_dim)[:in_dim,:].T
    #return jnp.power(1/in_dim, 1/2) * jax.random.truncated_normal(key, lower=-2, upper=2, shape=weight.shape)

def init_ortho(model, key):
  is_linear = lambda x: isinstance(x, eqx.nn.Linear)
  is_bias = lambda x: x.bias!=None if is_linear(x) else False
  get_weights = lambda m: [x.weight  for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
  get_biases = lambda m: [x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_bias) if is_bias(x)]
  weights = get_weights(model)
  biases = get_biases(model)
  new_weights = [ortho_init(weight, subkey) for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
  new_biases = [jnp.zeros(b.shape) for b in biases]
  model = eqx.tree_at(get_weights, model, new_weights)
  model = eqx.tree_at(get_biases, model, new_biases)
  return model

def init_he(model, key):
  is_linear = lambda x: isinstance(x, eqx.nn.Linear)
  is_bias = lambda x: x.bias!=None if is_linear(x) else False
  get_weights = lambda m: [x.weight  for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
  get_biases = lambda m: [x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_bias) if is_bias(x)]
  weights = get_weights(model)
  biases = get_biases(model)
  new_weights = [trunc_init(weight, subkey) for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
  new_biases = [jnp.zeros(b.shape) for b in biases]
  model = eqx.tree_at(get_weights, model, new_weights)
  model = eqx.tree_at(get_biases, model, new_biases)
  return model

def save_model(model, log, path='../eqx_models/', stamp=None):
    if not os.path.exists(path): os.mkdir(path)
    stamp = stamp if stamp else str(int(time.time())) 
    with open(path + f'log_{stamp}.pkl','wb') as f: 
        pickle.dump(log,f)
    eqx.tree_serialise_leaves(path + f'/renonet_{stamp}.eqx', model)

def read_model(args):
    if isinstance(args,dict):
        args = Namespace(**args)
        args_path = glob.glob(f'../eqx_models/log*{args.log_path}*')[0]
        param_path = glob.glob(f'../eqx_models/renonet*{args.log_path}*')[0]
    elif args.log_path:
        args_path = glob.glob(f'../eqx_models/log*{args.log_path}*')[0]
        param_path = glob.glob(f'../eqx_models/renonet*{args.log_path}*')[0]
        with open(args_path, 'rb') as f: data = pickle.load(f)
        args_load = Namespace(**data['args'])
        #print(args, args_load)
        #args = Namespace(**args) #args_load
        for k in args.__dict__.keys():
            continue
            if k not in ['lr', 'epochs', 'weight_decay', 'max_norm', 'opt_study', 'w_data', 'w_pde', 'verbose']:
                setattr(args, k, getattr(args_load,k))
    else:
        print('need type(args) == dict or args.log_path == True !')
        raise
    model = ROMA(args)
    model = eqx.tree_deserialise_leaves(param_path, model)
    return model, args

def round_to_nearest_thousands(number):
    suffix = ['','k','M','B','T','P']
    for i in range(5):
        if 1000**i > number:
            break
    sig = i-1
    return f'{number / 1000**sig:.0f} {suffix[sig]}'
