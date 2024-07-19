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


class pooling(eqx.Module):
    
    pools: Dict[int,eqx.Module]
    embed: Dict[int,eqx.Module]
    
    def __init__(self, args, module='pool'):
        super(pooling, self).__init__()
        self.pools = {}
        self.embed = {}
        for i in range(args.pool_steps):
            self.pools[i] = getattr(models, args.pool)(args, module='pool')
        for i in range(args.pool_steps):
            self.embed[i] = getattr(models, args.pool)(args, module='embed')

    def __getitem__(self, i):
        return self.pools[i]

    def keys(self):
        return self.pools.keys()
