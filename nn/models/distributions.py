import jax
import jax.numpy as jnp
import equinox as eqx
import tensorflow_probability.substrates.jax as jpr

#cov_param_dict = {
#    'dense': DenseNormal,
#    'diagonal': Normal,
    # 'lowrank': LowRankCovariance
#}

#def get_parameterization(p):
#  if p in cov_param_dict:
#    return cov_param_dict[p]
#  else:
#    raise ValueError('Must specify a valid covariance parameterization.')

def tp(M):
    return M.transpose(-1,-2)

def sym(M):
    return (M + tp(M))/2.

class Normal(jpr.distributions.Normal):
    def __init__(self, loc, chol):
        super(Normal, self).__init__(loc, chol)

    @property
    def mean(self):
        return self.loc

    @property
    def var(self):
        return self.scale ** 2

    @property
    def chol_covariance(self):
        #return jnp.diag_embed(self.scale)
        return jnp.diag(self.scale)

    @property
    def covariance_diagonal(self):
        return self.var

    @property
    def covariance(self):
        #return jnp.diag_embed(self.var)
        return jnp.diag(self.var)

    @property
    def precision(self):
        #return jnp.diag_embed(1./self.var)
        return jnp.diag(1./self.var)

    @property
    def logdet_covariance(self):
        return 2 * jnp.log(self.scale).sum(-1)

    @property
    def logdet_precision(self):
        return -2 * jnp.log(self.scale).sum(-1)

    @property
    def trace_covariance(self):
        return self.var.sum(-1)

    @property
    def trace_precision(self):
        return (1./self.var).sum(-1)

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (self.var.unsqueeze(-1) * (b ** 2)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((b ** 2)/self.var.unsqueeze(-1)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __add__(self, inp):
        if isinstance(inp, Normal):
            new_cov =  self.var + inp.var
            return Normal(self.mean + inp.mean, jnp.sqrt(jnp.clip(new_cov, min = 1e-12)))
        elif isinstance(inp, jnp.array):
            return Normal(self.mean + inp, self.scale)
        else:
            raise NotImplementedError('Distribution addition only implemented for diag covs')

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(inp.unsqueeze(-3), reduce_dim = False)
        return Normal(self.loc @ inp, jnp.sqrt(jnp.clip(new_cov, min = 1e-12)))

    def squeeze(self, idx):
        return Normal(self.loc.squeeze(idx), self.scale.squeeze(idx))
