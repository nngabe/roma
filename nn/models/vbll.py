import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from dataclasses import dataclass
from nn.models.distributions import Normal
from collections.abc import Callable
import tensorflow_probability.substrates.jax as jpr

def KL(p, q_scale):
    feat_dim = p.mean.shape[-1]
    mse_term = (p.mean ** 2).sum(-1).sum(-1) / q_scale
    trace_term = (p.trace_covariance / q_scale).sum(-1)
    logdet_term = (feat_dim * np.log(q_scale) - p.logdet_covariance).sum(-1)

    return 0.5*(mse_term + trace_term + logdet_term) # currently exclude constant

@dataclass
class VBLLReturn():
    predictive: Normal 
    train_loss_fn: Callable[[jnp.array], jnp.array]
    val_loss_fn: Callable[[jnp.array], jnp.array]
    ood_scores: None | Callable[[jnp.array], jnp.array] = None



class VBLL(eqx.Module):
    """
    Variational Bayesian Last Layer

    Parameters
    ----------
    in_features : int ; Number of input features
    out_features : int ; Number of output features
    regularization_weight : float ; Weight on regularization term in ELBO
    parameterization : str ; Parameterization of covariance matrix. Currently supports 'dense' and 'diagonal'
    prior_scale : float ; Scale of prior covariance matrix
    wishart_scale : float ; Scale of Wishart prior on noise covariance
    dof : float ; Degrees of freedom of Wishart prior on noise covariance
    """
    
    in_features: int
    out_features: int
    regularization_weight: jnp.float32
    prior_scale: jnp.float32
    wishart_scale: jnp.float32
    dof: jnp.float32
    regularization_weight: jnp.float32
    W_mean: jnp.array
    W_dist: jpr.distributions.Normal
    noise_mean: jnp.array
    noise_logdiag: jnp.array

    def __init__(self,
                 in_features,
                 out_features,
                 regularization_weight,
                 parameterization='dense',
                 prior_scale=1.,
                 wishart_scale=1e-2,
                 dof=1.):
        super(VBLL, self).__init__()

        self.wishart_scale = wishart_scale
        self.dof = (dof + out_features + 1.)/2.
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_scale = prior_scale

        # noise distribution
        self.noise_mean = jnp.zeros(out_features)
        self.noise_logdiag = jr.normal(prng, (out_features,)) - 1.

        # last layer distribution
        self.W_dist = jpr.Normal #get_parameterization(parameterization)
        self.W_mean = jr.normal(prng, (out_features, in_features))
        self.W_logdiag = jr.normal(prng, (out_features, in_features)) - 0.5 * np.log(in_features)
        if parameterization == 'dense':
          self.W_offdiag = jr.normal(prng, (out_features, in_features, in_features)) / in_features

    def noise_chol(self):
      return jnp.exp(self.noise_logdiag)

    def W_chol(self):
      out = jnp.exp(self.W_logdiag)
      if self.W_dist == DenseNormal:
        out = jnp.tril(self.W_offdiag, diagonal=-1) + jnp.diag_embed(out)

      return out

    def W(self):
      return self.W_dist(self.W_mean, self.W_chol())

    def noise(self):
      return Normal(self.noise_mean, self.noise_chol())

    def forward(self, x):
        out = VBLLReturn(self.predictive(x),
                         self._get_train_loss_fn(x),
                         self._get_val_loss_fn(x))
        return out

    def predictive(self, x):
        return (self.W() @ x[..., None]).squeeze(-1) + self.noise()

    def _get_train_loss_fn(self, x):

        def loss_fn(y):
            # construct predictive density N(W @ phi, Sigma)
            W = self.W()
            noise = self.noise()
            pred_density = Normal((W.mean @ x[...,None]).squeeze(-1), noise.scale)
            pred_likelihood = pred_density.log_prob(y)

            trace_term = 0.5*((W.covariance_weighted_inner_prod(jnp.expand_dims(x,-2)[..., None])) * noise.trace_precision)

            kl_term = KL(W, self.prior_scale)
            wishart_term = (self.dof * noise.logdet_precision - 0.5 * self.wishart_scale * noise.trace_precision)
            total_elbo = jnp.mean(pred_likelihood - trace_term)
            total_elbo += self.regularization_weight * (wishart_term - kl_term)
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            # compute log likelihood under variational posterior via marginalization
            return -jnp.mean(self.predictive(x).log_prob(y))

        return loss_fn
