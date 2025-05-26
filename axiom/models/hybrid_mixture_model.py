from functools import partial
from typing import Dict, List

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax.nn as nn
from jax.nn import softmax
from jax.scipy.special import logsumexp
from jaxtyping import Array

import equinox as eqx

from axiom.vi import ArrayDict
from axiom.vi.conjugate import Multinomial, MultivariateNormal


class HybridMixture(eqx.Module):
    """A mixture model with both discrete and continuous likelihoods."""

    discrete_likelihoods: List[Multinomial]
    continuous_likelihood: MultivariateNormal
    prior: Multinomial
    pi_opts: Dict
    likelihood_opts: Dict

    def __init__(
        self,
        discrete_likelihoods: List[Multinomial],
        continuous_likelihood: MultivariateNormal,
        prior: Multinomial,
        pi_opts=None,
        likelihood_opts=None,
    ):
        super().__init__()
        self.discrete_likelihoods = discrete_likelihoods
        self.continuous_likelihood = continuous_likelihood
        self.prior = prior

        self.pi_opts = pi_opts if pi_opts is not None else {"lr": 1.0, "beta": 0.0}
        self.likelihood_opts = (
            likelihood_opts if likelihood_opts is not None else {"lr": 1.0, "beta": 0.0}
        )

    @property
    def mix_dims(self):
        return tuple(range(-self.prior.event_dim, 0))

    def expand_to_categorical_dims(self, data, batch_dim, event_dim):
        mix_dims = tuple(range(-batch_dim - event_dim, -event_dim))
        if type(data) is tuple:
            data = jtu.tree_map(lambda d: jnp.expand_dims(d, mix_dims), data)
        else:
            data = jnp.expand_dims(data, mix_dims)
        return data

    def get_sample_dims(self, data, batch_dim, lh_event_dim):
        sample_dims = tuple(
            range(len(data.shape) - batch_dim - self.prior.event_dim - lh_event_dim)
        )

        return sample_dims

    def _to_stats(self, posterior, sample_dims: int):
        return ArrayDict(eta=ArrayDict(eta_1=posterior.sum(sample_dims)), nu=None)

    @jax.jit
    def _e_step(self, c_data: Array, d_data: List[Array], w_disc=None):
        if w_disc is None:
            w_disc = jnp.array([1.0] * len(self.discrete_likelihoods))

        c_data = self.expand_to_categorical_dims(
            c_data,
            self.continuous_likelihood.batch_dim,
            self.continuous_likelihood.event_dim,
        )

        cont_ell = self.continuous_likelihood.expected_log_likelihood(c_data)

        disc_ell = jnp.zeros_like(cont_ell)
        for i, dlh in enumerate(self.discrete_likelihoods):
            d_data_i = self.expand_to_categorical_dims(
                d_data[i], dlh.batch_dim, dlh.event_dim
            )

            # More weight on action
            disc_ell += w_disc[i] * dlh.expected_log_likelihood(d_data_i)

        posterior = softmax(cont_ell + disc_ell, self.mix_dims)

        return posterior, cont_ell, disc_ell

    def _m_step(self, c_data: Array, d_data: List[Array], qz: Array):
        self_copy = jtu.tree_map(lambda x: x, self)

        c_data = self.expand_to_categorical_dims(
            c_data,
            self.continuous_likelihood.batch_dim,
            self.continuous_likelihood.event_dim,
        )

        sample_dims = self.get_sample_dims(
            c_data, self.prior.batch_dim, self.continuous_likelihood.event_dim
        )

        self.prior.update_from_statistics(
            self._to_stats(qz, sample_dims), **self.pi_opts
        )
        self.continuous_likelihood.update_from_data(c_data, qz, **self.likelihood_opts)

        for i, dlh in enumerate(self.discrete_likelihoods):
            d_data_i = self.expand_to_categorical_dims(
                d_data[i], dlh.batch_dim, dlh.event_dim
            )
            dlh.update_from_data(d_data_i, qz, **self.likelihood_opts)

        # If the component has been used before, we want to sum the SST of the previous
        # step as well. In case it wasn't used, the second mask will be zero, and this
        # is essentially a no-op. NOTE: assumes only one data point.
        was_used = self_copy.prior.alpha[qz[0].argmax()] > 1
        self._sum_components(self_copy, jnp.ones_like(qz[0]), qz[0] * was_used)

    def _sum_components(self, other, self_mask, other_mask):
        """Sum two components together, w/o double counting prior"""

        def fn(other_prior, self_posterior, other_posterior):
            shape = (-1, *(1 for _ in range(len(other_prior.shape) - 1)))
            pos_1 = self_posterior * self_mask.reshape(shape)
            pos_2 = other_posterior * other_mask.reshape(shape)
            pri_2 = other_prior * other_mask.reshape(shape)
            res = pos_1 + pos_2 - pri_2
            return res

        self.prior.posterior_params = jtu.tree_map(
            fn,
            other.prior.prior_params,
            self.prior.posterior_params,
            other.prior.posterior_params,
        )

        self.continuous_likelihood.posterior_params = jtu.tree_map(
            fn,
            other.continuous_likelihood.prior_params,
            self.continuous_likelihood.posterior_params,
            other.continuous_likelihood.posterior_params,
        )

        for i in range(len(self.discrete_likelihoods)):
            self.discrete_likelihoods[i].posterior_params = jtu.tree_map(
                fn,
                other.discrete_likelihoods[i].prior_params,
                self.discrete_likelihoods[i].posterior_params,
                other.discrete_likelihoods[i].posterior_params,
            )

    def _combine_params(self, other, select_from_other_mask):
        """Select components form two models together"""

        def fn(x_0, x_1):
            u = select_from_other_mask.reshape(
                -1, *[1 for _ in range(len(x_0.shape) - 1)]
            )
            return x_1 * u + x_0 * (1 - u)

        # Update prior
        self.prior.posterior_params = jtu.tree_map(
            lambda x_0, x_1: fn(x_0, x_1),
            self.prior.posterior_params,
            other.prior.posterior_params,
        )

        # update likelihoods
        self.continuous_likelihood.posterior_params = jtu.tree_map(
            lambda x_0, x_1: fn(x_0, x_1),
            self.continuous_likelihood.posterior_params,
            other.continuous_likelihood.posterior_params,
        )

        for i in range(len(self.discrete_likelihoods)):
            self.discrete_likelihoods[i].posterior_params = jtu.tree_map(
                lambda x_0, x_1: fn(x_0, x_1),
                self.discrete_likelihoods[i].posterior_params,
                other.discrete_likelihoods[i].posterior_params,
            )

    def _merge_clusters(self, idx_1, idx_2):
        """Sum two components together, w/o double counting prior"""

        def combine_fn(prior, posterior):
            posterior = posterior.at[idx_1].set(
                posterior[idx_1] + posterior[idx_2] - prior[idx_2]
            )
            posterior = posterior.at[idx_2].set(prior[idx_2])
            return posterior

        self.prior.posterior_params = jtu.tree_map(
            combine_fn, self.prior.prior_params, self.prior.posterior_params
        )

        self.continuous_likelihood.posterior_params = jtu.tree_map(
            combine_fn,
            self.continuous_likelihood.prior_params,
            self.continuous_likelihood.posterior_params,
        )

        for i in range(len(self.discrete_likelihoods)):
            self.discrete_likelihoods[i].posterior_params = jtu.tree_map(
                combine_fn,
                self.discrete_likelihoods[i].prior_params,
                self.discrete_likelihoods[i].posterior_params,
            )

    def _m_step_keep_unused(self, c_data: Array, d_data: List[Array], qz: Array):
        model_updated: HybridMixture = jax.tree_map(lambda x: x, self)
        model_updated._m_step(c_data, d_data, qz)

        # 0.25 ensures that it only overwrites if the component is actually used
        active_components = jnp.any(qz > 0.25, axis=0) & (
            model_updated.prior.alpha < 32
        )
        mask_updated = jnp.where(active_components, 1.0, 0.0)
        self._combine_params(model_updated, mask_updated)

    def sample(self, key, n_samples=1):
        @jax.jit
        def make_pd(mat):
            # ensure that the matrix is positive definite
            min_eig = jnp.clip(jnp.min(jnp.linalg.eigvalsh(mat)), max=0.0)
            eps = jnp.finfo(mat.dtype).eps
            mat = mat + jnp.broadcast_to(jnp.eye(mat.shape[-1]), mat.shape) * 2 * (
                eps - min_eig
            ) * (min_eig < 0)
            return mat

        # NOTE: we sample from the mean of the MVN, so the uncertainty over the distribution
        # is not captured. #TODO!
        mu = self.continuous_likelihood.mean
        si = jax.vmap(make_pd)(self.continuous_likelihood.expected_sigma())

        mask = self.prior.alpha > self.prior.prior_alpha

        @partial(jax.jit, static_argnames=["to_1h"])
        def _sample_discrete(_key, alpha, to_1h=False):
            z = jr.choice(_key, jnp.arange(alpha.shape[-1]), p=alpha / alpha.sum())
            if to_1h:
                z = nn.one_hot(z, num_classes=alpha.shape[-1])
            return z

        @jax.jit
        def _sample(_key):
            _key, *_subkeys = jr.split(_key, 3 + len(self.discrete_likelihoods))
            z = _sample_discrete(_subkeys[0], self.prior.alpha * mask)

            c_sample = jr.multivariate_normal(_subkeys[1], mu[z, :, 0], si[z])[
                ..., None
            ]

            d_sample = [
                _sample_discrete(
                    _subkeys[2 + i],
                    self.discrete_likelihoods[i].alpha[z, :, 0],
                    to_1h=True,
                )[..., None]
                for i in range(len(self.discrete_likelihoods))
            ]

            return c_sample, d_sample

        key, *subkeys = jr.split(key, n_samples + 1)
        return jax.vmap(lambda seed: _sample(seed))(jnp.array(subkeys))

    @jax.jit
    def compute_elbo(self, c_data, d_data):
        """
        NOTE: not the real elbo. This only accounts for the likelihoods on the TMM
        switch and the reward, and only considers the prior complexities. As these are
        the only ones we want to optimize for in BMR
        """
        mask = self.prior.alpha > self.prior.prior_alpha

        # Only account for the TMM and reward likelihoods!

        # First calculate predicted qz's
        qz_predict, *_ = self._e_step(
            c_data, d_data, w_disc=jnp.ones(len(d_data)).at[-2:].set(0)
        )

        # Then calculate ELL for the TMM and reward likelihoods
        *_, d_ell = self._e_step(
            c_data, d_data, w_disc=jnp.zeros(len(d_data)).at[-2:].set(1)
        )

        # Remove unused components
        d_ell = d_ell * mask[None]

        # And get ELL for inferred qz as elbo contrib
        d_ell = jax.vmap(lambda i, j: d_ell[i, j])(
            jnp.arange(d_ell.shape[0]), qz_predict.argmax(-1)
        )

        _elbo_contrib = d_ell.mean()
        _elbo = _elbo_contrib - self.prior.kl_divergence()

        return _elbo

    def get_means_as_data(self):
        cx = self.continuous_likelihood.mean
        dx = [d.mean() for d in self.discrete_likelihoods]
        return cx, dx


def create_mm(
    key,
    num_components,
    continuous_dim,
    discrete_dims,
    discrete_alphas=None,
    cont_scale=1,
    color_precision_scale=None,
    opt=None,
):
    if opt is None:
        opt = {"lr": 1, "beta": 1}

    component_shape = (num_components,)

    # TODO: proper parameterization
    prior_params = MultivariateNormal.init_default_params(
        component_shape, (continuous_dim, 1), cont_scale, dof_offset=continuous_dim
    )
    prior_params = eqx.tree_at(
        lambda x: x.kappa, prior_params, prior_params.kappa * 1e-4
    )

    if color_precision_scale is not None:
        spatial_dim = 2
        color_dim = continuous_dim - spatial_dim

        u = prior_params.u
        for i in range(spatial_dim, continuous_dim):
            u = u.at[..., i, i].set(u[..., i, i] * color_precision_scale)

        prior_params = eqx.tree_at(lambda x: x.u, prior_params, u)

    key, subkey = jr.split(key)
    params = eqx.tree_at(
        lambda x: x.mean,
        prior_params,
        prior_params.mean + jr.normal(subkey, shape=prior_params.mean.shape),
    )

    key, subkey = jr.split(key)
    continuous_likelihood = MultivariateNormal(
        batch_shape=component_shape,
        event_shape=(continuous_dim, 1),
        event_dim=2,
        init_key=subkey,
        params=params,
        prior_params=prior_params,
    )

    if discrete_alphas is None:
        discrete_alphas = [1e-4] * len(discrete_dims)

    discrete_likelihoods = []
    for discrete_alpha, discrete_dim in zip(discrete_alphas, discrete_dims):
        key, subkey = jr.split(key)
        alpha = discrete_alpha * jnp.ones(component_shape + (discrete_dim, 1))
        prior_params = ArrayDict(alpha=alpha)

        # Eye prior
        d = min(num_components, discrete_dim)
        eye = jnp.eye(d)
        nr, nc = int(jnp.ceil(num_components / d)), int(jnp.ceil(discrete_dim / d))
        eye = jnp.tile(eye, (nr, nc))
        eye = eye[:num_components, :discrete_dim, None]

        key, subkey = jr.split(key)
        idcs = jr.permutation(subkey, eye.shape[0], independent=True)

        alpha = alpha + eye[idcs] * 10

        alpha = alpha * (jr.uniform(subkey, shape=alpha.shape, minval=0.1, maxval=0.9))
        discrete_likelihoods.append(
            Multinomial(
                batch_shape=component_shape,
                event_shape=(discrete_dim, 1),
                initial_count=1,
                init_key=subkey,
                params=ArrayDict(alpha=alpha),
                prior_params=prior_params,
            )
        )

    key, subkey = jr.split(key)
    alpha = 0.1 * jnp.ones(component_shape)
    prior = Multinomial(
        batch_shape=(),
        event_shape=component_shape,
        prior_params=ArrayDict(alpha=alpha),
        params=ArrayDict(alpha=alpha),
    )

    return HybridMixture(
        discrete_likelihoods,
        continuous_likelihood,
        prior,
        pi_opts=opt,
        likelihood_opts=opt,
    )


def train_step_fn(model: HybridMixture, mask, c_sample, d_sample, logp_thr=-0.1):
    qz, c_ell, d_ell = model._e_step(c_sample, d_sample)
    elogp = c_ell + d_ell

    # Mask out the contributions of unused components

    elogp = elogp * mask[None] + (1 - mask[None]) * (-1e10)

    qz = softmax(elogp, model.mix_dims)

    def true_fn(qz, mask):
        return qz, mask

    def false_fn(qz, mask):
        # Find a component that is unused
        idx = jnp.argmax(mask == 0)
        # Assign the the data point to this component
        qz_new = (qz * 0.0).at[:, idx].set(1.0)
        # Update the mask to reflect that this component is now used
        mask_new = mask.at[idx].set(1.0)
        return qz_new, mask_new

    qz, mask = jax.lax.cond(elogp.max() > logp_thr, true_fn, false_fn, qz, mask)

    model._m_step_keep_unused(c_sample, d_sample, qz)

    mask = (model.prior.alpha > model.prior.prior_alpha) * 1.0

    return model, mask, qz
