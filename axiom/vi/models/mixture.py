from typing import Union, Optional, Tuple
from jaxtyping import Array

from jax import lax, jit, vmap
from jax.tree_util import tree_map
import jax.numpy as jnp
from jax.nn import softmax
from jax.scipy.special import logsumexp
from functools import partial

from axiom.vi import ArrayDict, Conjugate, Distribution
from axiom.vi.models import Model
from ..utils import assign_unused, ArrayDict

# This version eliminates the idiosyncratic use batch and event shapes that were creating some confusion
# The rules that govern the shape are now:
# likelihood.batch_shape = prior.batch_shape + prior.event_shape
# self.batch_shape = prior_batch_shape
# self.event_shape = likelhood.event_shape
#
# Using this convention, the mixture model will expect as data an object of the form
#     data.shape = sample_shape + self.batch_shape + self.event_shape
#                = sample_shape + prior.batch_shape + likelhood.event_shape
#
# This creates a new problem.  specifically, the log probabilities are computed from
# the likelihood.expected_log_likelihood(x) function which expects x to be of the form
#
#     x.shape = sample_shape + likelhood.batch_shape + likelhood.event_shape
#             = sample_shape + prior.batch_shape + prior.event_shape + likelhood.event_shape
#
# So on input the shape of data is missing the contribution from prior.event_shape.  We fix this
# (and probably break jitting) by expanding data to the correct shape before passing it to the
# expected_log_likelihood function.


class Mixture(Model):
    pytree_data_fields = ("likelihood", "prior")
    pytree_aux_fields = (
        "pi_opts",
        "likelihood_opts",
        "assign_unused_opts",
        "batch_shape_prior",
        "event_shape_prior",
        "flattened_batch_shape",
    )

    def __init__(
        self,
        likelihood: Conjugate,
        prior: Conjugate,
        pi_opts: Optional[dict] = None,
        likelihood_opts: Optional[dict] = None,
        assign_unused_opts: Optional[dict] = None,
    ):
        assert prior.batch_dim + prior.event_dim == likelihood.batch_dim
        super().__init__(likelihood.default_event_dim, prior.batch_shape, likelihood.event_shape)

        self.likelihood = likelihood
        self.prior = prior

        self.pi_opts = pi_opts if pi_opts is not None else {"lr": 1.0, "beta": 0.0}
        self.likelihood_opts = likelihood_opts if likelihood_opts is not None else {"lr": 1.0, "beta": 0.0}
        self.assign_unused_opts = (
            assign_unused_opts if assign_unused_opts is not None else {"d_alpha_thr": 1.0, "fill_value": 1.0}
        )

        self.batch_shape_prior = prior.batch_shape
        self.event_shape_prior = prior.event_shape
        flattened_batch_shape = 1
        for i in self.batch_shape_prior:
            flattened_batch_shape *= i
        self.flattened_batch_shape = (flattened_batch_shape,)

    def update_from_data(
        self, data: Union[Array, tuple], iters: int = 1, assign_unused: bool = False
    ) -> tuple[Array, Array]:

        # expand data to a shape consistent with the likelihood by inserting singleton dimemsions for the mixture component
        data = self.expand_to_categorical_dims(data)
        likelihood, prior, elbo = self._update_from_data(
            data, self.likelihood, self.prior, iters=iters, assign_unused=assign_unused
        )
        mix_dims = tuple(range(-self.prior.event_dim, 0))
        posterior = softmax(likelihood.expected_log_likelihood(data) + prior.log_mean(), mix_dims)
        self.likelihood, self.prior = likelihood, prior
        return posterior, elbo

    @partial(jit, static_argnames=["iters", "assign_unused"])
    def _update_from_data(
        self,
        data: Union[Array, tuple],
        likelihood: Conjugate,
        prior: Conjugate,
        iters: int = 1,
        assign_unused: bool = False,
    ):
        # sample_dims = tuple(range(len(data[0].shape) - self.batch_dim - self.prior.event_dim - self.likelihood.event_dim))
        sample_dims = self.get_sample_dims(data)
        mix_dims = tuple(range(-self.prior.event_dim, 0))

        def step_fn(carry, _):
            likelihood, prior = carry

            """ E step """
            log_probs = likelihood.expected_log_likelihood(data) + prior.log_mean()
            posterior = softmax(log_probs, mix_dims)

            """ Compute ELBO """
            elbo_contrib = logsumexp(log_probs, mix_dims)
            elbo = (
                elbo_contrib.sum(sample_dims)
                - self.likelihood.kl_divergence().sum(mix_dims)
                - self.prior.kl_divergence()
            )
            """ M step """
            # temp = self._to_stats(posterior, sample_dim)

            if assign_unused:
                posterior = self.assign_unused(elbo_contrib, posterior, **self.assign_unused_opts)

            prior.update_from_statistics(self._to_stats(posterior, sample_dims), **self.pi_opts)
            likelihood.update_from_data(data, posterior, **self.likelihood_opts)

            return (likelihood, prior), elbo

        init_distributions = (likelihood, prior)
        (likelihood, prior), elbo = lax.scan(step_fn, init_distributions, jnp.arange(iters))
        return likelihood, prior, elbo

    def update_from_probabilities(
        self, inputs: Distribution, iters: int = 1, assign_unused: bool = False
    ) -> tuple[Array, Array]:
        inputs = self.expand_to_categorical_dims_for_probs(inputs)
        likelihood, prior, elbo = self._update_from_probabilities(
            inputs, self.likelihood, self.prior, iters=iters, assign_unused=assign_unused
        )
        mix_dims = tuple(range(-self.prior.event_dim, 0))
        posterior = softmax(likelihood.average_energy(inputs) + prior.log_mean(), mix_dims)
        self.likelihood, self.prior = likelihood, prior
        return posterior, elbo

    @partial(jit, static_argnames=["iters", "assign_unused"])
    def _update_from_probabilities(
        self, inputs: Distribution, likelihood: Conjugate, prior: Conjugate, iters: int = 1, assign_unused: bool = False
    ):
        sample_dims = self.get_sample_dims(inputs)
        mix_dims = tuple(range(-self.prior.event_dim, 0))

        def step_fn(carry, _):
            likelihood, prior = carry

            """ E step """
            log_probs = likelihood.average_energy(inputs) + prior.log_mean()
            elbo_contrib = logsumexp(log_probs, mix_dims)
            elbo = (
                elbo_contrib.sum(sample_dims)
                - self.likelihood.kl_divergence().sum(mix_dims)
                - self.prior.kl_divergence()
            )
            posterior = softmax(log_probs, mix_dims)
            if assign_unused:
                posterior = self.assign_unused(elbo_contrib, posterior, **self.assign_unused_opts)

            """ M step """
            prior.update_from_statistics(self._to_stats(posterior, sample_dims), **self.pi_opts)
            likelihood.update_from_probabilities(inputs, posterior, **self.likelihood_opts)

            return (likelihood, prior), elbo

        init_distributions = (likelihood, prior)
        (likelihood, prior), elbo = lax.scan(step_fn, init_distributions, jnp.arange(iters))
        return likelihood, prior, elbo

    def assign_unused(
        self, elbo_contrib: Array, assignments: Array, d_alpha_thr: float = 1.0, fill_value: float = 1.0
    ) -> Array:

        assignments_r = assignments.reshape((-1,) + self.flattened_batch_shape + self.event_shape_prior)
        d_alpha_r = (self.prior.alpha - self.prior.prior_alpha).reshape(
            self.flattened_batch_shape + self.event_shape_prior
        )
        elbo_contrib_r = elbo_contrib.reshape((-1,) + self.flattened_batch_shape)
        assignments_reass = vmap(assign_unused, in_axes=(1, 0, 1), out_axes=1)(assignments_r, d_alpha_r, elbo_contrib_r)
        return assignments_reass.reshape((-1,) + self.batch_shape_prior + self.event_shape_prior)

    def get_sample_dims(self, data):
        if type(data) is tuple:
            sample_dims = tuple(
                range(len(data[0].shape) - self.batch_dim - self.prior.event_dim - self.likelihood.event_dim)
            )
        else:
            sample_dims = tuple(
                range(len(data.shape) - self.batch_dim - self.prior.event_dim - self.likelihood.event_dim)
            )
        return sample_dims

    def get_sample_shape(self, data):
        sample_dims = self.get_sample_dims(data)
        return sample_dims

    def expand_to_categorical_dims(self, data: Array) -> Array:
        mix_dims = tuple(range(-self.prior.event_dim - self.likelihood.event_dim, -self.likelihood.event_dim))
        if type(data) is tuple:
            data = tree_map(lambda d: jnp.expand_dims(d, mix_dims), data)
        else:
            data = jnp.expand_dims(data, mix_dims)
        return data

    def expand_to_categorical_dims_for_probs(
        self, inputs: Union[Tuple[Distribution], Distribution]
    ) -> Union[Tuple[Distribution], Distribution]:
        mix_dims = tuple(range(-self.prior.event_dim, 0))
        if isinstance(inputs, tuple):
            expanded_inputs = tree_map(
                lambda x: x.expand_batch_shape(mix_dims), inputs, is_leaf=lambda x: isinstance(x, Distribution)
            )
        else:
            expanded_inputs = inputs.expand_batch_shape(mix_dims)
        return expanded_inputs

    def elbo(self, data: Array) -> Array:
        data = self.expand_to_categorical_dims(data)
        sample_dims = tuple(range(len(data.shape) - self.batch_dim - self.prior.event_dim - self.likelihood.event_dim))
        mix_dims = tuple(range(-self.prior.event_dim, 0))
        log_probs = self.likelihood.expected_log_likelihood(data) + self.prior.log_mean()
        log_z = logsumexp(log_probs, mix_dims)
        return jnp.sum(log_z, sample_dims) - self.likelihood.kl_divergence().sum(mix_dims) - self.prior.kl_divergence()

    def _to_stats(self, posterior: Array, sample_dims: int) -> ArrayDict:
        return ArrayDict(eta=ArrayDict(eta_1=posterior.sum(sample_dims)), nu=None)

    def get_assignments_from_data(self, data: Union[Array, Tuple]) -> Array:
        data = self.expand_to_categorical_dims(data)
        return softmax(
            self.likelihood.expected_log_likelihood(data) + self.prior.log_mean(), list(range(-self.prior.event_dim, 0))
        )

    def get_assignments_from_probabilities(self, inputs: Union[Distribution, Tuple[Distribution]]) -> Array:
        inputs = self.expand_to_categorical_dims_for_probs(inputs)
        return softmax(
            self.likelihood.average_energy(inputs) + self.prior.log_mean(), list(range(-self.prior.event_dim, 0))
        )

    def predict(self, X):
        pY = self.likelihood.predict(X)
        Res = pY.residual
        log_p = Res + self.prior.log_mean()

        log_p = log_p - jnp.max(log_p, axis=-1, keepdims=True)[0]

        p = jnp.exp(log_p)
        p = p / jnp.sum(p, axis=-1, keepdims=True)

        p = jnp.expand_dims(jnp.expand_dims(p, -1), -1)
        Sigma = ((pY.sigma + pY.mean @ pY.mean.swapaxes(-2, -1)) * p).sum(-3)
        mu = (pY.mean * p).sum(-3)
        Sigma = Sigma - mu @ mu.swapaxes(-2, -1)
        return mu, Sigma, p.squeeze(-1).squeeze(-1)


if __name__ == "__main__":
    from optimus.conjugate import Multinomial, MultivariateNormal
    from optimus.exponential import MultivariateNormal as MVN
    import jax.random as jr

    key = jr.PRNGKey(11)
    batch_shape = ()
    num_samples = 400
    component_shape = (8,)
    x_dim = 2
    event_shape = (x_dim, 1)
    scale = 1.0 / jnp.sqrt(jnp.prod(jnp.array(component_shape)))

    """init prior on cluster assignment probabilities"""
    prior_params = ArrayDict(alpha=jnp.full(component_shape, 0.5))
    params = ArrayDict(alpha=prior_params.alpha * (1 + jr.uniform(key, component_shape)))
    prior = Multinomial(params, prior_params).to_event(len(component_shape) - 1)

    """init multivariate normal observation model"""
    mean = jnp.zeros(component_shape + event_shape)
    kappa = jnp.ones(component_shape + (1, 1))
    u_expand_dims = list(range(-len(component_shape) - len(event_shape), -len(event_shape)))

    u = jnp.expand_dims((1.0 / scale) ** 2 * jnp.eye(event_shape[-2]), u_expand_dims)
    u = jnp.broadcast_to(u, component_shape + event_shape[:-1] + (event_shape[-2],))
    n = jnp.full(component_shape + (1, 1), event_shape[-2] + 2.0)

    params = ArrayDict(mean=mean + jr.normal(key, mean.shape), kappa=kappa, u=u, n=n)
    prior_params = ArrayDict(mean=mean, kappa=kappa, u=u, n=n)

    likelihood = MultivariateNormal(params, prior_params)

    num_components = jnp.prod(jnp.array(component_shape) // 2)
    mu = 2 * jr.normal(key, (num_components,) + event_shape)
    A = jr.normal(key, (num_components,) + event_shape[:-1] + event_shape[-2:-1]) / jnp.sqrt(num_components)

    data = mu[0] + jr.normal(key, (num_samples // num_components,) + event_shape)
    for i in range(1, num_components):
        data = jnp.concatenate(
            (data, mu[i] + A[i] @ jr.normal(key, (num_samples // num_components,) + event_shape)), axis=0
        )
    mu = mu / data.std()
    data = data / data.std()
    mu = mu.squeeze()
    data_plot = data

    x_dim = data.shape[-2]
    inv_sigma = jnp.broadcast_to(jnp.eye(x_dim) * 100, data.shape[:-2] + (x_dim, x_dim))
    px = MVN(ArrayDict(inv_sigma_mu=data * 100, inv_sigma=inv_sigma))
    model = Mixture(likelihood, prior)
    p, e = model.update_from_data(data, iters=20)

    p = p.reshape(p.shape[: -len(component_shape)] + (jnp.prod(jnp.array(p.shape[-len(component_shape) :])),))
    mu_hat = model.likelihood.mean.squeeze()

    from matplotlib import pyplot as plt

    plt.scatter(data_plot[:, 0], data_plot[:, 1], c=p.argmax(-1) / num_components)
    plt.scatter(mu[:, 0], mu[:, 1], c="k")
    plt.scatter(mu_hat[..., 0], mu_hat[..., 1], c="r")
    plt.show()

    model = Mixture(likelihood, prior)
    model.optim_args["iters"] = 10
