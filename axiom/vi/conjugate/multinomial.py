from typing import Optional
import jax.numpy as jnp
from jax.lax import lgamma
from jax import random as jr
from jax.scipy import special as jsp
from jaxtyping import Array, PRNGKeyArray

from .base import Conjugate
from ..exponential.base import ExponentialFamily
from ..exponential import Multinomial as MultinomialLikelihood
from ..utils import params_to_tx, ArrayDict, tree_copy

DEFAULT_EVENT_DIM = 1


@params_to_tx({"eta_1": "x"})
class Multinomial(Conjugate):
    """
    Multinomial model class that extends Distribution with a Dirichlet conjugate prior

    The likelihood is given by:

    .. math::
        f(x) = \frac{n!}{x_1! cdots x_K!} p_1^{x_1} cdots p_K^{x_K}

    where the :math:`x_i` are non-negative integers and :math:`sum_{i=1}^K x_i = n`.
    The conjugate prior can be expressed as:

    .. math::

        f(x) = \frac{1}{mathrm{B}(\boldsymbol\alpha)} prod_{i=1}^K x_i^{\alpha_i - 1}

    where

    .. math::

        mathrm{B}(\boldsymbol\alpha) = \frac{prod_{i=1}^K Gamma(\alpha_i)}
                                     {Gamma\bigl(sum_{i=1}^K \alpha_i\bigr)}

    and :math:`\boldsymbol\alpha=(\alpha_1,ldots,\alpha_K)`, the
    concentration parameters and :math:`K` is the dimension of the space
    where :math:`x` takes values.
    """

    def __init__(
        self,
        params: Optional[ArrayDict] = None,
        prior_params: Optional[ArrayDict] = None,
        batch_shape: Optional[tuple] = None,
        event_shape: Optional[tuple] = None,
        event_dim: Optional[int] = None,
        initial_count: float = 1.0,  # initial pseudo-counts for the Dirichlet prior
        init_key: Optional[
            PRNGKeyArray
        ] = None,  # random key for initializing the posterior parameters based on the prior
    ):
        if event_shape is not None:
            event_dim = len(event_shape)
        else:
            event_dim = event_dim if event_dim is not None else DEFAULT_EVENT_DIM

        if prior_params is None:
            prior_params = self.init_default_params(batch_shape, event_shape, initial_count)
        if params is None:  # set the values of posterior parameters equal to the prior, if not provided
            init_key = jr.PRNGKey(0) if init_key is None else init_key
            params = {}
            for k, v in prior_params.items():
                if k == "alpha":
                    params[k] = v * (1 + jr.uniform(init_key, shape=v.shape))
                else:
                    params[k] = v
            params = ArrayDict(**params)

        inferred_batch_shape, inferred_event_shape = self.infer_shapes(params.alpha, event_dim)
        batch_shape = batch_shape if batch_shape is not None else inferred_batch_shape
        event_shape = event_shape if event_shape is not None else inferred_event_shape

        super().__init__(DEFAULT_EVENT_DIM, MultinomialLikelihood, params, prior_params, batch_shape, event_shape)

    @staticmethod
    def init_default_params(batch_shape, event_shape, initial_counts: float = 1.0) -> ArrayDict:
        """Initialize the default canonical parameters of the distribution."""

        return ArrayDict(alpha=initial_counts * jnp.ones(batch_shape + event_shape))

    @property
    def alpha(self) -> Array:
        """
        Property that accesses the posterior parameters of the Dirichlet distribution.
        This function must implicitly or explicitly implement the inverse of `transform_params`
        to get the common parameters from the natural parameters.

        Returns:
            Array (batch_shape, event_shape)
        """
        return self.posterior_params.eta.eta_1

    @property
    def prior_alpha(self) -> Array:
        """
        Property that accesses the prior parameters of the Dirichlet distribution.
        This function must implicitly or explicitly implement the inverse of `transform_params`
        to get the common parameters from the natural parameters.

        Returns:
            Array (batch_shape, event_shape)
        """
        return self.prior_params.eta.eta_1

    def to_natural_params(self, params: ArrayDict) -> ArrayDict:
        """
        Transforms the common parameters to the natural parameters of the distribution.
        """
        return ArrayDict(eta=ArrayDict(eta_1=params.alpha), nu=None)

    def expected_likelihood_params(self) -> ArrayDict:
        """
        Computes the expected natural parameters of the likelihood <S(\theta)>_{q(\theta|eta,\nu)},
        with respect to a posterior over theta, parameterized by eta and \nu aka q(\theta|eta,\nu).

        .. math::
            <S(\theta)>_{q(\theta|eta,\nu)}
        """
        return ArrayDict(eta_1=self.alpha / self.sum_events(self.alpha, keepdims=True))

    def expected_log_likelihood(self, x: Array) -> Array:
        """
        Computes the expected log likelihood of data under the distribution.

         .. math::
            log(phi(x)) + T(x) cdot <S(\theta)>_q(\theta|eta,\nu) - <A(\theta)>_{q(\theta|eta,\nu)}

        Args:
            x: Array (sample_shape, batch_shape, event_shape)
                Data point(s) to compute the expected log probability for.

        Returns:
            Array (sample_shape, batch_shape)
                The expected log likelihood value(s)
        """
        return self.sum_events(x * self.log_mean()) + lgamma(1 + self.sum_events(x)) - self.sum_events(lgamma(1 + x))

    def expected_posterior_statistics(self) -> ArrayDict:
        """
        Computes the expected sufficient statistics for eta and \nu:

        .. math::
            (<S(\theta)>_{q(\theta|eta,\nu)}, -<A(\theta)>_{q(\theta|eta,\nu)})
        """
        alpha_stats = jsp.digamma(self.alpha) - jsp.digamma(self.sum_events(self.alpha, keepdims=True))
        return ArrayDict(eta=ArrayDict(eta_1=alpha_stats), nu=None)

    def expected_log_partition(self) -> Array:
        """
        Computes the log partition of the distribution <A(\theta)>_{q(\theta|eta,\nu)}.
        """
        return self.sum_events(jsp.digamma(self.alpha)) - jsp.digamma(self.sum_events(self.alpha))

    def log_prior_partition(self) -> Array:
        """
        Computes the log partition of the prior distribution log Z(eta_0,\nu_0)
        """
        return self.sum_events(jsp.gammaln(self.alpha)) - jsp.gammaln(self.sum_events(self.alpha))

    def log_posterior_partition(self) -> Array:
        """
        Computes the log partition of the distribution log Z(eta,\nu)
        """
        return self.sum_events(jsp.gammaln(self.alpha)) - jsp.gammaln(self.sum_events(self.alpha))

    def residual(self) -> Array:
        """
        Computes the difference between expected log partition and the log partition evaluated at
        the expected sufficient statistic:

        .. math::
            A(left<\theta\right>_{q(\theta|eta,\nu)}) - <A(\theta)>_{q(\theta|eta,\nu)}
            log_partition() - expected_log_partition()
        """
        raise NotImplementedError

    def kl_divergence(self) -> Array:
        """
        Computes the kl divergence between the distribution and the prior.

        Returns:
            Array (batch_shape)
                The kl divergence
        """

        alpha_sum = self.sum_events(self.alpha)
        prior_alpha_sum = self.sum_events(self.prior_alpha)
        return (
            lgamma(alpha_sum)
            - self.sum_events(lgamma(self.alpha))
            - lgamma(prior_alpha_sum)
            + self.sum_events(lgamma(self.prior_alpha))
            + self.sum_events(
                (self.alpha - self.prior_alpha)
                * (jsp.digamma(self.alpha) - self.expand_event_dims(jsp.digamma(alpha_sum)))
            )
        )

    def forward(self) -> ExponentialFamily:
        """
        Returns a message distribution and associated residual of the same type
        as the likelihood with parameters given by <S(\theta)>.

        .. math::
            p(\theta|<S(\theta)>_{q(\theta|eta,\nu)})
                = phi(x) exp (T(x) cdot <S(\theta)>{q(\theta|eta,\nu)}
                - A(<S(\theta)>_{q(\theta|eta,\nu)}))
        """
        raise NotImplementedError

    def mean(self) -> Array:
        """
        The mean of the distribution.

        Returns:
            Array (batch_shape, event_shape)
                The mean of the distribution
        """
        return self.alpha / self.sum_events(self.alpha, keepdims=True)

    def log_mean(self) -> Array:
        """
        The log geometric mean of the distribution.

        Returns:
            Array (batch_shape, event_shape)
                The variance of the distribution
        """
        return jsp.digamma(self.alpha) - jsp.digamma(self.sum_events(self.alpha, keepdims=True))

    def mode(self) -> Array:
        """
        Computes the mode of the distribution.
        """
        raise NotImplementedError

    def variance(self) -> Array:
        """
        The variance of the distribution.

        Returns:
            Array (batch_shape, event_shape)
                The variance of the distribution
        """
        alpha_sum = self.sum_events(self.alpha, keepdims=True)
        return self.alpha * (alpha_sum - self.alpha) / (alpha_sum**2 * (alpha_sum + 1))

    # def entropy(self) -> Array:
    #     """
    #     The entropy of the distribution.

    #     Returns:
    #         Array (batch_shape)
    #             The entropy of the distribution
    #     """
    #     k = self.alpha.shape[-1]
    #     alpha_sum = self.sum_events(self.alpha)
    #     return (
    #         self.sum_events(lgamma(self.alpha))
    #         - lgamma(alpha_sum)
    #         - (k - alpha_sum) * jsp.digamma(alpha_sum)
    #         - self.sum_events((self.alpha - 1.0) * jsp.digamma(self.alpha))
    #     )

    def sample(self, key, shape=()) -> Array:
        """
        Draw random samples from the distribution.

        Args:
            key : array_like
                Random number generator key.
            shape : tuple, default=()
                Shape of the output sample.

        Returns:
            Array (shape + batch_shape)
                Array of random samples
        """
        # TODO: handle arbitary event shapes
        samples = jr.dirichlet(key, self.alpha, shape=shape + self.batch_shape)
        return jnp.clip(samples, a_min=jnp.finfo(samples).tiny, a_max=1 - jnp.finfo(samples).eps)
