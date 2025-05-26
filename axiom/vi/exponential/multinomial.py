from typing import Optional
from jaxtyping import Array
from multimethod import multimethod

from jax import numpy as jnp, nn

from axiom.vi import ArrayDict, Delta, Distribution
from axiom.vi.exponential import ExponentialFamily
from axiom.vi.utils import params_to_tx, sum_pytrees, stable_logsumexp

DEFAULT_EVENT_DIM = 1


@params_to_tx({"logits": "x"})
class Multinomial(ExponentialFamily):
    """Multinomial distribution"""

    pytree_data_fields = ("_logZ",)

    def __init__(
        self,
        nat_params: Optional[ArrayDict] = None,
        batch_shape: Optional[tuple] = None,
        event_shape: Optional[tuple] = None,
        event_dim: Optional[int] = DEFAULT_EVENT_DIM,
        input_logZ: Optional[Array] = 0.0,
        **parent_kwargs,
    ):

        if event_shape is not None:
            assert len(event_shape) == event_dim, "event_shape must have length equal to event_dim"

        if nat_params is None and "expectations" not in parent_kwargs:
            nat_params = self.init_default_params(batch_shape, event_shape)

        if nat_params is not None:
            inferred_batch_shape, inferred_event_shape = self.infer_shapes(nat_params.logits, event_dim)
        elif "expectations" in parent_kwargs:
            inferred_batch_shape, inferred_event_shape = self.infer_shapes(parent_kwargs["expectations"].x, event_dim)

        batch_shape = batch_shape if batch_shape is not None else inferred_batch_shape
        event_shape = event_shape if event_shape is not None else inferred_event_shape

        super().__init__(
            DEFAULT_EVENT_DIM,
            batch_shape,
            event_shape,
            nat_params=nat_params,
            **parent_kwargs,
        )

        self._logZ = stable_logsumexp(self.logits, dims=tuple(range(-self.event_dim, 0)), keepdims=True) + input_logZ

    @staticmethod
    def init_default_params(batch_shape, event_shape) -> ArrayDict:
        """Initialize the default canonical parameters of the distribution."""

        dim = event_shape[-DEFAULT_EVENT_DIM]

        return ArrayDict(logits=jnp.zeros(batch_shape + event_shape[:-DEFAULT_EVENT_DIM] + (dim,)))

    @property
    def logits(self) -> Array:
        """
        Returns log probabilities.
        """
        return self.nat_params.logits

    @property
    def log_normalizer(self) -> Array:
        """
        Returns the log normalizer of the distribution.
        """

        if self._logZ is not None:
            return self._logZ
        else:
            logZ = stable_logsumexp(self.logits, dims=tuple(range(-self.event_dim, 0)), keepdims=True)
            self._logZ = logZ
            return logZ

    @property
    def mean(self) -> Array:
        """
        Returns probabilities. Axis is defined this way to accomodate non-trivial event_shapes.
        The nan_to_num call handles cases where 0's are added to arrays used to compute logits. This happens for
        broadcasting reasons, for example in MNLR_Bouchard
        """
        return jnp.nan_to_num(nn.softmax(self.logits, axis=tuple(range(-self.event_dim, 0))))

    @property
    def variance(self) -> Array:
        """
        Variance of the Mutlinomial distribution
        """
        return jnp.diag(self.mean) - self.mean @ self.mean.mT

    @property
    def log_mean(self) -> Array:
        """
        Computes the log mean of the distribution.
        """
        return self.logits - self.log_normalizer

    def log_likelihood(self, x: Array) -> Array:
        """
        Computes the log likelihood log p(x|\theta) of the distribution
        """
        return self.sum_events(x * (self.logits - self.log_normalizer))

    def statistics(self, x: Array) -> ArrayDict:
        """
        Computes the sufficient statistics T(x) = x
        """
        return ArrayDict(x=x)

    def expected_statistics(self) -> ArrayDict:
        """
        Computes the expected sufficient statistics <T(x)|<S(\theta)|eta, \nu>>
        """
        return ArrayDict(x=self.mean)

    def log_partition(self) -> Array:
        """
        Computes the logarithm of the partition function.
        """
        return (
            self.log_normalizer
        )  
        # caveat, not 100% sure that this is consistent with definition of residual as arising
        # from nat_params \dot T(x) + residual(z) = <log p(stuff)>

    def log_measure(self, x: Array) -> Array:
        """
        Computes the log measure of the distribution.
        """
        return 0.0

    def expected_log_measure(self) -> Array:
        """
        Computes the expected log base measure log phi(x) of the distribution under self.posterior_params
        """
        return 0.0

    def entropy(self) -> Array:
        """
        Computes the entropy of the distribution.
        """
        return -self.sum_events(self.mean * self.log_mean)

    def expected_x(self) -> Array:
        """
        Computes and returns the expected value of x: <x>

        Returns:
            Array: The expected value of x: <x>
                 (batch_shape + custom_event_shape + (dim, 1)).
        """
        return jnp.expand_dims(self.mean, -1)

    def expected_xx(self) -> Array:
        """
        Computes the expected outer product of x: <xxᵀ>.

        Returns:
            Array: The expected outer product of x: <xxᵀ>
                 (batch_shape + custom_event_shape + (dim, dim)).
        """
        return jnp.diag(self.mean)

    def params_from_statistics(self, stats: ArrayDict) -> ArrayDict:
        """
        Computes the inverse of `expected_statistics` \theta = mu^{-1}(<T(x)>) using self._expectations
        """
        return ArrayDict(logits=jnp.log(stats.x))

    def _update_cache(self):
        """
        Invoked whenever natural parameters or expectations are updated.
        """
        pass

    @multimethod
    def __mul__(self, other: Delta) -> Delta:
        """
        Overloads the * operator to combine the natural parameters of a exponential.Multinomial and exponential.Delta instance
        """
        return other.copy()

    @multimethod
    def __mul__(self, other: Distribution):
        """
        Overloads the * operator to combine the natural parameters of two exponential.Multinomial instances
        """
        # Check if the other instance is of the same class as self
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot multiply {type(self)} with {type(other)}")

        # combined_unnormalized_logits = combined_normalized_logits + logZ_individual # this computes the sum of the full, un-normalized logits
        nat_params_combined = ArrayDict(logits=self.logits + other.logits)

        # Sum the residual arrays
        if self.residual is not None and other.residual is not None:
            summed_residual = self.residual + other.residual
        elif self.residual is not None:
            summed_residual = self.residual
        elif other.residual is not None:
            summed_residual = other.residual
        else:
            summed_residual = None

        # Create a new instance with the combined natural parameters and summed residual
        return self.__class__(
            nat_params=nat_params_combined,
            input_logZ=(self.log_normalizer + other.log_normalizer),
            residual=summed_residual,
        )

    # def params_from_statistics(self, stats: ArrayDict) -> ArrayDict:
    #     """
    #     Computes the inverse of `expected_statistics` \theta = mu^{-1}(<T(x)>) using self._expectations
    #     """
    #     return ArrayDict(logits=jnp.log(stats.x / neg_log_residual))

    # def sample(self, key, shape=()) -> Array:
    #     """
    #     Samples from the distribution given \theta
    #     """
    #     return jr.multinomial(key, self.mean, shape=shape)
