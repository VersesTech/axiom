from typing import Optional, Union, Tuple
from jaxtyping import Array
from multimethod import multimethod

from jax import numpy as jnp
import jax.tree_util as jtu

from axiom.vi.utils import map_and_multiply, sum_pytrees, ArrayDict
from axiom.vi.distribution import Distribution, Delta


class ExponentialFamily(Distribution):
    """
    Base class for exponential family probability distributions without conjugate priors.
    This class will serve two related roles:
        1) As likelihoods for conjugate exponential families (see `Conjugate` in `conjugate.base`)
        2) As messages in a variational message passing framework.

    The exponential family distribution is defined as:

    .. math::
        p(x|θ) = φ(x) * exp(S(θ) ⋅ T(x) - A(S(θ)))

    where x is data, S(θ) are the natural parameters, φ(x) is the measure function,
    T(x) are the sufficient statistics, and A(S(θ)) is the log partition function.
    For convienence, we depart from the usual convention and include in the log
    partition function and terms typically included in the measure function that
    are independent of x.

    This class stores either the natural parameters S(θ) in `self.nat_params`, or the expected statistics
    <E[T(x) | S(θ)]> in `self.expectations`. The constructor accepts either the natural parameters S(θ)
    or the expected statistics <E[T(x) | S(θ)]>. Note that when constructing an instance of `ExponentialFamily`
    from within a `Conjugate` class, the values of S(θ) may in fact correspond to the expected value of
    S(θ) under the posterior/prior distribution of the conjugate distribution, i.e.:

    .. math::
        <E[S(θ)]>_{q} = ∫ S(θ) q(θ|v, η) dθ

    where v, η are the natural parameters of the conjugate posterior (prior) distribution.

    Each derived `ExponentialFamily` class implements the following methods:
        log_likelihood(x)                     = log p(x|θ)
        natural_params(x)                     = S(θ)
        measure(x)                            = φ(x)
        statistics(x)                         = T(x)
        log_partition()                       = A(S(θ))
        expected_statistics()                 = <E[T(x)|S(θ)]> = μ_T(θ)
        entropy()                             = -<φ(x)|S(θ)> - <E[T(x)|S(θ)]> ⋅ θ + A(S(θ))
        params_from_statistics(<E[T(x)]>)     = S(θ) = μ_T⁻¹(<E[T(x)]>)

    where S(θ) can be obtained via the property `nat_params` and μ_T(θ) via the property
    `expectations`. As calculating the expected statistics μ_T(θ) can be costly to compute on the fly,
    we introduce a dynamic data-field `expectations` to cache computational primitives (intermediate quantities).

    The `nat_params` and `expectations` are ArrayDicts (PyTrees) that store the natural parameters and
    expected statistics. Each class can include a `params_to_tx` decorator that maps the natural parameters
    keys to the sufficient statistics keys.
    """

    _nat_params: ArrayDict
    _expectations: ArrayDict
    pytree_data_fields = ("_nat_params", "_expectations", "_residual")
    pytree_aux_fields = ("default_event_dim",)

    def __init__(
        self,
        default_event_dim: int,
        batch_shape: tuple[int],
        event_shape: tuple[int],
        nat_params: Optional[ArrayDict] = None,
        expectations: Optional[ArrayDict] = None,
        residual: Optional[Array] = None,
    ):
        "Must provide one of nat_params or expectations."
        assert nat_params is not None or expectations is not None
        super().__init__(default_event_dim, batch_shape, event_shape)
        self._nat_params = nat_params
        self._expectations = expectations
        self._residual = residual if residual is not None else jnp.empty(batch_shape)

    @property
    def nat_params(self) -> ArrayDict:
        if self._nat_params is None:
            self._nat_params = self.params_from_statistics(self.expectations)
        return self._nat_params

    @nat_params.setter
    def nat_params(self, value: ArrayDict):
        self._nat_params = value
        self._update_cache()

    @property
    def expectations(self) -> ArrayDict:
        if self._expectations is None:
            # TODO: self._expectations = expected_statistics(self, self.nat_params) # [fixme]
            self._expectations = self.expected_statistics()
        return self._expectations

    @expectations.setter
    def expectations(self, value: ArrayDict):
        self._expectations = value
        self._update_cache()

    @property
    def residual(self) -> Optional[Array]:
        if self._residual is None:
            self._residual = 0.0
        return self._residual

    @residual.setter
    def residual(self, value: Optional[Array] = None):
        self._residual = value

    def expand(self, shape: tuple):
        """
        Expands parameters and prior parameters into a larger batch shape.
        The resulting self.shape will be equal to shape.
        """
        # TODO needs to be generalised using tree_flatten to pick out fields to expand # [fixme]
        assert shape[-self.batch_dim - self.event_dim :] == self.shape
        shape_diff = shape[: -self.batch_dim - self.event_dim]
        self.nat_params = jtu.tree_map(lambda x: jnp.broadcast_to(x, shape_diff + x.shape), self.nat_params)
        self.batch_shape = shape_diff + self.batch_shape
        self.batch_dim = len(self.batch_shape)
        return self

    def log_likelihood(self, x: Array) -> Array:
        """Computes the log likelihood of the distribution given the data.

        Computes the log likelihood log p(x|θ) given the parameters of the distribution:

        .. math::
            log p(x|θ) = S(θ) ⋅ T(x) - A(S(θ))

        Args:
            x (Array): The data for which the log likelihood should be computed.
                            (sample_shape + batch_shape + event_shape)

        Returns:
            Array: The computed log likelihood for each sample.
                        (sample_shape + batch_shape)

        """
        # TODO: Some distribution also have measure term, how is this added here? # [fixme]
        probs = self.params_dot_statistics(x) - self.log_partition()
        return self.sum_events(probs)

    def statistics(self, x: Array) -> ArrayDict:
        """
        Computes the sufficient statistics T(x) of the likelihood distribution.
        """
        raise NotImplementedError

    def expected_statistics(self) -> ArrayDict:
        """
        Computes the expected sufficient statistics <E[T(x)|S(θ)]>
        """
        raise NotImplementedError

    def log_partition(self) -> Array:
        """
        Computes the log partition of the distribution A(S(θ))
        """
        raise NotImplementedError

    def expected_log_measure(self) -> Array:
        """
        Computes the expected log measure of the distribution <log φ(x)|S(θ)>
        """
        raise NotImplementedError

    def entropy(self) -> Array:
        """
        Computes the entropy of the distribution.

        .. math::
            H(p) = - <T(x)|S(θ))> ⋅ S(θ) + A(S(θ))

        Returns:
            Array: The entropy of the distribution
                        (batch_shape, )
        """
        entropy = -self.params_dot_expected_statistics() + self.log_partition() - self.expected_log_measure()
        return self.sum_events(entropy)

    def sample(self, key, shape: tuple) -> Array:
        """
        Sample x from the distribution given θ
        """
        raise NotImplementedError

    def params_from_statistics(self, stats: ArrayDict) -> ArrayDict:
        """
        Computes the inverse of `expected_statistics` S(θ) = μ_T⁻¹(<E[T(x)]>)
        """
        raise NotImplementedError

    def params_dot_statistics(self, x: Array) -> Array:
        """
        Computes the dot product of the natural parameters and the sufficient statistics
        """
        mapping = self._get_params_to_stats_mapping()
        return map_and_multiply(self.nat_params, self.statistics(x), self.default_event_dim, mapping)

    def params_dot_expected_statistics(self) -> Array:
        """
        Computes the dot product of the natural parameters and the expected sufficient statistics
        """
        mapping = self._get_params_to_stats_mapping()
        return map_and_multiply(self.nat_params, self.expectations, self.default_event_dim, mapping)

    def combine(self, others: Union["ExponentialFamily", Tuple["ExponentialFamily"]]) -> "ExponentialFamily":
        """
        Combine the natural parameters of this instance with those of other instances.
        """
        # Ensure 'others' is a tuple for consistent processing
        if not isinstance(others, tuple):
            others = (others,)

        # Check if all instances are of the same class as self
        for other in others:
            if not isinstance(other, self.__class__):
                raise ValueError("All instances must be of type {}".format(self.__class__.__name__))

        # Extract the natural parameters from other instances
        nat_params_others = [other.nat_params for other in others]

        # Combine the natural parameters
        nat_params_combined = sum_pytrees(self.nat_params, *nat_params_others)

        # Create a new instance with the combined natural parameters
        new_instance = self.__class__(nat_params=nat_params_combined)
        return new_instance

    @multimethod
    def __mul__(self, other: Delta):
        """
        Overloads the + operator to combine the natural parameters of two instances.
        """
        return other.copy()

    @multimethod
    def __mul__(self, other: Distribution):
        """
        Overloads the * operator to combine the natural parameters of two instances.
        """

        # Check if the other instance is of the same class as self
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot multiply {type(self)} with {type(other)}")

        # Combine the natural parameters
        nat_params_combined = sum_pytrees(self.nat_params, other.nat_params)

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
        return self.__class__(nat_params=nat_params_combined, residual=summed_residual)

    def _validate_nat_params(self, nat_params: ArrayDict):
        """
        Validates the natural parameters.
        """
        mapping = self._get_params_to_stats_mapping()
        assert (
            mapping.keys() == nat_params.keys()
        ), f"Invalid natural parameters. Expected {mapping.keys()}, got {nat_params.keys()}"

        for k, v in nat_params.items():
            assert len(v.shape) >= self.default_event_dim, f"Invalid shape for natural parameter {k}"

    def _update_cache(self):
        """
        Invoked whenever natural parameters or expectations are updated.
        """
        pass

    def _get_params_to_stats_mapping(self):
        exponential_cls = self.__class__
        mapping = getattr(exponential_cls, "params_to_tx", None)
        return mapping
