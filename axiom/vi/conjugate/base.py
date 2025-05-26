from typing import Type, Optional, Union
from jaxtyping import Array

import jax.numpy as jnp
import jax.tree_util as jtu

from axiom.vi import Distribution, ExponentialFamily, ArrayDict, utils


class Conjugate(Distribution):
    """
    Base class for an exponential family probability distribution with a conjugate prior.

    The likelihood takes the form:

    .. math::
        p(x|θ) = φ(x) * exp(S(θ) ⋅ T(x) - A(S(θ)))

    where S(θ) are the natural parameters of the likelihood, φ(x) is the base measure function,
    T(x) is the likelihood sufficient statistic, and A(S(θ)) is the log partition function.

    The conjugate prior takes the form:

    .. math::
        p(θ|η₀, ν₀) = exp(S(θ) ⋅ η₀ - ν₀ A(S(θ)) - log Z(η₀, ν₀))

    where η₀, ν₀ are the natural parameters of the prior, and log Z(η₀, ν₀) is
    the prior log partition function.

    Given a dataset {x₁, x₂, ..., xₙ}, the likelihood and prior define a posterior over θ:

    .. math::
        p(θ|{x₁, x₂, ... xₙ}) = p(θ|η, v)
        η = η₀ + ∑_{n=1}^N T(xₙ)
        v = ν₀ + N

    where η and v are the natural parameters of the posterior.

    In situations where x is not observed but approximated by another probability distribution q(x),
    the η update rule simply becomes:

    .. math::
        η = η₀ + ∑_{n=1}^N <T(xₙ)>_{q(xₙ)}

    Inference requires specifying parameters on a per distribution basis:
        posterior_params = (η, v)
        prior_params = (η₀, ν₀)

    For each distribution, these are encoded in `posterior_params` and `prior_params` as an ArrayDict
    which contains the natural parameters of the posterior/prior distribution, respectively.
    The constructors for derived classes accept parameters in their canonical form θ,
    which are then converted to the natural parameters η, v.

    This class has a number of methods that must be implemented by derived classes:
        expected_likelihood_params       = <S(θ)|η, v>
        expected_log_likelihood(x)       = <log p(x|θ)>_{q(θ|η, v)}
                                         = log φ(x) + T(x) ⋅ <S(θ)>_{q(θ|η, v)} - <A(S(θ))>_{q(θ|η, v)}
        expected_posterior_statistics    = <S(θ)>_{q(θ|η, v)}, -<A(S(θ))>_{q(θ|η, v)}
        expected_log_partition           = <A(S(θ))|η, v>
        log_posterior_partition          = log Z(η,v)
        log_prior_partition              = log Z(η₀, ν₀)
        residual                         = A(<S(θ)|η, v>) - <A(S(θ))|η, v>
        kl_divergence                    = KL(q(θ|η, v), p(θ|η₀, ν₀))
        forward (Message)                = p(x|<S(θ)>_{q(θ|η, v)}), residual


    For likelihood_statistics, expected_likelihood_statistics, and expected_prior_statistics,
    the output should be an ArrayDict. The keys of this dictionary correspond to each natural parameter.
    During learning, the sufficient statistics are applied to the prior parameters to update the posterior parameters.
    The names of the sufficient statistics must match the names of the natural parameters.

    This distribution provides three update routes. `update_from_data` accepts data x and optionally a set of weights.
    From this data, it calculates the likelihood statistics T(x) before summing these (with optional weights)
    and applying the posterior updates. The routine `update_from_statistics` accepts the (potentially-weighted)
    summed statistics T(x) and applies the posterior updates. Finally, `update_from_probabilities` accepts some
    probability distribution p(x) and computes the relevant statistics.
    """

    _likelihood: ExponentialFamily
    _posterior_params: ArrayDict
    _prior_params: ArrayDict

    pytree_data_fields = ("_likelihood", "_posterior_params")
    pytree_aux_fields = ("_prior_params",)

    def __init__(
        self,
        default_event_dim: int,
        likelihood_cls: Type[ExponentialFamily],
        posterior_params: ArrayDict,
        prior_params: ArrayDict,
        batch_shape: tuple = (),
        event_shape: tuple = (),
    ):
        super().__init__(default_event_dim, batch_shape, event_shape)
        self._prior_params = self.to_natural_params(prior_params)
        self._posterior_params = self.to_natural_params(posterior_params)

        self._update_cache()

        likelihood_params = self.map_params_to_likelihood(
            self.expected_likelihood_params(), likelihood_cls=likelihood_cls
        )
        self._likelihood = likelihood_cls(likelihood_params, event_dim=self.event_dim)

    @property
    def likelihood(self) -> ExponentialFamily:
        return self._likelihood

    @likelihood.setter
    def likelihood(self, value: ExponentialFamily):
        self._likelihood = value

    @property
    def posterior_params(self) -> ArrayDict:
        return self._posterior_params

    @posterior_params.setter
    def posterior_params(self, value: ArrayDict):
        self._posterior_params = value
        self._update_cache()

    @property
    def prior_params(self) -> ArrayDict:
        return self._prior_params

    @prior_params.setter
    def prior_params(self, value: ArrayDict):
        self._prior_params = value
        self._update_cache()

    def expand(self, shape: tuple):
        """
        Expands parameters and prior parameters into a larger batch shape.
        The resulting self.shape will be equal to shape.
        """
        # TODO needs to be generalised using tree_flatten to pick out fields to expand
        assert shape[-self.batch_dim - self.event_dim :] == self.shape
        shape_diff = shape[: -self.batch_dim - self.event_dim]
        self.posterior_params = jtu.tree_map(lambda x: jnp.broadcast_to(x, shape_diff + x.shape), self.posterior_params)
        self.prior_params = jtu.tree_map(lambda x: jnp.broadcast_to(x, shape_diff + x.shape), self.prior_params)
        self.batch_shape = shape_diff + self.batch_shape
        self.batch_dim = len(self.batch_shape)

    def map_params_to_likelihood(self, params: ArrayDict, likelihood_cls: Type[ExponentialFamily] = None) -> ArrayDict:
        """
        Maps the natural parameters of the conjugate prior to the natural parameters of the likelihood
        """
        conjugate_to_lh_mapping = self._conjugate_to_likelihood_mapping(likelihood_cls=likelihood_cls)
        return utils.map_dict_names(params, name_mapping=conjugate_to_lh_mapping)

    def expected_likelihood_params(self) -> ArrayDict:
        """
        Returns the expected natural parameters of the likelihood with respect to the posterior over θ,
        parameterized by η and v, denoted as q(θ|η, v).

        .. math::
            <S(θ)>_{q(θ|η, v)}

        Returns:
            ArrayDict: A structure containing the expected natural parameters, with shapes matching
                    the likelihood's parameterization.
        """
        raise NotImplementedError

    def expected_log_likelihood(self, data: Union[Array, tuple[Array]]) -> Array:
        """
        Computes the expected log likelihood of the given data point(s) under the distribution.

        .. math::
            T(x) ⋅ <S(θ)>_{q(θ|η, v)} - <A(θ)>_{q(θ|η, v)}

        Args:
            x (Array): Data point(s) for which the log likelihood is computed.
                            Shape: (sample_shape, batch_shape, event_shape).

        Returns:
            Array: The expected log likelihood values for the given data points, with shape
                        (sample_shape, batch_shape).
        """

        x = data[0] if isinstance(data, tuple) else data

        counts_shape = self.get_sample_shape(x) + self.get_batch_shape(x)
        shape = counts_shape + (1,) * self.event_dim
        counts = jnp.ones(counts_shape)

        param_stats = self.map_stats_to_params(self.likelihood.statistics(data), counts)

        tx_dot_stheta_minus_A = utils.map_and_multiply(
            self.expected_posterior_statistics(), param_stats, self.default_event_dim
        )

        return self.sum_events(self._likelihood.log_measure(data) + tx_dot_stheta_minus_A)

    def expected_posterior_statistics(self) -> ArrayDict:
        """
        Computes the expected sufficient statistics of the posterior distribution's parameters η and v.

        .. math::
            (<S(θ)>_{q(θ|η, v)}, -<A(θ)>_{q(θ|η, v)})
        """
        raise NotImplementedError

    def expected_log_partition(self) -> ArrayDict:
        """
        Computes the expected log partition of the distribution under the variational distribution q(θ|η, v).
        """
        raise NotImplementedError

    def to_natural_params(self, params) -> ArrayDict:
        """ "Map canonical parameters to natural ones"""
        raise NotImplementedError

    def log_prior_partition(self) -> Array:
        """
        Computes the log partition function of the prior distribution, log Z(η₀, ν₀).

        Returns:
            Array: Scalar or array representing the log partition of the prior
        """
        raise NotImplementedError

    def log_posterior_partition(self) -> Array:
        """
        Computes the log partition function of the posterior distribution, log Z(η, v).

        Returns:
            Array: Scalar or array representing the log partition of the posterior
        """
        raise NotImplementedError

    def residual(self) -> Array:
        """
        Computes the residual, the difference between the log partition function evaluated at the expected
        sufficient statistics and the expected log partition function.

        .. math::
            A(<θ>_{q(θ|η, v)}) - <A(θ)>_{q(θ|η, v)}
        """
        raise NotImplementedError

    def kl_divergence(self) -> Array:
        """
        Computes the KL divergence between the prior and posterior distributions over θ.

        .. math::
            KL(q(θ|η, v), p(θ|η₀, ν₀))
        """

        log_qp = jtu.tree_map(lambda x, y: x - y, self.posterior_params, self.prior_params)
        expected_log_qp = utils.map_and_multiply(self.expected_posterior_statistics(), log_qp, self.default_event_dim)

        kl_div = self.log_prior_partition() - self.log_posterior_partition() + expected_log_qp
        return self.sum_events(kl_div)

    def variational_residual(self):
        raise NotImplementedError

    def variational_forward(self) -> ExponentialFamily:
        forward_message = self.likelihood.copy()
        forward_message.residual = self.variational_residual()
        return forward_message

    def statistics_dot_expected_params(self, x: Array) -> Array:
        """
        Computes the expected dot product between the sufficient statistics and the expected likelihood parameters.

        Specifically, calculates T(x) ⋅ <S(θ)> under the variational distribution q(θ|η, v).
        """
        # this works when we assume self.likelihood.nat_params = self.expected_likelihood_params()
        return self.likelihood.params_dot_statistics(x)

    def update_from_data(
        self,
        data: Union[Array, tuple],
        weights: Optional[Array] = None,
        lr: float = 1.0,
        beta: float = 0.0,
    ):
        """
        Updates the natural parameters given data.

        Args:
            data (Array): (sample_shape, batch_shape, event_shape)
                The data to update the parameters with.
            weights (Array): (sample_shape, batch_shape),
                The weights to apply to each sample. Default is None.
            lr (float): Learning rate for the update, default: 1.0
            beta (float): Batch decay for the update, default: 0.0.
        """

        # TODO: wip solution for when data is (x, y)
        # The other option is the shape methods optionally accept tuple data
        x = data[0] if isinstance(data, tuple) else data

        # TODO: should this be batch shape of data or batch shape of distribution?
        counts_shape = self.get_sample_shape(x) + self.get_batch_shape(x)
        shape = counts_shape + (1,) * self.event_dim
        counts = jnp.ones(counts_shape)
        sample_dims = self.get_sample_dims(x)

        weights = self.expand_event_dims(weights) if weights is not None else jnp.ones(shape)
        # weights = weights if weights is not None else jnp.ones(shape)

        likelihood_stats = self.likelihood.statistics(data)

        param_stats = self.map_stats_to_params(likelihood_stats, counts)
        summed_stats = self.sum_stats_over_samples(param_stats, weights, sample_dims)

        self.update_from_statistics(summed_stats, lr, beta)

    def update_from_statistics(self, summed_stats: ArrayDict, lr: float = 1.0, beta: float = 0.0):
        """
        Updates the natural parameters of posterior given likelihood statistics.

        This function applies the summed likelihood statistics T(x) to the prior parameters
        using a learning rate `lr` and some batch decay `beta`.

        The posterior parameters η and v are updated as:

        .. math::
            η = η₀ + Δη = η₀ + lr * Σ_n T(xₙ)
            v = ν₀ + Δν = ν₀ + lr * N

        where η and v are the natural parameters of the posterior, Δη and Δν are the adjustments from
        the likelihood statistics and learning rate, η₀ and ν₀ are the natural parameters of the prior,
        and N is the number of observations (summed through the batch dimensions).

        Args:
            summed_stats (ArrayDict): The likelihood statistics of the distribution, evaluations of sum T(x)
                                    for each natural parameter.
            lr (float): Learning rate for the update. Scalar value. Default is 1.0.
            beta (float): Batch decay for the update. Scalar value. Default is 0.0.
        """

        # Need tree version of assert statement of the form:
        #   assert(summed_stats[key].shape == self.posterior_params[key].shape for key in summed_stats.keys()
        # or something like that unless that is unncecessary because apply_add takes care of it...
        scaled_updates = jtu.tree_map(lambda x: lr * x, summed_stats)
        scaled_prior = jtu.tree_map(lambda x: lr * (1.0 - beta) * x, self.prior_params)
        posterior_past = jtu.tree_map(lambda x: (1.0 - lr * (1.0 - beta)) * x, self.posterior_params)
        updated_posterior_params = utils.apply_add(posterior_past, utils.apply_add(scaled_prior, scaled_updates))

        self.posterior_params = updated_posterior_params

        # this will have keys corresponding to natural parameters of the likelihood (e.g. alpha, beta)
        self.likelihood.nat_params = self.map_params_to_likelihood(self.expected_likelihood_params())

    def update_from_probabilities(
        self,
        data: Union[Distribution, tuple[Distribution]],
        weights: Optional[Array] = None,
        lr: float = 1.0,
        beta: float = 0.0,
    ):
        """
        Update distribution from probabilities
        """
        distribution = data[0] if isinstance(data, tuple) else data  # Deal with the case where we have multiple inputs

        counts_shape = self.get_sample_shape(distribution.mean) + self.get_batch_shape(distribution.mean)
        shape = counts_shape + (1,) * self.event_dim
        counts = jnp.ones(counts_shape)

        # Adapted from the conjugate base class
        sample_dims = self.get_sample_dims(distribution.mean)
        # list(range(len(counts_shape)))

        counts = jnp.ones(counts_shape)
        weights = self.expand_event_dims(weights) if weights is not None else jnp.ones(shape)
        # weights = weights if weights is not None else jnp.ones(shape)

        # If we have multiple inputs, the likelihood has to handle things
        distribution_stats = (
            distribution.expected_statistics()
            if not isinstance(data, tuple)
            else self.likelihood.stats_from_probs(data)
        )
        param_stats = self.map_stats_to_params(distribution_stats, counts)

        summed_stats = self.sum_stats_over_samples(param_stats, weights, sample_dims)

        self.update_from_statistics(summed_stats, lr, beta)

    def sum_stats_over_samples(self, stats: ArrayDict, weights: Array, sample_dims: list[int]) -> ArrayDict:
        """
        Sums over the sample dimensions of the statistics, which are nested in an arbitrary pytree structure
        """

        return jtu.tree_map(lambda leaf_array: (leaf_array * weights).sum(sample_dims), stats)

    def map_stats_to_params(self, likelihood_stats: ArrayDict, counts: Array) -> ArrayDict:
        """
        Maps keys of an ArrayDict of statistics to keys of an ArrayDict of natural parameters,
        and instantiates an ArrayDict with the same structure as the natural parameters,
        but whose leaves are evaluates of the sufficient statistics
        """
        # Ensure structure of eta matches structure of likelihood_stats
        stats_leaves, stats_treedef = jtu.tree_flatten(likelihood_stats)
        eta_treedef = jtu.tree_structure(self.posterior_params.eta)

        # @TODO: Write a tree_like util that tests whether trees are isomorphs
        assert len(eta_treedef.node_data()[1]) == len(stats_treedef.node_data()[1])

        # retrieve the mapping of natural parameters --> sufficient stats from the Conjugate class
        mapping = self._get_params_to_stats_mapping()

        def map_fn(key):
            """
            This function retrieves the evaluations of the sufficient statistics
            """
            return likelihood_stats.get(mapping.get(key, None), None)

        mapped_leaves = jtu.tree_map(map_fn, eta_treedef.node_data()[1])
        eta_stats = jtu.tree_unflatten(eta_treedef, mapped_leaves)

        nu_stats = jtu.tree_map(lambda x: self.expand_event_dims(counts), self.posterior_params.nu)

        return ArrayDict(eta=eta_stats, nu=nu_stats)

    def _get_params_to_stats_mapping(self):
        """
        Retrieve the mapping from the conjugate class, that maps the natural parameters to their associated
        sufficient statistics of the likelihood ($T(x)$).))"""
        conjugate_class = self.__class__
        mapping = getattr(conjugate_class, "params_to_tx", None)
        return mapping

    def _conjugate_to_likelihood_mapping(self, likelihood_cls: Type[ExponentialFamily] = None):
        """
        Returns a mapping from the natural parameters of the conjugate prior (e.g., `eta1`, `eta2`, ...)
        to the natural parameters of the likelihood (with distribution-specific names).
        In case the likelihood class where the mapping is defined is not specified, the mapping is retrieved from the likelihood class
        that is stored in `self`.
        """
        conjugate_mapping = self._get_params_to_stats_mapping()

        if likelihood_cls is None:
            likelihood_mapping = self.likelihood._get_params_to_stats_mapping()
        else:  # this is needed during the __init__ when self.likelihood is not yet initialized
            likelihood_mapping = getattr(likelihood_cls, "params_to_tx", None)

        conjugate_to_lh_mapping = {
            key_a: key_b
            for key_a, value_a in conjugate_mapping.items()
            for key_b, value_b in likelihood_mapping.items()
            if value_a == value_b
        }

        return conjugate_to_lh_mapping

    def _update_cache(self):
        """
        Called whenever posterior parameters are updated.
        """
        pass
