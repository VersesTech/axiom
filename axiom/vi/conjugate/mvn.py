from typing import Optional
from jaxtyping import Array

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
from jaxtyping import Array

from ..distribution import Distribution
from ..exponential import MultivariateNormal as MultivariateNormalLikelihood
from .base import Conjugate
from ..utils import mvgammaln, mvdigamma, params_to_tx, ArrayDict, tree_copy
from ..utils import inv_and_logdet, bdot

DEFAULT_EVENT_DIM = 2


@jax.jit
def clip_small_values(x):
    return jnp.where((-1e-3 < x) & (x < 1e-3), 0.0, x)


@params_to_tx({"eta_1": "x", "eta_2": "minus_half_xxT"})
class MultivariateNormal(Conjugate):
    """
    Multivariate normal distribution with a Normal-Inverse-Wishart conjugate prior.
    The likelihood function is given by the normal distribution N(μ, Σ), where:

    .. math::
        log p(x) = -0.5 * (x - μ)ᵀΣ⁻¹(x - μ) + 0.5 * log |Σ⁻¹| - 0.5 * D * log(2π)

    where:
        - :math: μ is the mean vector,
        - :math: Σ is the covariance matrix,
        - :math: Σ⁻¹ is the inverse covariance matrix (also known as the precision matrix),
        - :math: D is the dimensionality of the multivariate normal distribution and
        - :math: x is a column vector of size D.

    The conjugate prior for this distribution is the Normal-Inverse-Wishart distribution.
    This can be expressed in exponential form as:

    .. math::
        log p(μ, Σ⁻¹ | m, κ, n, U)
            = ⟨Σ⁻¹μ, -0.5Σ⁻¹⟩ ⋅ ⟨κm, U⁻¹ + κmmᵀ⟩
            - ⟨κ, n - D - 1⟩ ⋅ ⟨0.5 * μᵀΣ⁻¹μ, 0.5 * log |Σ⁻¹|⟩

    where:
        - :math: m is the mean of μ (called `mean` below)
        - :math: κ is the concentration parameter of the distribution over the mean (i.e., μ)
        - :math: n is the number of degrees of freedom of the Wishart
        - :math: U is the scale matrix of the Wishart

    This implies that:
        - :math: Σ⁻¹μ, -0.5Σ⁻¹ are the sufficient statistics of the likelihood function
        - :math: ⟨κm, U⁻¹ + κmmᵀ⟩ are the natural parameters η of the likelihood prior
        - :math: ⟨κ, n - D - 1⟩ are the prior degrees of freedom v of the conjugate prior
        - :math: ⟨0.5 * μᵀΣ⁻¹μ, 0.5 * log |Σ⁻¹|⟩ make up the log partition of the likelihood A(η)

    In the class below, we use the following definitions of the natural parameters η:
        - :math: eta_1 = κm
        - :math: eta_2 = U⁻¹ + κmmᵀ

    And for the degrees of freedom v:
        - :math: nu_1 = κ
        - :math: nu_2 = n - D - 1

    The associated sufficient statistics are:
        - :math: eta_1 = Σ⁻¹μ
        - :math: eta_2 = -0.5Σ⁻¹
        - :math: nu_1 = 0.5 * μᵀΣ⁻¹μ
        - :math: nu_2 = 0.5 * log |Σ⁻¹|

    The posterior distribution of the parameters given the data is also a Normal-Inverse-Wishart distribution.
    """

    _u: Array
    _logdet_inv_u: Array
    _prior_logdet_inv_u: Array

    pytree_data_fields = ("_u", "_logdet_inv_u", "_prior_logdet_inv_u")
    pytree_aux_fields = ("fixed_precision",)

    def __init__(
        self,
        params: Optional[ArrayDict] = None,
        prior_params: Optional[ArrayDict] = None,
        batch_shape: Optional[tuple] = None,
        event_shape: Optional[tuple] = None,
        event_dim: Optional[int] = DEFAULT_EVENT_DIM,
        fixed_precision: bool = False,
        scale: float = 1.0,  # scale parameter for the prior scale matrix (U = scale^2 * I)
        dof_offset: float = 0.0,  # offset for the prior degrees of freedom (n = 1 + dim + dof_offset) If dof_offset = 1.0 then expected_sigma() is undefined, so one can use dof_offset=1.0 if expected_sigma() is needed
        init_key: Optional[
            Array
        ] = None,  # initialization key for sampling the posterior mean from the prior
    ):
        if event_shape is not None:
            assert (
                len(event_shape) == event_dim
            ), "event_shape must have length equal to event_dim"

        if prior_params is None:
            assert dof_offset >= 0.0, "dof_offset must be non-negative"
            prior_params = self.init_default_params(
                batch_shape, event_shape, scale, dof_offset, DEFAULT_EVENT_DIM
            )
        if params is None:
            # set the values of posterior parameters based on
            # those of the prior (mean is sampled from the prior mean)
            init_key = jr.PRNGKey(0) if init_key is None else init_key
            # sample a posterior mean centered on the prior mean
            params = {}
            for k, v in prior_params.items():
                if k == "mean":
                    params[k] = v + jr.normal(init_key, v.shape)
                else:
                    params[k] = v

            params = ArrayDict(**params)

        inferred_batch_shape, inferred_event_shape = self.infer_shapes(
            params.mean, event_dim
        )
        batch_shape = batch_shape if batch_shape is not None else inferred_batch_shape
        event_shape = event_shape if event_shape is not None else inferred_event_shape

        self.fixed_precision = fixed_precision

        super().__init__(
            DEFAULT_EVENT_DIM,
            MultivariateNormalLikelihood,
            params,
            prior_params,
            batch_shape,
            event_shape,
        )

        _prior_logdet_inv_u = jnp.linalg.slogdet(self.prior_inv_u)[1]
        self._prior_logdet_inv_u = self.expand_default_event_dims(_prior_logdet_inv_u)

    @staticmethod
    def init_default_params(
        batch_shape,
        event_shape,
        scale: float = 1.0,
        dof_offset: float = 0.0,
        default_event_dim=2,
    ) -> ArrayDict:
        """Initialize the default canonical parameters of the distribution."""

        dim = event_shape[-default_event_dim]

        mean = jnp.zeros(batch_shape + event_shape)
        kappa = jnp.full(batch_shape + event_shape[:-default_event_dim] + (1, 1), 1.0)
        u = (scale**2) * jnp.broadcast_to(
            jnp.eye(dim),
            batch_shape + event_shape[:-default_event_dim] + (dim, dim),
        )
        n = jnp.full(
            batch_shape + event_shape[:-default_event_dim] + (1, 1),
            1.0 + dim + dof_offset,
        )  # NOTE: defining n = 1.0 + dim means expected_sigma() is undefined.
        return ArrayDict(mean=mean, kappa=kappa, u=u, n=n)

    @property
    def sqrt_diag_norm(self) -> jnp.ndarray:
        diag = jnp.diagonal(
            jnp.abs(self.posterior_params.eta.eta_2), axis1=-2, axis2=-1
        )
        diag = jnp.maximum(diag, 1.0)
        return jnp.sqrt(diag)

    @property
    def norm(self) -> jnp.ndarray:
        sqrt_diag = self.sqrt_diag_norm
        norm = bdot(sqrt_diag[..., :, None], sqrt_diag[..., None, :])
        return norm

    @property
    def mean(self):
        """
        Property to access the mean vector of the posterior normal inverse wishart distribution
        """
        return self.posterior_params.eta.eta_1 / self.posterior_params.nu.nu_1

    @property
    def prior_mean(self):
        """
        Property to access the mean vector of the prior normal inverse wishart distribution
        """
        return self.prior_params.eta.eta_1 / self.prior_params.nu.nu_1

    @property
    def kappa(self):
        """
        Property to access the strength of the posterior normal inverse wishart distribution
        """
        return self.posterior_params.nu.nu_1

    @property
    def prior_kappa(self):
        """
        Property to access the strength of the prior normal inverse wishart distribution
        """
        return self.prior_params.nu.nu_1

    @property
    def n(self):
        """
        Property to access the degrees of freedom of the posterior normal inverse wishart distribution
        """
        if self.fixed_precision:
            return self.prior_n
        else:
            return self.posterior_params.nu.nu_2 + self.dim

    @property
    def prior_n(self):
        """
        Property to access the degrees of freedom of the prior normal inverse wishart distribution
        """
        return self.prior_params.nu.nu_2 + self.dim

    @property
    def _scaled_inv_u(self):
        """
        Property to access the scaled inverse scale matrix of the posterior normal inverse wishart distribution
        """
        norm = self.norm
        sqrt_diag_norm = self.sqrt_diag_norm
        scaled_eta_2 = (
            self.posterior_params.eta.eta_2 / norm
        )  # shape batch_shape + (d, d)
        kappa = self.posterior_params.nu.nu_1
        scaled_eta_1 = (
            self.posterior_params.eta.eta_1 / sqrt_diag_norm[..., None]
        )  # shape batch_shape + (d, 1)
        scaled_inv_u = -2 * scaled_eta_2 - (1.0 / kappa) * bdot(
            scaled_eta_1, scaled_eta_1.mT
        )
        clipped_diag = jnp.diagonal(scaled_inv_u, axis1=-2, axis2=-1).clip(min=1e-3)
        scaled_inv_u = scaled_inv_u.at[
            ..., jnp.arange(self.dim), jnp.arange(self.dim)
        ].set(clipped_diag)
        return clip_small_values(scaled_inv_u)

    @property
    def inv_u(self):
        """
        Property to access the inverse scale matrix of the posterior normal inverse wishart distribution
        """
        if self.fixed_precision:
            return self.prior_inv_u
        else:
            return self._scaled_inv_u * self.norm

    @property
    def prior_inv_u(self):
        """
        Property to access the inverse scale matrix of the prior normal inverse wishart distribution
        """
        return -2 * self.prior_params.eta.eta_2 - (
            1.0 / self.prior_params.nu.nu_1
        ) * bdot(self.prior_params.eta.eta_1, self.prior_params.eta.eta_1.mT)

    @property
    def u(self):
        """
        Property to access the scale matrix of the posterior normal inverse wishart distribution
        """
        if self._u is None:
            if self.fixed_precision:
                self._u, self._logdet_inv_u = inv_and_logdet(self.prior_inv_u)
            else:
                _scaled_u, _logdet_inv_u_scaled = inv_and_logdet(self._scaled_inv_u)
                self._u = _scaled_u / self.norm
                log_diag = jnp.log(jnp.diagonal(self.norm, axis1=-1, axis2=-2)).sum(
                    axis=-1
                )

                self._logdet_inv_u = _logdet_inv_u_scaled + jnp.expand_dims(
                    log_diag, (-1, -2)
                )
        return self._u

    @property
    def logdet_inv_u(self):
        """
        Property to access the logdet of the inverse scale matrix of the posterior
        """
        if self._logdet_inv_u is None:
            _logdet_inv_u_scaled = inv_and_logdet(
                self._scaled_inv_u, return_inverse=False
            )
            log_diag = jnp.log(jnp.diagonal(self.norm, axis1=-1, axis2=-2)).sum(axis=-1)
            self._logdet_inv_u = _logdet_inv_u_scaled + jnp.expand_dims(
                log_diag, (-1, -2)
            )

        return self._logdet_inv_u

    @property
    def prior_logdet_inv_u(self):
        """
        Property to access the logdet of the inverse scale matrix of the prior
        """
        if self._prior_logdet_inv_u is None:
            self._prior_logdet_inv_u = inv_and_logdet(
                self.prior_inv_u, return_inverse=False
            )
        return self._prior_logdet_inv_u

    def to_natural_params(self, params: ArrayDict) -> ArrayDict:
        """
        Converts canonical parameters to natural parameters
        """
        eta_1 = params.mean * params.kappa
        eta_2 = -0.5 * (
            inv_and_logdet(params.u, return_logdet=False) + bdot(eta_1, params.mean.mT)
        )
        nu_1 = params.kappa
        nu_2 = params.n - self.dim
        eta = ArrayDict(eta_1=eta_1, eta_2=eta_2)
        nu = ArrayDict(nu_1=nu_1, nu_2=nu_2)
        return ArrayDict(eta=eta, nu=nu)

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
        inv_sigma_mu = self.expected_inv_sigma_mu()
        inv_sigma = self.expected_inv_sigma()
        return ArrayDict(eta_1=inv_sigma_mu, eta_2=inv_sigma)

    def expected_posterior_statistics(self) -> ArrayDict:
        """
        Computes the expected sufficient statistics of the posterior distribution's parameters η and v.

        .. math::
            (<S(θ)>_{q(θ|η, v)}, -<A(θ)>_{q(θ|η, v)})

        N.B. the negation before the return <A(θ)>_{q(θ|η, v)} is done in order
        to make it ready for computing the dot products that help with things like `expected_log_likelihood`
        """
        eta_stats = self.expected_likelihood_params()
        nu_stats = jtu.tree_map(lambda x: -x, self.expected_log_partition())
        return ArrayDict(eta=eta_stats, nu=nu_stats)

    def expected_log_partition(self) -> Array:
        """
        Computes the log partition A(θ) of the likelihood expected under the variational distribution,
        i.e., <A(θ)>_{q(θ|η, v)}
        """
        nu1_term = 0.5 * self.expected_mu_inv_sigma_mu()
        nu2_term = -0.5 * self.expected_logdet_inv_sigma()
        return ArrayDict(nu_1=nu1_term, nu_2=nu2_term)

    def log_prior_partition(self) -> Array:
        """
        Computes the log partition function of the prior distribution, log Z(η₀, ν₀).
        """
        return self._log_partition(
            self.prior_kappa, self.prior_n, self.prior_logdet_inv_u
        )

    def log_posterior_partition(self) -> Array:
        """
        Computes the log partition function of the posterior distribution, log Z(η, v).
        """
        return self._log_partition(self.kappa, self.n, self.logdet_inv_u)

    def _log_partition(self, kappa: Array, n: Array, logdet_inv_u: Array) -> Array:
        half_dim = 0.5 * self.dim
        term_1 = -half_dim * jnp.log(kappa)
        # by negating this term, we make it equal to logdet_u, because log |A⁻¹| = -log |A|
        term_2 = -0.5 * n * logdet_inv_u
        term_3 = half_dim * (jnp.log(2 * jnp.pi) + n * jnp.log(2))
        term_4 = mvgammaln(n / 2.0, self.dim)
        return term_1 + term_2 + term_3 + term_4

    def expected_logdet_inv_sigma(self) -> Array:
        return (
            self.dim * jnp.log(2)
            - self.logdet_inv_u
            + mvdigamma(self.n / 2.0, self.dim)
        )

    def logdet_expected_inv_sigma(self):
        return -self.logdet_inv_u + self.dim * jnp.log(self.n)

    def variational_residual(self):
        return 0.5 * (
            self.dim * (jnp.log(2) - jnp.log(self.n) - 1.0 / self.kappa)
            + mvdigamma(self.n / 2.0, self.dim)
        ).squeeze((-2, -1))

    def collapsed_residual(self):
        return self.variational_residual()

    def update_from_probabilities(
        self, pX: Distribution, weights: Optional[Array] = None, **kwargs
    ):
        """
        Update the parameters of the distribution given the expected sufficient statistics.

        Args:
            expected_sufficient_statistics (ArrayDict): Expected sufficient statistics of the distribution.
        """
        sample_shape = pX.shape[: -self.event_dim - self.batch_dim]
        sample_dims = tuple(range(len(sample_shape)))

        if weights is None:
            SEx = pX.expected_x().sum(sample_dims)
            SExx = pX.expected_xx().sum(sample_dims)
            N = jnp.broadcast_to(
                jnp.prod(jnp.array(sample_shape)), SEx.shape[:-2] + (1, 1)
            )
        else:
            weights = self.expand_event_dims(weights)
            SEx = (weights * pX.expected_x()).sum(sample_dims)
            SExx = (weights * pX.expected_xx()).sum(sample_dims)
            N = weights.sum(sample_dims)

        summed_stats = ArrayDict(
            eta=ArrayDict(eta_1=SEx, eta_2=-0.5 * SExx),
            nu=ArrayDict(nu_1=N, nu_2=N),
        )
        if "lr" in kwargs:
            lr = kwargs["lr"]
        else:
            lr = 1.0
        if "beta" in kwargs:
            beta = kwargs["beta"]
        else:
            beta = 0.0
        self.update_from_statistics(summed_stats, lr=lr, beta=beta)

    def expected_inv_sigma(self) -> Array:
        """
        Compute the expected inverse covariance matrix.

        .. math::
            E[Σ⁻¹] = nU

        Returns:
            Array: Expected inverse covariance matrix.
        """
        return self.u * self.n

    def expected_inv_sigma_mu(self) -> Array:
        """
        Compute the expected inverse covariance matrix times the mean vector.

        .. math::
            E[Σ⁻¹μ] = κUM

        Returns:
            Array: Expected inverse covariance matrix times the mean vector.
        """
        return bdot(self.expected_inv_sigma(), self.mean)

    def expected_sigma(self) -> Array:
        """
        Compute the expected covariance matrix.

        .. math::
            E[Σ] = U⁻¹ / (n - d - 1)

        Returns:
            Array: Expected covariance matrix.
        """
        return self.inv_u / (self.n - self.dim - 1)

    def inv_expected_inv_sigma(self) -> Array:
        return self.inv_u / self.n

    def expected_mu_inv_sigma_mu(self) -> Array:
        """
        Compute the expected mean vector times the inverse covariance matrix times the mean vector.

        .. math::
            E[μᵀΣ⁻¹μ] = Mᵀ(κU)M + d/κ

        Returns:
            Array: Expected mean vector times the inverse covariance matrix times the mean vector.
        """
        return (
            bdot(self.mean.mT, bdot(self.expected_inv_sigma(), self.mean))
            + self.dim / self.kappa
        )

    def expected_xx(self) -> Array:
        """
        Compute the expected outer product of the data points.

        .. math::
            E[xxᵀ] = Σ + μμᵀ

        Returns:
            Array: Expected outer product of the data points.
        """
        return self.expected_sigma() + bdot(self.mean, self.mean.mT)

    def _update_cache(self):
        """
        Update the scale matrix and logdet inverse scale matrix.
        """
        if self.fixed_precision:
            self._u, self._logdet_inv_u = inv_and_logdet(self.prior_inv_u)
        else:
            norm = self.norm
            _scaled_u, _logdet_inv_u_scaled = inv_and_logdet(self._scaled_inv_u)
            self._u = _scaled_u / norm

            log_diag = jnp.log(jnp.diagonal(self.norm, axis1=-1, axis2=-2)).sum(axis=-1)
            self._logdet_inv_u = _logdet_inv_u_scaled + jnp.expand_dims(
                log_diag, (-1, -2)
            )

    def _kl_divergence(self) -> Array:
        """
        Computes the KL divergence between the prior and posterior distributions over θ.

        .. math::
            KL(q(θ|η, v), p(θ|η₀, ν₀))
        """
        kl = (
            0.5
            * (
                self.prior_kappa / self.kappa
                - 1
                + jnp.log(self.kappa / self.prior_kappa)
            )
            * self.dim
        )
        pred_error = self.mean - self.prior_mean
        kl = kl + 0.5 * self.prior_kappa * bdot(
            pred_error.mT, bdot(self.expected_inv_sigma(), pred_error)
        )
        kl = kl + self.kl_divergence_wishart()
        return self.sum_events(kl)

    def kl_divergence_wishart(self) -> Array:
        """
        Compute the KL divergence between the posterior and prior wishart distributions.

        Returns:
            Array: KL divergence between the posterior and prior distributions.
                (batch_shape)
        """
        kl = self.prior_n / 2.0 * (self.logdet_inv_u - self.prior_logdet_inv_u)
        kl = kl + self.n / 2.0 * (self.prior_inv_u * self.u).sum(
            (-2, -1), keepdims=True
        )
        kl = kl - self.n * self.dim / 2.0
        kl = (
            kl
            + mvgammaln(self.prior_n / 2.0, self.dim)
            - mvgammaln(self.n / 2.0, self.dim)
            + (self.n - self.prior_n) / 2.0 * mvdigamma(self.n / 2.0, self.dim)
        )
        return kl

    def _expected_log_likelihood(self, x: Array) -> Array:
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

        diff = x - self.mean
        tx_dot_stheta = -0.5 * bdot(diff.mT, bdot(self.expected_inv_sigma(), diff))
        atheta_1 = -0.5 * self.dim / self.kappa
        atheta_2 = 0.5 * self.expected_logdet_inv_sigma()

        log_base_measure = -0.5 * self.dim * jnp.log(2 * jnp.pi)

        # tx_dot_stheta = tx_stheta_1 + tx_stheta_2
        negative_expected_atheta = atheta_1 + atheta_2

        return self.sum_events(
            log_base_measure + tx_dot_stheta + negative_expected_atheta
        )

    def expected_log_likelihood(self, data: Array) -> Array:
        return self._expected_log_likelihood(data)

    def average_energy(self, x: Distribution):
        r"""
        Computes the average energy term of the distribution, aka
        .. math::
            -\int q(x) q(\mu,\Sigma^{-1}) log p(x | mu, \Sigma^{-1}) dx dmu d\Sigma^{-1}

        If we rewrite in exponential family form and noting that \log[ \phi(x)] = 0, we get
        .. math::

            = <T(x)⋅ <S(θ)>_{q(θ|η, v)} - <A(θ)>_{q(θ|η, v)} >_{q(x)}
            = <T(x)>_{q(x)} ⋅ <S(θ)>_{q(θ|η, v)} - <A(θ)>_{q(θ|η, v)}

        where the last line follows from separating out terms that depend on x. Useful for working
        with mixture when the inputs are probability distributions, q(x).

        """

        expected_x = x.expected_x()
        expected_xx = x.expected_xx()
        sigma = expected_xx - bdot(expected_x, expected_x.mT)
        diff = expected_x - self.mean

        exp_inv_sigma = self.expected_inv_sigma()

        energy = -0.5 * jnp.sum(exp_inv_sigma * sigma, (-2, -1), keepdims=True)
        energy -= 0.5 * bdot(diff.mT, bdot(exp_inv_sigma, diff))
        # energy += jnp.sum(expected_x * self.expected_inv_sigma_mu(), -2, keepdims=True)
        energy -= 0.5 * self.dim / self.kappa
        energy += 0.5 * self.expected_logdet_inv_sigma()
        energy -= 0.5 * self.dim * jnp.log(2 * jnp.pi)

        return self.sum_events(energy)


if __name__ == "__main__":
    batch_shape = (4,)
    dim = 2
    numsamps = 100

    import jax.random as jr

    key = jr.PRNGKey(0)
    prior_mu = jnp.zeros(batch_shape + (dim, 1))
    kappa = jnp.ones(batch_shape + (1, 1))
    u = jnp.broadcast_to(jnp.eye(dim), batch_shape + (dim, dim))
    n = jnp.full(batch_shape + (1, 1), 2.0 + dim)

    # sample a posterior mean centered on the prior mean
    mu = prior_mu + jr.normal(key, batch_shape + (dim, 1))
    params = ArrayDict(mean=mu, kappa=kappa, u=u, n=n)
    prior_params = ArrayDict(mean=prior_mu, kappa=kappa, u=u, n=n)

    mvn = MultivariateNormal(params, prior_params)

    from optimus.exponential.mvn import MultivariateNormal as expmvn

    inv_sigma_mu = jr.normal(key, (numsamps,) + batch_shape + (dim, 1))

    pX = expmvn(ArrayDict(inv_sigma_mu=inv_sigma_mu, inv_sigma=jnp.eye(2)))
    mvn.update_from_probabilities(pX, None)

    from matplotlib import pyplot as plt

    plt.scatter(inv_sigma_mu.mean(0).squeeze(), mvn.mean.squeeze())
