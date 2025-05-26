from typing import Optional, Tuple
from jaxtyping import Array

from jax.lax import lgamma
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util
from jax.scipy import special as jsp
from jax.numpy import expand_dims as expand

from axiom.vi.distribution import Distribution
from axiom.vi.exponential import Linear as LinearLikelihood
from axiom.vi.transforms.base import Transform

from axiom.vi.exponential import MultivariateNormal, ExponentialFamily, MixtureMessage
from axiom.vi.utils import params_to_tx, ArrayDict, bdot, tree_at, inv_and_logdet

DEFAULT_EVENT_DIM = 2


@params_to_tx({"eta_1": "xx", "eta_2": "yx", "eta_3": "yy", "eta_4": "ones"})
class LinearMatrixNormalGamma(Transform):
    r"""
    This distribution models the linear transformation:

    .. math::
       y = Ax + \epsilon

    where:
        :math:`y` is the output vector with size p
        :math:`x` is the input vector with size d
        :math:`A` are the linear transformation parameters with size p x d
        :math:`\epsilon` is additive Gaussian noise

    The distribution models this transformation with the following likelihood:

    .. math::
        \log p(y|x,A,\Sigma) = - 0.5 * (y - Ax)^T \Sigma^{-1}(y - Ax)
                               + 0.5 \log |\Sigma^{-1}| -\frac{d}{2} \log (2 \pi)

    where:
        :math: `Sigma` is a positive definite covariance matrix with size p x p

    The conjugate prior to :math:`\A` and :math:`\Sigma` is the Matrix Normal Gamma distribution:

    .. math::
        A | \Sigma^{-1} ~ \mathcal{MN}(A | \mu_0, \Sigma^{-1}, V_0)
        \Sigma^{-1} := Diag(\gamma) ~ \mathcal{Gamma}(\gamma | a_0, b_0)

    where:
        :math:`\mu_0` is the prior mean of the linear transformation
        :math:`V_0` is the prior covariance of the linear transformation
        :math:`a_0` is the prior shape of the diagonal elements of covariance
        :math:`b_0` is the prior scale of the diagonal elements of covariance

    The full joint distribution is then given by:

    .. math::
        log p(y,x,A,\Sigma) = log p(y|x, A, \Sigma) + log p(A | \Sigma^{-1}) + log p(\Sigma^{-1})

    The posterior distribution is also a Matrix Normal Gamma distribution:

    .. math::
        A | \Sigma^{-1} ~ \mathcal{MN}(A | \mu, \Sigma^{-1}, V)
        \Sigma^{-1} := Diag(\gamma) ~ \mathcal{Gamma}(\gamma | a, b)

    where:
        :math:`\mu` is the posterior mean of the linear transformation
        :math:`V` is the posterior covariance of the linear transformation
        :math:`a` is the posterior shape of the diagonal elements of covariance
        :math:`b` is the posterior scale of the diagonal elements of covariance

    This distribution utilises the following (natural) parameterisation:

    .. math::
        \eta_1 = V^{-1}
        \eta_2 = \mu V^{-1}
        \eta_{3,k} = 2b + [U^{-1} + \mu V^{-1} \mu^T]_{kk}
        \eta_{4,k} = 2 (a - 1) + d

    which have the following sufficient statistics T(x):

    .. math::
        \eta_1 = \sum_{i=1}^n x_i x_i^T
        \eta_2 = \sum_{i=1}^n y_i x_i^T
        \eta_3 = \sum_{i=1}^n y_i y_i^T
        \eta_4 = N

    The named parameter fields of the arguments `params` and `prior_params` are:

    - `mu`: the mean of the linear transformation mapping from X to Y
    - `inv_v`: the inverse covariance matrix of the linear transformation mapping from X to Y
    - `a`: the shape parameters of each Gamma distribution over the inverse variance of output variate y_{i}
    - `b`: the scale parameters of each Gamma distribution over the inverse variance of output variate y_{i}

    """

    _v: Array
    _logdet_inv_v: Array
    _b: Array
    _prior_logdet_inv_v: Array
    _prior_v: Array

    x_dim: int
    y_dim: int

    pytree_data_fields = (
        "_v",
        "_logdet_inv_v",
        "_b",
        "_prior_logdet_inv_v",
        "_prior_v",
        "_prior_b",
    )
    pytree_aux_fields = (
        "x_dim",
        "y_dim",
        "use_bias",
        "fixed_precision",
        "trivial_batch_axes",
    )

    def __init__(
        self,
        params: ArrayDict = None,
        prior_params: ArrayDict = None,
        event_dim: int = DEFAULT_EVENT_DIM,
        use_bias: bool = True,
        fixed_precision: bool = False,
        scale: float = 1.0,
        dof_offset: float = 1.0,  # offset for the prior degrees of freedom (n = 1 + dim + dof_offset) If dof_offset = 0.0 then expected_sigma() is undefined, so one can use dof_offset=1.0 if expected_sigma() is needed
        inv_v_scale: float = 1.0,  # scale for the prior precision matrix V^{-1} of the (columns of the) linear transformation
        batch_shape: Optional[Tuple[int]] = None,
        event_shape: Optional[Tuple[int]] = None,
        init_key=None,
    ):

        if params is not None:
            self.x_dim, self.y_dim = (
                params.mu.shape[-(DEFAULT_EVENT_DIM - 1)],
                params.mu.shape[-DEFAULT_EVENT_DIM],
            )

        elif prior_params is not None:
            self.x_dim, self.y_dim = (
                prior_params.mu.shape[-(DEFAULT_EVENT_DIM - 1)],
                prior_params.mu.shape[-DEFAULT_EVENT_DIM],
            )

        elif event_shape is not None:
            self.x_dim, self.y_dim = (
                event_shape[-(DEFAULT_EVENT_DIM - 1)],
                event_shape[-DEFAULT_EVENT_DIM],
            )

        self.use_bias = use_bias
        self.fixed_precision = fixed_precision

        if prior_params is None:
            prior_params = self.init_default_params(
                batch_shape,
                event_shape,
                self.x_dim,
                self.y_dim,
                scale,
                dof_offset,
                inv_v_scale,
                DEFAULT_EVENT_DIM,
            )

        if fixed_precision:  # this is so that _update_cache() works when `super().__init__()` is called below
            self._prior_v = jnp.linalg.inv(prior_params.inv_v)

        self._prior_b = prior_params.b  # Need to initialise this before super()__init__ is called

        if params is None:
            key = init_key if init_key is not None else jr.PRNGKey(0)
            s = scale / jnp.sqrt(self.x_dim)
            mu = prior_params.mu + jr.uniform(
                key,
                batch_shape + event_shape[:-event_dim] + (self.y_dim, self.x_dim),
                minval=-3 * s,
                maxval=3 * s,
            )
            inv_v = jnp.where(prior_params.inv_v > 0, 1.0, 0.0)
            a = jnp.ones_like(prior_params.a) * 2.0
            b = jnp.ones_like(prior_params.b)
            params = tree_at(lambda x: (x.mu, x.inv_v, x.a, x.b), prior_params, (mu, inv_v, a, b))

        inferred_batch_shape, inferred_event_shape = self.infer_shapes(params.mu, event_dim)
        if batch_shape is None:
            batch_shape = inferred_batch_shape
        if event_shape is None:
            event_shape = inferred_event_shape

        self._prior_b = prior_params.b  # Need to initialise this before super()__init__ is called

        super().__init__(DEFAULT_EVENT_DIM, LinearLikelihood, params, prior_params, batch_shape, event_shape)

        self._prior_v, self._prior_logdet_inv_v = inv_and_logdet(self.prior_inv_v)
        self.trivial_batch_axes = tuple([i for i, d in enumerate(batch_shape) if d == 1])

    @staticmethod
    def init_default_params(
        batch_shape,
        event_shape,
        x_dim,
        y_dim,
        scale=1.0,
        dof_offset=1.0,
        inv_v_scale=1.0,
        default_event_dim=2,
    ) -> ArrayDict:
        """
        Initialize the default parameters of the distribution.
        """

        prior_mu = jnp.full(
            batch_shape + event_shape[:-default_event_dim] + (y_dim, x_dim), 0.0
        )  # shape should be (batch_shape,) + (y_dim, x_dim)

        prior_inv_v = inv_v_scale * jnp.broadcast_to(
            jnp.eye(x_dim),
            batch_shape + event_shape[:-default_event_dim] + (x_dim, x_dim),
        )  # shape should be (batch_shape,) + (x_dim, x_dim).

        prior_a = jnp.full(
            batch_shape + event_shape[:-default_event_dim] + (y_dim, 1),
            1.0 + dof_offset,
        )  # shape should be (batch_shape,) + (1, 1)

        prior_b = scale**2 * jnp.broadcast_to(
            jnp.ones((y_dim, 1)),
            batch_shape + event_shape[:-default_event_dim] + (y_dim, 1),
        )  # shape should be (batch_shape,) + (y_dim, y_dim).

        return ArrayDict(mu=prior_mu, inv_v=prior_inv_v, a=prior_a, b=prior_b)

    @property
    def norm(self):
        return jnp.expand_dims(self.inv_v.reshape(self.batch_shape + (-1,)).max(-1), (-1, -2))

    @property
    def mu(self):
        return bdot(self.posterior_params.eta.eta_2, self.v)

    @property
    def inv_v(self):
        return self.posterior_params.eta.eta_1

    @property
    def b(self):
        if self.fixed_precision:
            return self.prior_b

        if self._b is None:
            norm = self.norm
            rho = jnp.diagonal(bdot(self.mu, self.posterior_params.eta.eta_2.mT / norm), axis1=-1, axis2=-2)
            self._b = (self.posterior_params.eta.eta_3 / norm - expand(rho, -1)) * norm / 2

        return jnp.clip(self._b, min=1e-6)

    @property
    def a(self):
        if self.fixed_precision:
            return self.prior_a
        else:
            return (self.posterior_params.eta.eta_4 - self.x_dim) / 2 + 1

    @property
    def prior_mu(self):
        return bdot(self.prior_params.eta.eta_2, self.prior_v)

    @property
    def prior_inv_v(self):
        return self.prior_params.eta.eta_1

    @property
    def prior_b(self):
        return self._prior_b

    @property
    def prior_a(self):
        return (self.prior_params.eta.eta_4 - self.x_dim) / 2 + 1

    @property
    def v(self):
        if self._v is None:
            norm = self.norm
            scaled_v, scaled_logdet_inv_v = inv_and_logdet(self.inv_v / norm)
            self._v = scaled_v / norm
            self._logdet_inv_v = scaled_logdet_inv_v + self.x_dim * jnp.log(norm)
        return self._v

    @property
    def prior_v(self):
        if self._prior_v is None:
            self._prior_v, self._prior_logdet_inv_v = inv_and_logdet(self.prior_inv_v)
        return self._prior_v

    @property
    def logdet_inv_v(self):
        if self._logdet_inv_v is None:
            norm = self.norm
            self._logdet_inv_v = inv_and_logdet(self.inv_v / norm, return_inverse=False) + self.x_dim * jnp.log(norm)
        return self._logdet_inv_v

    @property
    def prior_logdet_inv_v(self):
        if self._prior_logdet_inv_v is None:
            self._prior_logdet_inv_v = inv_and_logdet(self.prior_inv_v, return_inverse=False)
        return self._prior_logdet_inv_v

    @property
    def weights(self):
        return self.mu[..., :-1] if self.use_bias else self.mu

    @property
    def bias(self):
        return self.mu[..., -1:] if self.use_bias else jnp.broadcast_to(jnp.zeros(1), self.mu.shape[:-1] + (1,))

    def update_from_probabilities(
        self,
        pXY: Tuple[Distribution],
        weights: Optional[Array] = None,
        lr: float = 1.0,
        beta: float = 0.0,
        apply_updates: bool = True,
    ):
        """
        Custom version of `update_from_probs` that accelerates computation in case of self.use_bias == True
        """
        pX, pY = pXY

        if isinstance(pX, MixtureMessage):
            pX = pX.marginalize(keepdims=False)

        sample_shape = self.get_sample_shape(pX.mean)
        sample_dims = self.get_sample_dims(pX.mean)

        px_exp_xx = pX.expected_xx()
        px_exp_x = pX.expected_x()
        if self.use_bias:
            sigma = px_exp_xx - px_exp_x * px_exp_x.mT

            pad_width = [(0, 0)] * sigma.ndim
            pad_width[-1] = (0, 1)
            pad_width[-2] = (0, 1)
            sigma = jnp.pad(sigma, pad_width=pad_width)

            pad_width = [(0, 0)] * px_exp_x.ndim
            pad_width[-2] = (0, 1)
            px_exp_x = jnp.pad(px_exp_x, pad_width, constant_values=1.0)

            px_exp_xx = sigma + px_exp_x * px_exp_x.mT

        py_exp_x = pY.expected_x()
        py_exp_xx = pY.expected_xx()

        # Determine the common batch_shape to broadcast SEyx to -- this matters in case
        # pX and pY have different batch_shapes, which occurs in the context of certain
        # latent variable models
        # (sample_dim is treated here as a batch_dim, so we need to exclude it first)
        pX_batch_shape = self.get_batch_shape(pX.mean)
        pY_batch_shape = self.get_batch_shape(pY.mean)
        common_batch_shape = jnp.broadcast_shapes(pX_batch_shape, pY_batch_shape)

        weights = jnp.ones((1,) * len(sample_dims) + common_batch_shape) if weights is None else weights
        weights = self.expand_event_dims(weights)
        weights_batch_shape = self.get_batch_shape(weights)
        common_batch_shape = jnp.broadcast_shapes(common_batch_shape, weights_batch_shape)
        py_exp_x = py_exp_x * weights
        ones = jnp.broadcast_to(weights, sample_shape + weights.shape[len(sample_shape) :])
        summed_stats = ArrayDict(
            xx=(px_exp_xx * weights).sum(sample_dims),
            yx=(py_exp_x * px_exp_x.mT).sum(sample_dims),
            yy=expand(jnp.diagonal(py_exp_xx * weights, axis1=-1, axis2=-2), -1).sum(sample_dims),
            ones=ones.sum(sample_dims),
        )

        summed_stats = tree_util.tree_map(lambda se: se.sum(self.trivial_batch_axes, keepdims=True), summed_stats)
        if apply_updates:
            self.update_from_statistics(self.map_stats_to_params(summed_stats, None), lr, beta)
        else:
            return self.map_stats_to_params(summed_stats, None)

    def update_from_data(
        self,
        data: Tuple[Array],
        weights: Optional[Array] = None,
        lr: float = 1.0,
        beta: float = 0.0,
    ):
        """Custom version of `update_from_data` that accelerates computation in case of self.use_bias == True"""

        likelihood_stats = self.likelihood.statistics(
            data, has_bias=self.use_bias
        )  # evaluate the sufficient statistics T(x)
        # changed sufficient statistics from MatrixNormalWishart to MatrixNormalGamma
        yy = expand(jnp.diagonal(likelihood_stats.yy, axis1=-1, axis2=-2), -1)
        likelihood_stats = tree_at(lambda lls: lls.yy, likelihood_stats, yy)

        X, Y = data
        sample_shape = self.get_sample_shape(X)
        sample_dims = self.get_sample_dims(X)

        if weights is None:
            summed_stats = tree_util.tree_map(
                lambda x: x.sum(sample_dims),
                likelihood_stats,
                is_leaf=lambda x: isinstance(x, Array),
            )
        else:
            weights = weights.reshape(weights.shape + self.event_dim * (1,))
            summed_stats = tree_util.tree_map(
                lambda x: (x * weights).sum(sample_dims),
                likelihood_stats,
                is_leaf=lambda x: isinstance(x, Array),
            )

        self.update_from_statistics(self.map_stats_to_params(summed_stats, None), lr, beta)

    def to_natural_params(self, params: ArrayDict) -> ArrayDict:
        r"""
        Convert the given parameters to natural parameters.

        .. math::
            \eta_1 = V^{-1}
            \eta_2 = \mu V^{-1}
            \eta_{3,k} = 2b + [U^{-1} + \mu V^{-1} \mu^T]_{kk}
            \eta_{4,k} = 2 (a - 1) + d
        """
        norm = jnp.expand_dims(params.inv_v.reshape(self.batch_shape + (-1,)).max(-1), (-1, -2))
        eta_1 = params.inv_v
        _eta_2 = bdot(params.mu, params.inv_v / norm)
        rho = jnp.diagonal(bdot(params.mu, _eta_2.mT), axis1=-1, axis2=-2)
        eta_3 = (2 * params.b / norm + expand(rho, -1)) * norm
        eta_4 = 2 * (params.a - 1) + self.x_dim
        return ArrayDict(eta=ArrayDict(eta_1=eta_1, eta_2=_eta_2 * norm, eta_3=eta_3, eta_4=eta_4), nu=None)

    def expected_posterior_statistics(self) -> ArrayDict:
        """
        Computes the expected sufficient statistics of the posterior distribution's parameters η and v.

        .. math::
            (<S(θ)>_{q(θ|η, v)}, -<A(θ)>_{q(θ|η, v)})

        N.B. the negation before the return <A(θ)>_{q(θ|η, v)} is done in order
        to make it ready for computing the dot products that help with things like `expected_log_likelihood`
        """
        eta_stats = self.expected_likelihood_params()
        return ArrayDict(eta=eta_stats, nu=None)
        # nu_stats = self.expected_log_partition()
        # return ArrayDict(eta=eta_stats, nu=nu_stats)

    def expected_likelihood_params(self) -> ArrayDict:
        """
        Compute the expected likelihood parameters.
        """
        inv_sigma = self.expected_inv_sigma()
        inv_sigma_x = self.expected_inv_sigma_x(inv_sigma)
        x_inv_sigma_x = self.expected_x_inv_sigma_x(inv_sigma_x)
        return ArrayDict(
            eta_1=-0.5 * x_inv_sigma_x,
            eta_2=inv_sigma_x,
            eta_3=-0.5 * jnp.diagonal(inv_sigma, axis1=-1, axis2=-2)[..., None],
            eta_4=0.5 * self.expected_logdet_inv_sigma(),
        )

    def expected_log_partition(self):
        """
        Computes the log partition A(θ) of the likelihood expected under the variational distribution,
        i.e., <A(θ)>_{q(θ|η, v)}
        """
        raise NotImplementedError

    def log_prior_partition(self) -> Array:
        """
        Computes the log partition function of the prior distribution, log Z(η₀, ν₀).
        """
        return self._log_partition(self.prior_mu, self.prior_logdet_inv_v, self.prior_a, self.prior_b)

    def log_posterior_partition(self) -> Array:
        """
        Computes the log partition function of the posterior distribution, log Z(η, v).
        """
        return self._log_partition(self.mu, self.logdet_inv_v, self.a, self.b)

    def _log_partition(self, mean: Array, logdet_inv_v: Array, a: Array, b: Array) -> Array:
        d = self.y_dim
        p = self.x_dim

        term_1 = ((d * p) / 2) * jnp.log(2 * jnp.pi)
        term_2 = -(d / 2) * logdet_inv_v
        term_3 = (lgamma(a) - a * jnp.log(b)).sum((-2, -1), keepdims=True)
        return term_1 + term_2 + term_3

    def predict(self, x: Array) -> ExponentialFamily:
        """
        Computes the variational prediction distribution.
        @TODO: Adapt to two settings of self.use_bias
        """

        if self.use_bias:
            inv_sigma_mu = bdot(self.expected_inv_sigma_x()[..., :-1], x) + self.expected_inv_sigma_x()[..., -1:]
            nat_params = ArrayDict(inv_sigma_mu=inv_sigma_mu, inv_sigma=self.expected_inv_sigma())
            res = -0.5 * bdot(x.mT, bdot(self.expected_x_inv_sigma_x()[..., :-1, :-1], x))
            res = (
                res
                - bdot(self.expected_x_inv_sigma_x()[..., -1:, :-1], x)
                - 0.5 * self.expected_x_inv_sigma_x()[..., -1:, -1:]
            )
        else:
            nat_params = ArrayDict(
                inv_sigma_mu=bdot(self.expected_inv_sigma_x(), x),
                inv_sigma=self.expected_inv_sigma(),
            )
            res = -0.5 * bdot(x.mT, bdot(self.expected_x_inv_sigma_x(), x))

        res += 0.5 * self.expected_logdet_inv_sigma().sum((-2, -1), keepdims=True)
        res = res.squeeze((-2, -1)) + self._likelihood.log_measure(x)
        return MultivariateNormal(nat_params, residual=res)

    def expected_inv_sigma_x(self, inv_sigma=None) -> Array:
        r"""
        Compute the expected value of the inverse covariance matrix times the input.

        .. math::
            E[\Sigma^{-1} A]
        """
        if inv_sigma is None:
            return bdot(self.expected_inv_sigma(), self.mu)
        else:
            return bdot(inv_sigma, self.mu)

    def expected_inv_sigma(self) -> Array:
        r"""
        Compute the expected value of the inverse covariance matrix.

        .. math::
            E[\Sigma^{-1}]
        """
        return (self.a / self.b) * jnp.eye(self.y_dim)

    def expected_sigma(self) -> Array:
        r"""
        Compute the expected value of the inverse covariance matrix.

        .. math::
            E[\Sigma]
        """
        return (self.b / (self.a - 1)) * jnp.eye(self.y_dim)

    def expected_inv_sigma_diag(self) -> Array:
        # return a vector (dim,1) that contains the diagonal
        return self.a / self.b

    def inv_expected_inv_sigma(self) -> Array:
        r"""
        Compute the inverse of the expected value of the inverse covariance matrix.

        .. math::
            E[\Sigma^{-1}]^{-1}
        """
        return (self.b / self.a) * jnp.eye(self.y_dim)

    def expected_logdet_inv_sigma(self) -> Array:
        r"""
        Compute the expectation of the log determinant of the inverse covariance matrix.

        .. math::
            E[ log | \Sigma^{-1} |] = \sum_{k=1}^K E[\log \gamma_k] = \sum(digamma(a) - log(b))
        Note that we keep the vectorformat of E[log \gamma_k] with K elements,
        because the expected_likelihood_params() will need to align this with the vectorformat
        \eta_4

        """
        return jnp.sum(jsp.digamma(self.a) - jnp.log(self.b), -2, keepdims=True)

    def logdet_expected_inv_sigma(self) -> Array:
        r"""
        Compute the log determinant of the expected value of the inverse covariance matrix.

        .. math::
            log | E[\Sigma^{-1}] | = \sum_{k=1}^K log(a / b)
        """
        return jnp.sum(jnp.log(self.a) - jnp.log(self.b), -2, keepdims=True)

    def expected_log_det_inv_sigma_minus_log_det_expected_inv_sigma(self) -> Array:
        return jnp.sum(jsp.digamma(self.a) - jnp.log(self.a), -2, keepdims=True)

    def expected_x_inv_sigma_x(self, inv_sigma_mu=None) -> Array:
        r"""
        Compute the expected value of the inverse covariance matrix times the input.

        .. math::
            E[A^T\Sigma^{-1} A]
        """
        if inv_sigma_mu is None:
            return self.y_dim * self.v + bdot(self.mu.mT, bdot(self.expected_inv_sigma(), self.mu))
        else:
            return self.y_dim * self.v + bdot(self.mu.mT, inv_sigma_mu)

    def _update_cache(self):
        """Update parameters required for computing expectations"""
        norm = self.norm
        _v, _logdet_inv_v = inv_and_logdet(self.inv_v / norm)
        self._v, self._logdet_inv_v = _v / norm, _logdet_inv_v + self.x_dim * jnp.log(norm)
        rho = jnp.diagonal(
            bdot(self.posterior_params.eta.eta_2 / norm, self.mu.mT),
            axis1=-1,
            axis2=-2,
        )
        self._b = (self.posterior_params.eta.eta_3 / norm - expand(rho, -1)) * norm / 2

    def expected_log_likelihood(self, data: Tuple[Array]) -> Array:
        r"""
        Compute the expected log likelihood of the given data.

        .. math::
            E_{p(A,\Sigma)}[log p(y|x,A,\Sigma)]
        """

        x, y = data
        ex_inv_sigma_x = self.expected_x_inv_sigma_x()
        e_inv_sigma_x = self.expected_inv_sigma_x()
        probs = -0.5 * bdot(y.mT, bdot(self.expected_inv_sigma(), y)).squeeze((-1, -2))
        if self.use_bias:
            probs = probs + (
                bdot(
                    y.mT,
                    bdot(e_inv_sigma_x[..., :, :-1], x) + e_inv_sigma_x[..., :, -1:],
                )
            ).squeeze((-1, -2))
            probs -= 0.5 * bdot(x.mT, bdot(ex_inv_sigma_x[..., :-1, :-1], x)).squeeze((-1, -2))
            probs -= bdot(ex_inv_sigma_x[..., -1:, :-1], x).squeeze((-1, -2))
            probs -= 0.5 * ex_inv_sigma_x[..., -1, -1]
        else:
            probs += bdot(y.mT, bdot(e_inv_sigma_x, x)).squeeze((-1, -2))
            probs -= 0.5 * bdot(x.mT, bdot(ex_inv_sigma_x, x)).squeeze((-1, -2))
        probs += 0.5 * self.expected_logdet_inv_sigma().sum((-1, -2))
        probs -= 0.5 * self.y_dim * jnp.log(2 * jnp.pi)
        return probs

    def average_energy(self, inputs: Tuple[Distribution]) -> Array:
        r"""
        Compute the average energy term of the likelihood factor, aka
        .. math::
            E_{P(x)P(y)q(A,\Sigma)}[log p(y|x,A,\Sigma)]

        Plays the role of the expected_log_likelihood term when the inputs are distributions over the inputs and outputs, aka
        .. math::
            P(X), P(Y)

        Note that the only requires of the input messages is that they have the following expectations defined: pX.mean and pX.expected_xx()
        """

        pX, pY = inputs
        py_exp_xx = pY.expected_xx()
        px_exp_xx = pX.expected_xx()

        U = -0.5 * (py_exp_xx * self.expected_inv_sigma()).sum((-2, -1))

        EXTinvUX = self.expected_x_inv_sigma_x()
        EinvUX = self.expected_inv_sigma_x()

        if self.use_bias:
            U += bdot(pY.mean.mT, bdot(EinvUX[..., :, :-1], pX.mean) + EinvUX[..., :, -1:]).squeeze((-1, -2))
            U -= 0.5 * (px_exp_xx * EXTinvUX[..., :-1, :-1]).sum((-2, -1))
            U -= bdot(EXTinvUX[..., -1:, :-1], pX.mean).squeeze((-1, -2))
            U -= 0.5 * EXTinvUX[..., -1, -1]
        else:
            U += bdot(pY.mean.mT, bdot(EinvUX, pX.mean)).squeeze((-1, -2))
            U -= 0.5 * (px_exp_xx * EXTinvUX).sum((-1, -2))

        U += 0.5 * self.expected_logdet_inv_sigma().sum((-1, -2)) - 0.5 * self.y_dim * jnp.log(2 * jnp.pi)
        return U

    def kl_divergence(self) -> Array:
        r"""
        Compute the KL divergence between the posterior and prior distributions.

        .. math::
            KL(p(A,\Sigma) || p(A,\Sigma)) = KL(p(A|\Sigma) || p(A|\Sigma)) + KL(p(\Sigma) || p(\Sigma))
        """
        kl = (
            self.y_dim / 2.0 * self.logdet_inv_v.squeeze((-2, -1))
            - self.y_dim / 2.0 * self.prior_logdet_inv_v.squeeze((-2, -1))
            - self.y_dim * self.x_dim / 2.0
        )

        traceV = (self.prior_inv_v * self.v).sum((-2, -1))
        kl = kl + 0.5 * self.y_dim * traceV

        traceMuUMu = (
            self.prior_inv_v
            * bdot(
                (self.mu - self.prior_mu).mT,
                bdot(self.expected_inv_sigma(), self.mu - self.prior_mu),
            )
        ).sum((-2, -1))
        kl = kl + 0.5 * traceMuUMu

        kl_gamma = self._kl_divergence_gamma()
        return kl + kl_gamma

    def _kl_divergence_gamma(self) -> Array:
        r"""
        Compute the KL divergence between the posterior and prior gamma distributions.

        .. math::
            sum_k KL(p(\gamma_k) || p(\gamma_k))
        """
        kl = self.prior_a * (jnp.log(self.b) - jnp.log(self.prior_b))
        kl = kl + lgamma(self.prior_a) - lgamma(self.a)
        kl = kl + (self.a - self.prior_a) * jsp.digamma(self.a)
        kl = kl - (self.b - self.prior_b) * (self.a / self.b)
        return kl.sum((-2, -1))

    def forward_from_normal(self, pX: MultivariateNormal, pass_residual=False) -> MultivariateNormal:
        r"""
        When the input distribution is a multivariate normal (or mixture of multivariate normals), then
        we can analytically compute the the forward message by marginalising over the input distribution.
        This is computationally more expensive, but gives better results.  The logic of this function is to
        take advantage of the fact that:
            exp\left( \left< \log p(y|x,A,\Sigma) \right>_{q(A,\Sigma)} \right)
            =
            p(y|x,<A>,<\Sigma^{-1}>^{-1}) \exp( \frac{1}{2} (< \log | \Sigma^{-1} |>  - \log | <\Sigma^{-1}> | - dim_y X^T V X ) )

        As a result the forward message is obtained by just adding some evidence for X with precision dim_y*V
        and then linearlly transforming it by <A> and then adding noise with covariance <\Sigma^{-1}>^{-1}
        """
        # The definition of residual used here is log q(y) + residual = int exp <log p(y|x,A,\Sigma)>_{q(A,\Sigma)} * q(x)dx

        # Stuff that doesnt go into joint goes into the residual (drop the measure function???)
        res = -pX.log_partition().squeeze((-1, -2))
        res += 0.5 * self.expected_logdet_inv_sigma().squeeze((-2, -1))  # - 0.5 * self.y_dim * jnp.log(2 * jnp.pi)
        if self.use_bias is False:
            # Joint Precision = [A, -B;-B^T, D]
            # Joint Precision@mu = [C_y; C_x]
            A = self.expected_inv_sigma()
            B = self.expected_inv_sigma_x()
            D = self.expected_x_inv_sigma_x() + pX.inv_sigma
            C_y = 0.0
            C_x = pX.inv_sigma_mu

            invD, logdetD = inv_and_logdet(D)
            # stuff that completes the square goes into the residual
            res += 0.5 * bdot(C_x.mT, bdot(invD, C_x)).squeeze((-2, -1)) - 0.5 * logdetD.squeeze((-2, -1))
        else:
            A = self.expected_inv_sigma()
            B = self.expected_inv_sigma_x()
            D = self.expected_x_inv_sigma_x()
            C_y = B[..., :, -1:]
            B = B[..., :, :-1]
            C_x = -D[..., :-1, -1:] + pX.inv_sigma_mu
            res += -0.5 * D[..., -1, -1]  # from bbT (doesnt go into the joint)
            D = D[..., :-1, :-1] + pX.inv_sigma

            invD, logdetD = inv_and_logdet(D)
            # stuff that completes the square goes into the residual
            res += 0.5 * bdot(C_x.mT, bdot(invD, C_x)).squeeze((-2, -1)) - 0.5 * logdetD.squeeze((-2, -1))

        inv_sigma_yy = A - bdot(B, bdot(invD, B.mT))
        inv_sigma_mu_y = C_y + bdot(B, bdot(invD, C_x))

        ndim_diff = inv_sigma_mu_y.ndim - inv_sigma_yy.ndim
        if ndim_diff > 0:
            inv_sigma_yy = jnp.expand_dims(inv_sigma_yy, tuple(range(ndim_diff)))

        if pass_residual:
            res += pX.residual
        pY = MultivariateNormal(ArrayDict(inv_sigma_mu=inv_sigma_mu_y, inv_sigma=inv_sigma_yy), residual=res)
        pY.residual += pY.log_partition().squeeze((-2, -1))
        return pY

    def backward_from_normal(self, pY: MultivariateNormal, pass_residual=False) -> MultivariateNormal:
        """
        When the output distribution is a multivariate normal (or mixture of multivariate normals), then
        we can analytically compute the backward message by marginalising over the output distribution.
        This is computationally more expensive, but gives better results.
        """

        # stuff that doesnt go into joint goes into the residual (drop measure???)
        res = -pY.log_partition().squeeze((-1, -2))
        res += 0.5 * self.expected_logdet_inv_sigma().squeeze((-1, -2))  # - 0.5 * self.y_dim * jnp.log(2 * jnp.pi)
        A = self.expected_inv_sigma() + pY.inv_sigma
        invA, logdetA = inv_and_logdet(A)

        if self.use_bias is False:
            # Joint Precision = [A, -B;-B^T, D]
            # Joint Precision@mu = [C_y; C_x]
            B = self.expected_inv_sigma_x()
            D = self.expected_x_inv_sigma_x()
            C_y = pY.inv_sigma_mu
            C_x = 0.0

            invA, logdetA = inv_and_logdet(A)
            res += 0.5 * bdot(C_y.mT, bdot(invA, C_y)).squeeze((-2, -1)) - 0.5 * logdetA.squeeze((-2, -1))
        else:
            B = self.expected_inv_sigma_x()
            D = self.expected_x_inv_sigma_x()
            C_y = pY.inv_sigma_mu + B[..., :, -1:]
            B = B[..., :, :-1]
            C_x = -D[..., :-1, -1:]
            res += -0.5 * D[..., -1, -1]  # from bbT (doesnt go into the joint)
            D = D[..., :-1, :-1]
            # stuff that completes the square goes into the residual
            res += 0.5 * bdot(C_y.mT, bdot(invA, C_y)).squeeze((-2, -1)) - 0.5 * logdetA.squeeze((-2, -1))

        inv_sigma_xx = D - bdot(B.mT, bdot(invA, B))
        inv_sigma_mu_x = C_x + bdot(B.mT, bdot(invA, C_y))

        ndim_diff = inv_sigma_mu_x.ndim - inv_sigma_xx.ndim
        if ndim_diff > 0:
            inv_sigma_xx = jnp.expand_dims(inv_sigma_xx, tuple(range(ndim_diff)))

        if pass_residual:
            res += pY.residual
        pX = MultivariateNormal(ArrayDict(inv_sigma_mu=inv_sigma_mu_x, inv_sigma=inv_sigma_xx), residual=res)
        return pX

    def variational_forward(self, pX: Distribution, pass_residual=False) -> MultivariateNormal:
        """
        This is distinct from forward in that it doesnt marginalize over the joint distribution of X and Y
        rather it computes the expected value of the log probability of Y given X.  This approximation
        fails to propagate uncertainty about X to Y, but is fast.  It also makes it possible to simply
        update means and variances coincidentally with the natural parameters.  (not implemented yet)
        """

        inv_sigma_y = self.expected_inv_sigma()  # yy term
        res = 0.5 * self.expected_logdet_inv_sigma().squeeze((-1, -2))  #  new addition (taken from Linear)
        expected_inv_sigma_x = self.expected_inv_sigma_x(inv_sigma=inv_sigma_y)
        inv_sigma_xx = self.expected_x_inv_sigma_x(inv_sigma_mu=expected_inv_sigma_x)

        if self.use_bias:
            inv_sigma_mu_y = (
                bdot(expected_inv_sigma_x[..., :, :-1], pX.expected_x()) + expected_inv_sigma_x[..., :, -1:]
            )  # xy, yb terms
            res -= 0.5 * jnp.sum(inv_sigma_xx[..., :-1, :-1] * pX.expected_xx(), (-2, -1))  # xx term
            res -= jnp.sum(inv_sigma_xx[..., :-1, -1:] * pX.expected_x(), (-2, -1))  # xb term
            res -= 0.5 * inv_sigma_xx[..., -1, -1]  # bb term
        else:
            inv_sigma_mu_y = bdot(expected_inv_sigma_x, pX.expected_x())  # xy term
            res -= 0.5 * jnp.sum(inv_sigma_xx * pX.expected_xx(), (-2, -1))  # xx term

        shape = jnp.broadcast_shapes(inv_sigma_y.shape[:-2], inv_sigma_mu_y.shape[:-2])
        inv_sigma_y = jnp.broadcast_to(inv_sigma_y, shape + inv_sigma_y.shape[-2:])
        inv_sigma_mu_y = jnp.broadcast_to(inv_sigma_mu_y, shape + inv_sigma_mu_y.shape[-2:])

        if pass_residual:  #  new addition (taken from Linear)
            res += pX.residual
        pY = MultivariateNormal(ArrayDict(inv_sigma_mu=inv_sigma_mu_y, inv_sigma=inv_sigma_y), residual=res)
        pY.residual += pY.log_partition().squeeze((-1, -2))  #  new addition (taken from Linear)
        return pY

    def variational_backward(self, pY: Distribution, pass_residual=False) -> Distribution:
        """
        More generally, i.e. when the output distribution is not multivariate normal (or mixture of multivariate
        normals), then an analytic computation of the backward message is not possible. Instead, we use a variational
        approximation for the backward message.  This approximation, leads to an larger residual term, but is
        computationally cheaper, even when the input distribution is normally distributed.
        """

        inv_sigma = self.expected_inv_sigma()
        res = 0.5 * self.expected_logdet_inv_sigma().sum((-2, -1))
        res -= 0.5 * jnp.sum(inv_sigma * pY.expected_xx(), (-2, -1))  # yy term

        if self.use_bias is False:
            inv_sigma_mu = self.expected_inv_sigma_x(inv_sigma=inv_sigma)
            inv_sigma_mu_x = bdot(inv_sigma_mu.mT, pY.expected_x())  # xy term
            inv_sigma_xx = self.expected_x_inv_sigma_x(inv_sigma_mu=inv_sigma_mu)  # xx term
        else:
            Einv_sigma_x = self.expected_inv_sigma_x(inv_sigma=inv_sigma)
            Ex_inv_sigma_x = self.expected_x_inv_sigma_x(inv_sigma_mu=Einv_sigma_x)

            inv_sigma_mu_x = (
                bdot(Einv_sigma_x[..., :, :-1].mT, pY.expected_x()) - Ex_inv_sigma_x[..., :-1, -1:]
            )  # yx and bx term
            res += (
                bdot(Einv_sigma_x[..., :, -1:].mT, pY.expected_x()).squeeze((-2, -1))
                - 0.5 * Ex_inv_sigma_x[..., -1, -1]
            )  # yb and bb term
            inv_sigma_xx = Ex_inv_sigma_x[..., :-1, :-1]  # xx term

        shape = jnp.broadcast_shapes(inv_sigma_xx.shape[:-2], pY.expected_x().shape[:-2])
        inv_sigma_xx = jnp.broadcast_to(inv_sigma_xx, shape + inv_sigma_xx.shape[-2:])

        pX = MultivariateNormal(ArrayDict(inv_sigma_mu=inv_sigma_mu_x, inv_sigma=inv_sigma_xx), residual=res)
        if pass_residual:
            pX.residual += pY.residual
        return pX

    def joint(self, pX: Distribution, pY: Distribution) -> Distribution:
        """
        computes the joint distribution of the input and output given the model parameters
        This is used to compute the equivalent of xi in the HMM case and is needed for exact
        inference in the case of linear dynamical systems.
        """
        raise NotImplementedError

    def elbo(self, data: Tuple[Array], weights: Optional[Array] = None) -> Array:
        """
        Compute the evidence lower bound of the model given the data.
        """
        X, Y = data
        sample_dims = self.get_sample_dims(X)
        if weights is None:
            ELL = self.expected_log_likelihood((X, Y)).sum(sample_dims)
        else:
            ELL = (self.expected_log_likelihood(data) * weights).sum(sample_dims)
        return ELL - self.kl_divergence()

    def elbo_contrib(self, pXY: Tuple[Distribution], weights: Optional[Array] = None) -> Array:
        """
        Compute the evidence lower bound of the model given the data.
        """
        pX, pY = pXY
        sample_dims = self.get_sample_dims(pX.mean)
        if weights is None:
            ELL = self.average_energy(pXY).sum(sample_dims)
        else:
            ELL = (self.average_energy(pXY) * weights).sum(sample_dims)
        return ELL - self.kl_divergence()


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    key = jr.PRNGKey(11)
    num_samples = 1000
    batch_shape = (5,)
    y_dim = 2
    x_dim = 2
    scale = 0.1
    use_bias = True
    fixed_precision = False

    key, a_key, x_key, eps_key, b_key = jr.split(key, 5)

    A = jr.normal(a_key, batch_shape + (y_dim, x_dim))
    X = jr.normal(x_key, (num_samples,) + batch_shape + (x_dim, 1))  # shape is (num_samples,) + batch_dim + (x_dim,)
    B = jr.normal(b_key, batch_shape + (y_dim, 1))
    epsilon = jr.normal(eps_key, (num_samples,) + batch_shape + (y_dim, 1)) * 0.1

    Y = A @ X + B * use_bias + epsilon  # shape is (num_samples,) + batch_dim + (y_dim,)

    key, a_key, v_key = jr.split(key, 3)

    x_dim = x_dim + use_bias
    event_shape = (y_dim, x_dim)

    prior_mu = jnp.full(batch_shape + event_shape, 0.0)  # shape should be (batch_shape,) + (y_dim, x_dim)
    prior_inv_v = jnp.broadcast_to(
        jnp.eye(x_dim), batch_shape + (x_dim, x_dim)
    )  # shape should be (batch_shape,) + (x_dim, x_dim). Same goes for the variable `v`
    prior_a = jnp.full(
        batch_shape + (y_dim, 1), 10.0
    )  # shape should be (batch_shape,) + (y_dim, y_dim). Same goes for the variable `inv_sigma` and `inv_expected_inv_sigma()`
    prior_b = jnp.full(
        batch_shape + (y_dim, 1), 10.0
    )  # shape should be (batch_shape,) + (y_dim, y_dim). Same goes for the variable `inv_sigma` and `inv_expected_inv_sigma()`
    mu = prior_mu + jr.normal(a_key, batch_shape + event_shape) / jnp.sqrt(x_dim)
    inv_v = prior_inv_v
    params = ArrayDict(mu=mu, inv_v=inv_v, a=prior_a, b=prior_b)
    prior_params = ArrayDict(mu=prior_mu, inv_v=prior_inv_v, a=prior_a, b=prior_b)

    model = LinearMatrixNormalGamma(params, prior_params, use_bias=use_bias, fixed_precision=fixed_precision)

    weights = jr.uniform(key, (num_samples,) + batch_shape)
    model.update_from_data((X, Y), weights=weights)

    pYhat = model.predict(X)
    plt.scatter(Y.squeeze(), pYhat.mean.squeeze())
    plt.title("predict")
    plt.show()

    pXhat = model.variational_backward(pYhat)
    plt.scatter(X.squeeze(), pXhat.mean.squeeze())
    plt.title("variational backward")
    plt.show()

    pXhat = model.backward_from_normal(pYhat)
    plt.scatter(X.squeeze(), pXhat.mean.squeeze())
    plt.title("backward_from_normal")
    plt.show()

    pYhat = model.forward_from_normal(pXhat)
    plt.scatter(Y.squeeze(), pYhat.mean.squeeze())
    plt.title("forward_from_normal")
    plt.show()

    pYhat = model.variational_forward(pXhat)
    plt.scatter(Y.squeeze(), pYhat.mean.squeeze())
    plt.title("variational_forward")
    plt.show()

    model.update_from_probabilities((pXhat, pYhat), weights=weights)
    pYhat = model.forward_from_normal(pXhat)
    plt.scatter(Y.squeeze(), pYhat.mean.squeeze())
    plt.title("forward_from_normal prupdate")
    plt.show()

    pXhat = model.backward_from_normal(pYhat)
    plt.scatter(X.squeeze(), pXhat.mean.squeeze())
    plt.title("backward_from_normal prupdate")
    plt.show()

    ellpxpy = model.elbo_contrib((pXhat, pYhat))
    ell = model.elbo((X, Y))
