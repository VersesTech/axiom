import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn import softmax
import equinox as eqx

from axiom.vi import ArrayDict
from axiom.vi.conjugate import Multinomial, MultivariateNormal
from axiom.vi.models.hybrid_mixture_model import HybridMixture


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
