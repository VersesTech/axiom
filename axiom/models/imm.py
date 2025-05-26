from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn import softmax
from jaxtyping import Array

import equinox as eqx

from axiom.vi.utils import bdot
from axiom.vi.models.hybrid_mixture_model import HybridMixture
from axiom.models.utils_hybrid import create_mm, train_step_fn


@dataclass(frozen=True)
class IMMConfig:
    """
    Configuration for the IMM
    """

    num_object_types: int = 32
    num_features: int = 5

    i_ell_threshold: float = -500

    cont_scale_identity: float = (0.5,)
    color_precision_scale: float = 1.0
    color_only_identity: bool = False


class IMM(NamedTuple):
    model: HybridMixture
    used_mask: Array


def infer_identity(imm, x, color_only_identity=False):
    # Weigh the color features more heavily
    x = x[:, -5:, :]  # (x: (B, 11, 1) -> (B, 5, 1))
    object_features = x[:, :].at[:, 2:].set(x[:, 2:] * 100)
    object_features = object_features[:, 2 * int(color_only_identity) :]
    _, c_ell, _ = imm.model._e_step(object_features, [])

    i_used_mask = imm.model.prior.alpha > imm.model.prior.prior_alpha
    elogp = (c_ell) * i_used_mask + (1 - i_used_mask) * (-1e10)
    qz = softmax(elogp, imm.model.mix_dims)

    class_labels = qz.argmax(-1)
    return class_labels


def create_imm(
    key: Array,
    num_object_types: int,
    num_features: int = 5,
    cont_scale_identity: float = 0.5,
    color_precision_scale=None,
    color_only_identity=False,
    **kwargs,
):
    key, subkey = jr.split(key)
    num_identity_features = num_features if not color_only_identity else 3
    identity_model = create_mm(
        subkey,
        num_components=num_object_types,
        continuous_dim=num_identity_features,
        discrete_dims=[],
        cont_scale=cont_scale_identity,
        color_precision_scale=color_precision_scale,
        opt={"lr": 1.0, "beta": 0.0},
    )

    i_used_mask = jnp.zeros(identity_model.continuous_likelihood.mean.shape[0])

    imm = IMM(
        model=identity_model,
        used_mask=i_used_mask,
    )
    return imm


def infer_remapped_color_identity(imm, obs, object_idx, num_features, **kwargs):
    # this method is called when we want to explicitly trigger remapping based on shape only
    object_features = obs[object_idx, None, obs.shape[-1] - num_features :, None]
    object_features = object_features.at[:, 2:, :].set(object_features[:, 2:, :] * 100)

    # check if the object is inferred from the given object features
    # if not, try shape only
    _, c_ell, _ = imm.model._e_step(object_features, [])
    i_used_mask = imm.model.prior.alpha > imm.model.prior.prior_alpha
    ell = (c_ell) * i_used_mask + (1 - i_used_mask) * (-1e10)

    def _infer_based_on_features():
        return jax.nn.softmax(ell)[0]

    def _infer_based_on_shape():
        # calculate ell using shape only, marginalizing color
        data = object_features[:, :2, :]

        mean = imm.model.continuous_likelihood.mean[:, :2, :]
        expected_inv_sigma = imm.model.continuous_likelihood.expected_inv_sigma()[
            :, :2, :2
        ]
        expected_logdet_inv_sigma = (
            imm.model.continuous_likelihood.expected_logdet_inv_sigma()[:, :2, :2]
        )
        dim = 2
        kappa = imm.model.continuous_likelihood.kappa

        diff = data - mean
        tx_dot_stheta = -0.5 * bdot(diff.mT, bdot(expected_inv_sigma, diff))
        atheta_1 = -0.5 * dim / kappa
        atheta_2 = 0.5 * expected_logdet_inv_sigma
        log_base_measure = -0.5 * dim * jnp.log(2 * jnp.pi)
        negative_expected_atheta = atheta_1 + atheta_2
        ell = imm.model.continuous_likelihood.sum_events(
            log_base_measure + tx_dot_stheta + negative_expected_atheta
        )

        qz = jax.nn.softmax(100 * ell)
        return qz

    qz = jax.lax.cond(ell.max() > -100, _infer_based_on_features, _infer_based_on_shape)
    return qz


def infer_and_update_identity(
    imm,
    obs,
    object_idx,
    num_features,
    i_ell_threshold,
    color_only_identity=False,
    **kwargs,
):
    object_features = obs[object_idx, None, obs.shape[-1] - num_features :, None]
    # scale the color features (esasier for the GMM to separate)
    object_features = object_features.at[:, 2:, :].set(object_features[:, 2:, :] * 100)
    object_features = object_features[:, 2 * int(color_only_identity) :, :]

    model_updated, used_mask, _ = train_step_fn(
        imm.model, imm.used_mask, object_features, [], logp_thr=i_ell_threshold
    )

    imm = eqx.tree_at(lambda x: x.model, imm, model_updated)
    imm = eqx.tree_at(lambda x: x.used_mask, imm, used_mask)
    return imm
