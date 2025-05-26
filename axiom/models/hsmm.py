from typing import Tuple, NamedTuple, Union, Sequence
from dataclasses import asdict
from jaxtyping import Array, PRNGKeyArray
from jax import vmap
from jax import random as jr, numpy as jnp
from jax.numpy import expand_dims as expand
from jax.scipy.linalg import block_diag

from .smm import (
    SMMConfig,
    SMM,
    create_smm,
    initialize_smm_model,
    infer_and_update as infer_and_update_smm,
    create_mvn
)

from axiom.vi import ArrayDict
from axiom.vi.exponential import MultivariateNormal as ExpMVN


class HierarchSMM(NamedTuple):
    models: Sequence[SMM]
    num_slots: Sequence[int]
    num_layers: int
    width: int
    height: int
    stats: dict


def create_hierarch_smm(key, layer_configs: Sequence[SMMConfig]) -> HierarchSMM:
    models = []
    num_slots = []
    for config in layer_configs:
        model = create_smm(key, **asdict(config))
        models.append(model)
        num_slots.append(model.num_slots)

    return HierarchSMM(
        models=models,
        num_slots=num_slots,
        num_layers=len(layer_configs),
        width=models[0].width,
        height=models[0].height,
        stats=models[0].stats,
    )


def initialize_hierarch_smm(
    key: PRNGKeyArray,
    model: HierarchSMM,
    init_inputs: Union[Array, ExpMVN],
    layer_configs: Sequence[SMMConfig],
) -> HierarchSMM:

    (
        models_updated,
        py_updated,
        qx_updated,
        qz_updated,
        used_updated,
        ell_max_updated,
    ) = ([], [], [], [], [], [])
        
    next_layer_input_dims = [2] * (len(layer_configs))

    for i, (m, layer_config) in enumerate(zip(model.models, layer_configs)):

        if i > 0:
            init_inputs = augment_state_with_velocity(init_inputs, prev_qx=None, velocity_precision=0.1)

        m, qx, _ = initialize_smm_model(
            m,
            init_inputs=(
                expand(init_inputs, 0)
                if isinstance(init_inputs, Array)
                else init_inputs
            ),
        )
        # Run a first step of inference and growing, in order to get an initial input for the higher layer
        key, subkey = jr.split(key)

        (m_updated, py, qx, qz, used, ell_max) = infer_and_update_smm(
            subkey,
            m,
            init_inputs,
            qx_prev=qx,
            **asdict(layer_config),
        )
        models_updated.append(m_updated)
        py_updated.append(py)
        qx_updated.append(qx)
        qz_updated.append(qz)
        used_updated.append(used)
        ell_max_updated.append(ell_max)

        # the inputs for the (i+1)-th  layer are the first (0, 1, ..., input_dim[i+1]) dimensions of the lower layer's decoded outputs
        inv_sigma_mu_i = py.nat_params.inv_sigma_mu.squeeze(0)[
            :, :, : next_layer_input_dims[i], :
        ]
        inv_sigma_i = py.nat_params.inv_sigma.squeeze(0)[
            :, :, : next_layer_input_dims[i], : next_layer_input_dims[i]
        ]
        init_inputs = ExpMVN(
            nat_params=ArrayDict(inv_sigma_mu=inv_sigma_mu_i, inv_sigma=inv_sigma_i),
            event_shape=(layer_config.slot_dim, 1),
            event_dim=2,
        )

    smm_updated = HierarchSMM(
        models=models_updated,
        num_slots=model.num_slots,
        num_layers=len(layer_configs),
        width=model.width,
        height=model.height,
        stats=model.stats,
    )

    return (
        smm_updated,
        py_updated,
        qx_updated,
        qz_updated,
        used_updated,
        ell_max_updated,
    )


def infer_and_update(
    key: PRNGKeyArray,
    model: HierarchSMM,
    inputs: Array,
    prev_qx: Sequence[ExpMVN],
    layer_configs: Sequence[SMMConfig],
    slots_to_pass_up: Sequence[Array],
) -> HierarchSMM:
    (
        models_updated,
        py_updated,
        qx_updated,
        qz_updated,
        used_updated,
        ell_max_updated,
    ) = ([], [], [], [], [], [])

    next_layer_position_dims = [2] * (len(layer_configs))
    for i, (m, qx_prev, layer_config) in enumerate(
        zip(model.models, prev_qx, layer_configs)
    ):

        key, subkey = jr.split(key)

        if i > 0:
            inputs = augment_state_with_velocity(inputs, prev_qx[i - 1], velocity_precision=5.0)

            inv_sigma_mu_select_slots = inputs.nat_params.inv_sigma_mu[:,slots_to_pass_up[i-1],:]
            inv_sigma_select_slots = inputs.nat_params.inv_sigma[:,slots_to_pass_up[i-1],:,:]

            inputs = create_mvn(inv_sigma_mu_select_slots, inv_sigma_select_slots)

            # mask = slots_to_pass_up[i-1]
            # default_inv_sigma_mu = jnp.full_like(inputs.nat_params.inv_sigma_mu, 1e4)
            # default_inv_sigma = jnp.broadcast_to(1e3*jnp.eye(inputs.dim), inputs.nat_params.inv_sigma.shape)
            
            # noop
            # default_inv_sigma_mu = inputs.nat_params.inv_sigma_mu
            # default_inv_sigma = inputs.nat_params.inv_sigma

            # low_precis_default = create_mvn(default_inv_sigma_mu, default_inv_sigma)
            # inputs = _combine_mvns(low_precis_default, inputs, mask[None, :, None, None])

        (m_updated, py, qx, qz, used, ell_max) = infer_and_update_smm(
            subkey,
            m,
            inputs,
            qx_prev=qx_prev,
            **asdict(layer_config),
        )

        models_updated.append(m_updated)
        py_updated.append(py)
        qx_updated.append(qx)
        qz_updated.append(qz)
        used_updated.append(used)
        ell_max_updated.append(ell_max)

        # the inputs for the (i+1)-th  layer are the first (0, 1, ..., input_dim[i+1]) dimensions of the lower layer's decoded outputs
        inv_sigma_mu_i = py.nat_params.inv_sigma_mu.squeeze(0)[
            :, :, : next_layer_position_dims[i], :
        ]
        inv_sigma_i = py.nat_params.inv_sigma.squeeze(0)[
            :, :, : next_layer_position_dims[i], : next_layer_position_dims[i]
        ]
        inputs = ExpMVN(
            nat_params=ArrayDict(inv_sigma_mu=inv_sigma_mu_i, inv_sigma=inv_sigma_i),
            event_shape=(layer_config.slot_dim, 1),
            event_dim=2,
        )

    smm_updated = HierarchSMM(
        models=models_updated,
        num_slots=model.num_slots,
        num_layers=model.num_layers,
        width=model.width,
        height=model.height,
        stats=model.stats,
    )

    return (
        smm_updated,
        py_updated,
        qx_updated,
        qz_updated,
        used_updated,
        ell_max_updated,
    )


def augment_state_with_velocity(qx: ExpMVN, prev_qx: ExpMVN = None, velocity_precision=1.0) -> ExpMVN:
    """
    Augment the latent state by computing a velocity term from the difference between the
    current state (qx.mu) and a previous state (prev_qx.mu), then concatenating the velocity
    to the original state. If prev_qx is None, velocity is zero.

    The latent state is assumed to be along axis 2 of qx.mu (e.g., shape (batch, components, latent_dim, 1)).
    The corresponding covariance has shape (batch, components, latent_dim, latent_dim).

    Args:
        qx: Current ExpMVN object with a `mu` property representing the current state.
        prev_qx: Previous ExpMVN object with a `mu` property representing the prior state.

    Returns:
        A new ExpMVN object whose state has been augmented to include the velocity component.
    """

    # Determine the latent state dimension (e.g., 4 in your printed shapes).
    latent_dim = qx.nat_params.inv_sigma.shape[-1]

    # Compute velocity as the difference between current and previous means,
    # or use zeros if no previous state is provided.
    if prev_qx is None:
        velocity = jnp.zeros_like(qx.mu)
    else:
        # In your case, prev_qx.mu has an extra dimension:
        # qx.mu shape:       (1, 16, 2, 1)
        # prev_qx.mu shape:   (1,  1, 16, 4, 1)
        # We assume that the extra axis (axis=1) in prev_qx is a singleton that needs to be removed.
        prev_mu = jnp.squeeze(prev_qx.mu, axis=1)  # shape: (1, 16, 4, 1)
        #

        prev_pos_mu = prev_mu[:, :, :2, :]  # shape: (1, 16, 2, 1)
        velocity = qx.mu - prev_pos_mu  # shape: (1, 16, 2, 1)

    # Concatenate the original state and its velocity along the latent state axis (axis=2)
    # augmented_mu = jnp.concatenate([qx.mu, velocity], axis=2)

    # Augment the natural parameter for the mean:
    # Pad inv_sigma_mu with zeros in the new latent state dimensions.
    augmented_inv_sigma_mu = jnp.concatenate(
        [qx.nat_params.inv_sigma_mu, velocity_precision*velocity], axis=2
    )

    # For the inverse covariance (inv_sigma), create a block-diagonal matrix.
    orig_inv_sigma = (
        qx.nat_params.inv_sigma
    )  # shape: (batch, components, latent_dim, latent_dim)
    batch_shape = orig_inv_sigma.shape[:-2]
    dim_dim_shape = orig_inv_sigma.shape[-2:]

    # Create the identity matrix for the velocity block.
    velocity_inv_sigma = jnp.broadcast_to(velocity_precision*jnp.eye(latent_dim), orig_inv_sigma.shape).reshape((-1,) + dim_dim_shape)  # shape: (batch*components, latent_dim, latent_dim)

    augmented_inv_sigma = vmap(block_diag)(orig_inv_sigma.reshape((-1,) + dim_dim_shape), velocity_inv_sigma) # shape (batch*components, 2*latent_dim, 2*latent_dim)
    augmented_dim_dim_shape = augmented_inv_sigma.shape[-2:] # now (2*latent_dim, 2*latent_dim)
    augmented_inv_sigma = augmented_inv_sigma.reshape(batch_shape + augmented_dim_dim_shape) # shape (batch, components, 2*latent_dim, 2*latent_dim)

    # Construct new natural parameters.
    new_nat_params = ArrayDict(
        inv_sigma_mu=augmented_inv_sigma_mu, inv_sigma=augmented_inv_sigma
    )
    event_shape = new_nat_params.inv_sigma_mu.shape[-2:]
    return ExpMVN(nat_params=new_nat_params, event_shape=event_shape)
