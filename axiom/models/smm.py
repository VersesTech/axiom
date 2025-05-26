import jax
from jax import numpy as jnp, random as jr
from jax.nn import softmax, one_hot
from jax import lax
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
from typing import Tuple, NamedTuple, Union
from dataclasses import dataclass, field

from axiom.vi.exponential import MultivariateNormal
from axiom.vi.transforms import LinearMatrixNormalGamma
from axiom.vi import ArrayDict, Distribution
from axiom.vi.utils import bdot


from axiom.vi.models.slot_mixture_model import (
    SlotMixtureModel,
    _inputs_to_delta,
    _m_step_keep_unused,
)


@dataclass(frozen=True)
class SMMConfig:
    """
    Configuration class for SMM model.

    Attributes:
        width (int): The width of the input image.
        height (int): The height of the input image.
        input_dim (int): The dimension of the input feature vector.
        slot_dim (int): The dimension of each slot latent.
        num_slots (int): The number of slots.
        use_bias (bool): Whether to use bias in the model.
        ns_a (float): The noise parameter of the transform on the mean of the linear transform.
        ns_b (float): The noise parameter of the bias on the mean of the linear transform.
        dof_offset (float): The offset value for the degrees of freedom.
        mask_prob (list[float]): The probability of using the linear template corresponding to the respective index.
        scale (list[float]): The scaling factor for each element of the prior on the bias.
        transform_inv_v_scale (float): The scaling factor for the inverse variance of the linear transform.
        bias_inv_v_scale (float): The scaling factor for the inverse variance of the bias vector.
        learning_rate (float): The learning rate for the model updates.
        beta (float): The beta value for the model updates.
        elbo_threshold (float): Threshold of the ELBO below which we grow the model,
        max_grow_steps (int): Maximum number of grow steps per infer_and_update call.
    """

    width: int = 160
    height: int = 210
    input_dim: int = 5
    slot_dim: int = 2
    num_slots: int = 32
    use_bias: bool = True
    ns_a: float = 1
    ns_b: float = 1
    dof_offset: float = 10
    mask_prob: tuple[float] = field(
        default_factory=lambda: tuple([0.0, 0.0, 0.0, 0.0, 1])
    )
    scale: tuple[float] = field(
        default_factory=lambda: tuple([0.075, 0.075, 0.75, 0.75, 0.75])
    )
    transform_inv_v_scale: float = 100
    bias_inv_v_scale: float = 0.001
    num_e_steps: int = 2
    learning_rate: float = 1.0
    beta: float = 0.0
    eloglike_threshold: float = 5.0
    max_grow_steps: int = 20


class SMM(NamedTuple):
    model: SlotMixtureModel
    num_slots: int
    width: int
    height: int
    stats: dict


def _create_slot_mask(
    key: PRNGKeyArray,
    input_dim: int,
    slot_dim: int,
    use_bias: bool,
    num_components: int,
    transform_dims: list[int],
    probs: list[float] = None,
):
    m = jnp.zeros((5, input_dim, slot_dim + int(use_bias)))
    # bias component
    m = m.at[0, :, -1:].set(1)
    # scale component
    m = m.at[1, :, :-1].set(1)
    # Bias for color, scale for position component
    m = m.at[2, transform_dims, :-1].set(1)
    non_transform_dims = jnp.setdiff1d(
        jnp.arange(input_dim), jnp.asarray(transform_dims)
    )
    m = m.at[2, non_transform_dims, -1].set(1)
    # full component
    m = m.at[3, :, :].set(1)

    # Identity for top left len(transform_dims)xlen(transform_dims) matrix, bias for the bottom right column
    # Construct identity matrix elements
    identity_indices = (
        jnp.array([4] * len(transform_dims)),
        transform_dims,
        transform_dims,
    )
    identity_values = jnp.ones(len(transform_dims))

    # Set identity elements
    m = m.at[identity_indices].set(identity_values)
    m = m.at[4, non_transform_dims, -1:].set(1)

    if probs is None:
        probs = jnp.ones(m.shape[0]) / m.shape[0]
    else:
        probs = jnp.asarray(probs)
        assert probs.shape[0] == m.shape[0], (
            f"Probs shape should equal {m.shape[0]} but is instead {probs.shape[0]}"
        )

    key, subkey = jr.split(key)
    p = jr.choice(subkey, jnp.arange(m.shape[0]), p=probs, shape=(num_components,))

    return p, m[p]


def _create_lmg_params(
    key: PRNGKeyArray,
    input_dim: int,
    slot_dim: int,
    num_slots: int,
    use_bias: bool,
    ns_a: float,
    ns_b: float,
    dof_offset: float,
    mask_prob: list[float],
    scale: list[float],
    transform_inv_v_scale: float,
    bias_inv_v_scale: float,
) -> Tuple[ArrayDict, ArrayDict]:
    """
    Creates the parameters for the Linear Matrix Gamma conjugate prior

    Returns:
        Tuple[ArrayDict, ArrayDict]: The prior and initial posterior parameters.
    """

    # TODO
    if input_dim == 5:
        transform_dims = [0, 1]
    else:
        transform_dims = [0, 1, 2, 3]

    if mask_prob is not None:
        key, subkey = jr.split(key)
        _, slot_mask = _create_slot_mask(
            subkey,
            input_dim,
            slot_dim,
            use_bias,
            num_slots,
            transform_dims=transform_dims,
            probs=jnp.asarray(mask_prob).flatten(),
        )
    else:
        slot_mask = 1

    batch_shape = (1, num_slots)
    use_bias = int(use_bias)
    mu_pri = jnp.zeros((input_dim, slot_dim + use_bias))
    prior_mu = jnp.broadcast_to(mu_pri, batch_shape + (input_dim, slot_dim + use_bias))

    inv_v_scale = jnp.ones(slot_dim + use_bias)
    inv_v_scale = inv_v_scale.at[:slot_dim].set(transform_inv_v_scale)
    if use_bias:
        inv_v_scale = inv_v_scale.at[-1].set(bias_inv_v_scale)

    prior_inv_v = jnp.broadcast_to(
        jnp.eye(slot_dim + use_bias) * inv_v_scale,
        batch_shape + (slot_dim + use_bias, slot_dim + use_bias),
    )

    # shape should be (batch_shape,) + (1, 1)
    prior_a = jnp.full(batch_shape + (input_dim, 1), 1.0 + dof_offset)

    # shape should be (batch_shape,) + (y_dim, y_dim).
    scale = scale
    if isinstance(scale, tuple):
        scale = jnp.asarray(scale)
        assert scale.shape[0] == input_dim
    else:
        scale = scale * jnp.ones((input_dim))

    prior_b = jnp.broadcast_to(
        (scale**2).reshape((input_dim, 1)), batch_shape + (input_dim, 1)
    )

    # if there is a strong prior on the slot mask template, this should be encoded in
    # the prior
    if transform_inv_v_scale > 1:
        prior_mu = prior_mu + slot_mask

    prior_params = ArrayDict(mu=prior_mu, inv_v=prior_inv_v, a=prior_a, b=prior_b)

    mu = prior_mu
    key, subkey = jr.split(key)
    mu_noise = jr.uniform(subkey, shape=prior_mu.shape, minval=1, maxval=2)

    mu = mu.at[..., :-1].set((mu[..., :-1] + ns_a * mu_noise[..., :-1]))
    mu = mu.at[..., -1].set((mu[..., -1] + ns_b * mu_noise[..., -1]))

    key, subkey = jr.split(key)
    a = prior_a * jr.uniform(subkey, shape=prior_a.shape, minval=1, maxval=3)

    inv_v = jnp.where(prior_params.inv_v > 0, 1.0, 0.0)
    initial_posterior_params = ArrayDict(mu=mu * slot_mask, inv_v=inv_v, a=a, b=prior_b)

    return prior_params, initial_posterior_params


def create_smm(
    key: PRNGKeyArray,
    width: int = 160,
    height: int = 210,
    input_dim: int = 5,
    slot_dim: int = 2,
    num_slots: int = 32,
    use_bias: bool = True,
    ns_a: float = 1,
    ns_b: float = 1,
    dof_offset: float = 10,
    mask_prob: list[float] = [0.0, 0.0, 0.0, 0.0, 1],
    scale: list[float] = [0.075, 0.075, 0.75, 0.75, 0.75],
    transform_inv_v_scale: float = 100,
    bias_inv_v_scale: float = 0.001,
    **kwargs,
):
    key, subkey = jr.split(key)

    prior_params, params = _create_lmg_params(
        key=subkey,
        input_dim=input_dim,
        slot_dim=slot_dim,
        num_slots=num_slots,
        use_bias=use_bias,
        ns_a=ns_a,
        ns_b=ns_b,
        dof_offset=dof_offset,
        mask_prob=mask_prob,
        scale=scale,
        transform_inv_v_scale=transform_inv_v_scale,
        bias_inv_v_scale=bias_inv_v_scale,
    )

    likelihood = LinearMatrixNormalGamma(
        params=params, prior_params=prior_params, use_bias=use_bias
    )

    smm = SlotMixtureModel(
        num_slots,
        input_dim=input_dim,
        slot_dim=slot_dim,
        multi_modality=False,
        likelihood=likelihood,
    )

    return SMM(
        model=smm,
        num_slots=num_slots,
        width=width,
        height=height,
        stats={
            "offset": jnp.array([width / 2, height / 2, 128, 128, 128]),
            "stdevs": jnp.array([width / 2, height / 2, 128, 128, 128]),
        },
    )


def add_position_encoding(img):
    u, v = jnp.meshgrid(jnp.arange(img.shape[1]), jnp.arange(img.shape[0]))

    data = jnp.concatenate(
        [
            (u.reshape(-1, 1)),
            (v.reshape(-1, 1)),
            img.reshape(-1, img.shape[-1]),
        ],
        axis=1,
    )
    return data


def format_single_frame(single_obs, offset, stdevs):
    """
    Format a single observation frame for the SMM model.
    """
    height, width, n_channels = single_obs.shape
    obs_with_xy = add_position_encoding(single_obs)
    obs_with_xy = obs_with_xy - offset
    obs_with_xy = obs_with_xy / stdevs
    return obs_with_xy.reshape((height * width, n_channels + 2))


def _compute_qx_given_qz(linear_likelihood, prior_x, inputs, qz):
    bkwd_message = linear_likelihood.variational_backward(inputs)

    inv_sigma_mu = (bkwd_message.inv_sigma_mu * qz[..., None, None]).sum(-4)
    inv_sigma = (bkwd_message.inv_sigma * qz[..., None, None]).sum(-4) + 1e-8

    qx = (
        MultivariateNormal(
            nat_params=ArrayDict(inv_sigma_mu=inv_sigma_mu, inv_sigma=inv_sigma),
            event_shape=prior_x.event_shape,
            event_dim=2,
        )
        * prior_x
    )
    return qx.expand_batch_shape(-2)


def _compute_qz_given_qx(linear_likelihood, prior_z_logmean, inputs, qx):
    ell = linear_likelihood.average_energy((qx, inputs))

    return softmax(ell + prior_z_logmean, axis=-1), ell


def _compute_qx_and_qz_given_qx(
    linear_likelihood, prior_z_logmean, prior_x, inputs, qx
):
    qz, ell = _compute_qz_given_qx(linear_likelihood, prior_z_logmean, inputs, qx)

    qx = _compute_qx_given_qz(linear_likelihood, prior_x, inputs, qz)

    return qx, qz, ell


def _update_optimized(
    smm: SMM, inputs, qz, learning_rate=1.0, beta=0.0, grow_mask=None
):
    linear_likelihood, prior_z, prior_x = (
        smm.model.likelihood,
        smm.model.pi,
        smm.model.px,
    )

    # first update qx given qz
    qx = _compute_qx_given_qz(linear_likelihood, prior_x, inputs, qz)

    # do an M step given the updated qx and input qz
    linear_likelihood, prior_z = _m_step_keep_unused(
        inputs,
        linear_likelihood,
        prior_z,
        qx,
        qz,
        lr=learning_rate,
        beta=beta,
        grow_mask=grow_mask,
    )

    pi_log_mean = prior_z.log_mean()

    # do e-m step
    qx, qz, ell = _compute_qx_and_qz_given_qx(
        linear_likelihood, jnp.zeros_like(pi_log_mean), prior_x, inputs, qx
    )

    linear_likelihood, prior_z = _m_step_keep_unused(
        inputs, linear_likelihood, prior_z, qx, qz, lr=learning_rate, beta=beta
    )

    # do another e-m step
    pi_log_mean = prior_z.log_mean()

    qx, qz, ell = _compute_qx_and_qz_given_qx(
        linear_likelihood, pi_log_mean, prior_x, inputs, qx
    )

    linear_likelihood, prior_z = _m_step_keep_unused(
        inputs, linear_likelihood, prior_z, qx, qz, lr=learning_rate, beta=beta
    )

    # Get the hard assignments (shape: (1, H*W))
    assignments = qz.argmax(-1)

    # Compute the histogram of assignments over the K components (shape: (K,))
    counts = jnp.bincount(
        assignments.ravel(), minlength=qz.shape[-1], length=qz.shape[-1]
    )

    # Convert to boolean (1 if component is used, 0 otherwise)
    used_component_idx = counts != 0

    smm = eqx.tree_at(
        lambda x: (x.model.likelihood, x.model.pi),
        smm,
        (linear_likelihood, prior_z),
    )

    # (1, H*W) array of integers, which component got selected for each pixel
    ell_max = ell.max(axis=-1)

    # NOTE: leave out returning full ell until we have the ell-based growing code working - CH
    return smm, qx, qz, ell_max, used_component_idx


def initialize_smm_model(smm: SMM, init_inputs: Union[Array, Distribution]):
    N_tokens = init_inputs.shape[1]
    if isinstance(init_inputs, Array):
        # convert to Delta distributions as input for the model
        init_inputs = _inputs_to_delta(init_inputs)
    else:
        init_inputs = init_inputs.expand_batch_shape(-1)

    # force assign all data to the first slot
    qz = one_hot(jnp.zeros(N_tokens), smm.num_slots)[None, ...]

    smm_model, qx, qz, _, _ = _update_optimized(
        smm, init_inputs, qz, learning_rate=1.0, beta=0.0
    )

    return smm_model, qx, qz


def create_mvn(inv_sigma_mu, inv_sigma):
    """
    Helper function for quickly creating an optimus.exponential.MultivariateNormal instance
    with given natural parameters inv_sigma_mu and inv_sigma
    """
    return MultivariateNormal(
        nat_params=ArrayDict(inv_sigma_mu=inv_sigma_mu, inv_sigma=inv_sigma),
        event_shape=(inv_sigma_mu.shape[-2], 1),
        event_dim=2,
    )


def _combine_mvns(model_0, model_1, select_from_1):
    def fn(x_0, x_1):
        return jnp.where(select_from_1, x_1, x_0)

    model_0.nat_params = jtu.tree_map(
        lambda x, y: fn(x, y), model_0.nat_params, model_1.nat_params
    )
    return model_0


def _increase_position_uncertainty(qx):
    """
    Sets the covariance of the position dimensions to be very large in order
    to have more emphasis on the color features of qx.
    NOTE: not properly tested:w

    """
    mu, inv_si = qx.mu, qx.inv_sigma.at[:, 0, 0].set(3).at[:, 1, 1].set(3)
    return MultivariateNormal(
        nat_params=ArrayDict(inv_sigma_mu=bdot(inv_si, mu), inv_sigma=inv_si)
    )


def create_e_step_fn(smm: SMM, inputs):
    """
    Given a fixed SMM model and some observations `input`, returns a function that can
    be passed into a lax.scan to perform a sequence of E-steps on the model to update qx
    and qz
    """

    linear_likelihood, prior_z_logmean, prior_x = (
        smm.model.likelihood,
        smm.model.pi.log_mean(),
        smm.model.px,
    )

    def e_step_scan_fn(carry, _):
        qx_prev, _, _ = carry

        qx, qz, ell = _compute_qx_and_qz_given_qx(
            linear_likelihood, prior_z_logmean, prior_x, inputs, qx_prev
        )
        ell_max = ell.max(axis=-1)

        # pass through to qx
        # qx_prev = _increase_position_uncertainty(qx_prev)
        select_from_qx = (qz[0].sum(0) > 0).reshape(1, 1, -1, 1, 1)
        qx = _combine_mvns(qx_prev, qx, select_from_qx)

        return (qx, qz, ell_max), None

    return e_step_scan_fn


def infer_and_update(
    key: PRNGKeyArray,
    smm: SMM,
    inputs: Union[Array, Distribution],
    qx_prev: MultivariateNormal,
    num_slots: int,
    num_e_steps=2,
    eloglike_threshold: float = 5.0,
    max_grow_steps: int = 10,
    learning_rate: float = 1.0,
    beta: float = 0.0,
    **kwargs,
):
    if isinstance(inputs, Array):
        N_tokens = inputs.shape[0]
        # rescale and normalize
        # add a leading dimension because format_frames expects at least one sample dimension
        inputs = _inputs_to_delta(inputs[None, ...])
    else:
        N_tokens = inputs.shape[1]
        inputs = inputs.expand_batch_shape(-1)

    ### Do E steps to update qx and qz, and to compute the maximum eloglike across components
    e_step_scan_fn = create_e_step_fn(smm, inputs)

    init = (
        qx_prev,
        jnp.empty((1, N_tokens, num_slots)),
        jnp.empty((1, N_tokens)),
    )
    (qx, qz, ell_max), _ = lax.scan(e_step_scan_fn, init, jnp.arange(num_e_steps))

    # Get the hard assignments (shape: (1, H*W))
    assignments = qz.argmax(-1)

    # Compute the histogram of assignments over the K components (shape: (K,))
    counts = jnp.bincount(assignments.ravel(), minlength=num_slots, length=num_slots)

    # Convert to boolean (1 if component is used, 0 otherwise)
    used_component_idx = counts != 0

    def grow_body(carry, t):
        """
        carry = (smm, qx, qz, ell_max, ell, used, tries, done)
        """
        done_last = carry[-1]

        # ---------- Utility functions -------------
        def do_nothing(carry_in):
            # just return carry_in as is, or you might prefer setting done=True
            smm_i, qx_i, qz_i, ell_max_i, used_i, tries_i, done_i = carry_in
            # keep the rest, set done=True
            return (smm_i, qx_i, qz_i, ell_max_i, used_i, tries_i, True)

        def do_grow(carry_in):
            ssmm_i, qx_i, qz_i, ell_max_i, used_i, tries_i, done_i = carry_in
            new_tries = tries_i + 1
            cond_ok = (ell_max_i.min() < eloglike_threshold) & (~done_i)

            def skip_grow(c2):
                smm_j, qx_j, qz_j, ell_max_j, used_j, tries_j, done_j = c2
                return (smm_j, qx_j, qz_j, ell_max_j, used_j, new_tries, True)

            def really_grow(c2):
                smm_j, qx_j, qz_j, ell_max_j, used_j, tries_j, done_j = c2

                slot_candidates = jnp.where(~used_j, jnp.arange(num_slots), num_slots)
                slot_to_use = slot_candidates.min()

                # Pick the component that best explains this datapoint from the unused
                # components to enforce slot consistency.
                # idx = jnp.argmin(ell_max_j)
                # ells_corrected = ell_max_j[0, idx] * (~used_j) + jnp.inf * used_j
                # slot_to_use = ells_corrected.argmin()

                def no_unused(c3):
                    # no unused => we are done
                    smm_k, qx_k, qz_k, ell_max_k, used_k, tries_k, done_k = c3
                    return (
                        smm_k,
                        qx_k,
                        qz_k,
                        ell_max_k,
                        used_k,
                        new_tries,
                        True,
                    )

                def force_assign(c3):
                    smm_k, qx_k, qz_k, ell_max_k, used_k, tries_k, done_k = c3
                    idx = jnp.argmin(ell_max_k)

                    grow_mask = jax.nn.one_hot(slot_to_use, num_slots)
                    new_qz = qz_k.at[0, idx].set(grow_mask)

                    new_smm, new_qx, newest_qz, new_ell_max, new_used = (
                        _update_optimized(
                            smm_k,
                            inputs,
                            new_qz,
                            learning_rate=learning_rate,
                            beta=beta,
                            grow_mask=grow_mask,
                        )
                    )
                    return (
                        new_smm,
                        new_qx,
                        newest_qz,
                        new_ell_max,
                        new_used,
                        new_tries,
                        False,
                    )

                # return lax.cond(slot_to_use == num_slots, no_unused, force_assign, c2)
                return lax.cond(used_j.sum() == num_slots, no_unused, force_assign, c2)

            return lax.cond(cond_ok, really_grow, skip_grow, carry_in)

        # ---------- Now choose between "done" and "grow" ----------
        new_carry = lax.cond(done_last, do_nothing, do_grow, carry)
        return new_carry, None

    def grow_loop(smm, qx, qz, ell_max, used, max_grow_steps):
        init_carry = (smm, qx, qz, ell_max, used, 0, False)
        final_carry, _ = jax.lax.scan(grow_body, init_carry, jnp.arange(max_grow_steps))
        return final_carry  # (smm, qx, qz, ell_max, used, tries, done)

    (
        smm_updated,
        qx_updated,
        qz_updated,
        ell_max_updated,
        used_updated,
        tries,
        done,
    ) = grow_loop(smm, qx, qz, ell_max, used_component_idx, max_grow_steps)

    # return the variational forward mean as object descriptors
    py = smm_updated.model.likelihood.variational_forward(qx_updated)

    return (
        smm_updated,
        py,
        qx_updated,
        qz_updated,
        used_updated,
        ell_max_updated,
    )
