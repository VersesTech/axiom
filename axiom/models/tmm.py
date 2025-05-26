from typing import NamedTuple
from dataclasses import dataclass

import equinox as eqx

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax import lax
from jaxtyping import Array

from time import time


@dataclass(frozen=True)
class TMMConfig:
    """
    Configuration class for TMM model

    Attributes:
    n_total_components (int): Total number of components in the TMM model
    state_dim (int): Dimension of the state space, specifically the number of position coordinates (e.g., 2 for X/Y)
    dt (float): Time step for the TMM model
    sigma_sqr (float): Variance of the Gaussian likelihood
    logp_threshold (float): Threshold for the log probability
    position_threshold (float): Threshold for the position
    """

    n_total_components: int = 200
    state_dim: int = 2
    dt: float = 1.0
    vu: float = 0.05
    use_bias: bool = True
    sigma_sqr: float = 2.0
    logp_threshold: float = -0.00001
    position_threshold: float = 0.15
    use_unused_counter: bool = True

    # whether to use a velocity tmm (default)
    use_velocity: bool = True

    # clip small values in the transition matrix to 0
    clip_value: float = 5e-4


class TMM(NamedTuple):
    transitions: Array
    used_mask: Array


def generate_default_dynamics_component(state_dim, dt=1.0, use_bias=True):
    """
    state_dim: int
    dt: float
    use_bias: bool

    Returns:
    transition_matrix: (2*state_dim, (2*state_dim)+1)
    """

    # encodes the assumption that velocity is constant in time
    velocity_coupling = jnp.eye(state_dim)
    base_transitions = block_diag(jnp.eye(state_dim), velocity_coupling)
    transition_matrix = jnp.pad(base_transitions, [(0, 0), (0, 1)])
    transition_matrix = transition_matrix.at[:state_dim, state_dim:-1].set(
        jnp.diag(dt * jnp.ones(state_dim))
    )

    if not use_bias:
        transition_matrix = transition_matrix[:, :-1]

    # Remove unused velocity and bias (assume used)
    transition_matrix = transition_matrix.at[2, :].set(0)
    transition_matrix = transition_matrix.at[5, :].set(0)

    return transition_matrix


def generate_default_become_unused_component(state_dim, dt=1, vu=1, use_bias=True):
    transition_matrix = jnp.zeros((state_dim * 2, state_dim * 2 + int(use_bias)))
    transition_matrix = transition_matrix.at[: state_dim - 1, : state_dim - 1].set(
        jnp.eye(state_dim - 1)
    )
    # set the unused component to dt * vu
    transition_matrix = transition_matrix.at[state_dim - 1, -1].set(dt * vu)

    if not use_bias:
        transition_matrix = transition_matrix[:, :-1]

    return transition_matrix


def generate_default_keep_unused_component(state_dim, dt=1, vu=1, use_bias=True):
    transition_matrix = jnp.zeros((state_dim * 2, state_dim * 2 + int(use_bias)))
    transition_matrix = transition_matrix.at[:state_dim, :state_dim].set(
        jnp.eye(state_dim)
    )
    transition_matrix = transition_matrix.at[
        state_dim : 2 * state_dim - 1, state_dim : 2 * state_dim - 1
    ].set(jnp.eye(state_dim - 1))

    # set the unused component to dt * vu
    transition_matrix = transition_matrix.at[state_dim - 1, -1].set(dt * vu)

    if not use_bias:
        transition_matrix = transition_matrix[:, :-1]

    return transition_matrix


def generate_default_stop_component(state_dim, use_bias=True):
    transition_matrix = jnp.zeros((state_dim * 2, state_dim * 2 + int(use_bias)))
    transition_matrix = transition_matrix.at[:2, :2].set(jnp.eye(2))
    return transition_matrix


def create_velocity_component(x_current, x_next, dt=1.0, use_unused_counter=True):
    state_dim = x_current.shape[-1] // 2
    base_dynamics = generate_default_dynamics_component(
        state_dim=state_dim, dt=dt, use_bias=True
    )
    # x = x_prev + vel_prev + vel_bias
    # v = vel_prev + vel_bias
    vel = x_next[:state_dim] - x_current[:state_dim]
    prev_vel = x_current[state_dim:]
    vel_bias = vel - prev_vel

    new_component = base_dynamics.at[:, -1].set(jnp.concatenate(2 * [vel_bias]))

    # No unused here
    if use_unused_counter:
        new_component = new_component.at[state_dim - 1, :].set(0)
        new_component = new_component.at[2 * state_dim - 1, :].set(0)

    return new_component


def create_bias_component(x, use_unused_counter):
    state_dim_with_vel = x.shape[-1]
    new_component = jnp.concatenate(
        [jnp.zeros((state_dim_with_vel, state_dim_with_vel)), x[..., None]], axis=-1
    )

    # No unused here
    if use_unused_counter:
        new_component = new_component.at[state_dim_with_vel // 2 - 1, :].set(0)
        new_component = new_component.at[state_dim_with_vel - 1, :].set(0)
    return new_component


def create_position_velocity_component(transitions, x_prev, x_curr, use_unused_counter):
    num_coords = x_curr.shape[-1] // 2 - int(use_unused_counter)

    vel = x_curr[:num_coords] - x_prev[:num_coords]
    new_component = jnp.zeros_like(transitions[0])

    new_component = new_component.at[:num_coords, :num_coords].set(jnp.eye(num_coords))
    new_component = new_component.at[:num_coords, -1].set(vel)

    new_component = new_component.at[
        num_coords + int(use_unused_counter) : 2 * num_coords + int(use_unused_counter),
        -1,
    ].set(vel)
    return new_component


def create_position_bias_component(transitions, x_prev, x_curr):
    new_component = jnp.zeros_like(transitions[0])
    new_component = new_component.at[:2, -1].set(x_curr[:2])
    return new_component


def add_component(existing_transitions, new_transition, used_mask):
    """
    existing_transitions: (K_max, ...)
    new_transition: shape (...)
    used_mask: shape (K_max,) - True if that row is already used
    """

    # Boolean for the first row that is False in `used_mask`
    # In effect, "the first slot that is free"
    first_unused_mask = jnp.logical_and(~used_mask, jnp.cumsum(~used_mask) == 1)

    # Expand dims so that it can broadcast to the shape of new_transition
    # E.g. if transitions are (K_max, state_dim, state_dim+1),
    # then expand to (K_max, 1, 1) or (K_max, 1, ...) so broadcast matches.
    # We'll assume new_transition is (state_dim, state_dim+1):
    mask_expanded = first_unused_mask[..., None, None]  # shape (K_max, 1, 1)
    update_tensor = (
        mask_expanded * new_transition
    )  # shape (K_max, state_dim, state_dim+1)

    return existing_transitions + update_tensor


def forward(transitions, x):
    """
    transitions: (K_max, 2*state_dim, (2*state_dim)+1)
    x: (2*state_dim,)
    """

    return (transitions[..., :-1] * x).sum(-1) + transitions[..., -1]


def gaussian_loglike(y, mu, sigma_sqr=2):
    """
    y: (..., D)
    mu: (..., K, D)
    sigma_sqr: (...,K)
    """
    squared_error = (y - mu) ** 2
    return -0.5 * squared_error.sum(axis=-1) / sigma_sqr - 0.5 * y.shape[-1] * jnp.log(
        2 * jnp.pi * sigma_sqr
    )
    # return -squared_error.mean(axis=-1)


def compute_logprobs(transitions, x_prev, x_curr, sigma_sqr=2, use_velocity=True):
    """
    transitions: (K_max, 2*state_dim, (2*state_dim)+1)
    x_prev: (2*state_dim,)
    x_curr: (2*state_dim,)
    sigma_sqr: float
    """

    mu = forward(transitions, x_prev)
    if use_velocity:
        return gaussian_loglike(x_curr, mu, sigma_sqr)
    else:
        return gaussian_loglike(x_curr.at[:3], mu[:, :3], sigma_sqr)


def add_vel_or_bias_component(
    transitions,
    x_prev,
    x_curr,
    used_mask,
    pos_thr,
    use_unused_counter=False,
    dt=1.0,
    use_velocity=True,
    clip_value=5e-4,
):
    state_dim = x_curr.shape[-1] // 2 - int(use_unused_counter)

    # conditions for creating a bias component
    teleport = jnp.linalg.norm(x_curr[:state_dim] - x_prev[:state_dim]) > pos_thr

    if use_velocity:
        component_to_add = lax.cond(
            teleport,
            lambda x: create_bias_component(x, use_unused_counter),
            lambda x: create_velocity_component(x_prev, x, dt, use_unused_counter),
            x_curr,
        )
    else:
        component_to_add = lax.cond(
            teleport,
            lambda x: create_position_bias_component(
                transitions, x_prev, x, use_unused_counter
            ),
            lambda x: create_position_velocity_component(
                transitions, x_prev, x, use_unused_counter
            ),
            x_curr,
        )

    # filter out noisy values below clip_value
    component_to_add = jnp.where(
        jnp.abs(component_to_add) < clip_value, 0.0, component_to_add
    )

    return add_component(transitions, component_to_add, used_mask)


def single_logprob(transition, x_prev, x_curr, sigma_sqr):
    """
    transition: shape (2*state_dim, 2*state_dim + 1)
    x_prev: shape (2*state_dim,)
    x_curr: shape (2*state_dim,)
    sigma_sqr: float
    """
    mu = forward(transition, x_prev)  # shape (2*state_dim,)
    return gaussian_loglike(x_curr, mu, sigma_sqr)


def compute_logprobs_masked_vmap(transitions, used_mask, x_prev, x_curr, sigma_sqr):
    def compute_logprob_if_used(transition, mask, x_p, x_c):
        return lax.cond(
            mask,
            lambda t: single_logprob(t, x_p, x_c, sigma_sqr),  # <— pass sigma_sqr
            lambda _: -jnp.inf,
            transition,
        )

    return jax.vmap(compute_logprob_if_used, in_axes=(0, 0, None, None))(
        transitions, used_mask, x_prev, x_curr
    )


def compute_logprobs_masked_fori(transitions, used_mask, x_prev, x_curr, sigma_sqr=2):
    """
    transitions: (K_max, 2*state_dim, 2*state_dim + 1)
    used_mask:   (K_max,)
    x_prev, x_curr: (2*state_dim,)
    sigma_sqr: float
    Returns: logps of shape (K_max,)
             logps[i] = single_logprob if used_mask[i] else -∞
    """
    K_max = transitions.shape[0]

    def body_fn(i, logps):
        logp_i = jax.lax.cond(
            used_mask[i],
            lambda t: single_logprob(t, x_prev, x_curr, sigma_sqr),
            lambda _: -jnp.inf,
            transitions[i],
        )
        return logps.at[i].set(logp_i)

    logps_init = jnp.full((K_max,), -jnp.inf)
    logps = jax.lax.fori_loop(0, K_max, body_fn, logps_init)
    return logps


def update_transitions(
    transitions,
    x_prev,
    x_curr,
    used_mask,
    sigma_sqr=2.0,
    logp_thr=-0.001,
    pos_thr=0.5,
    dt=1.0,
    use_unused_counter=False,
    use_velocity=True,
    clip_value=5e-4,
):
    """
    transitions: (K_max, 2*state_dim, (2*state_dim)+1)
    x_prev: (2*state_dim,)
    x_curr: (2*state_dim,)
    sigma_sqr: float
    """

    ### OPTION 1: FIRST COMPTUE LOGPROBS FOR ALL, THEN MASK to -∞ WITH USED_MASK AFTERWARDS
    # shape (K_used,)
    logprobs_all = compute_logprobs(
        transitions, x_prev, x_curr, sigma_sqr, use_velocity
    )

    # 2) Replace the "unused" transitions’ logprobs with -∞
    logprobs_used = jnp.where(used_mask, logprobs_all, -jnp.inf)

    ### OPTION 2: VMAP AND COND ACROSS K_MAX COMPONENTS
    # logprobs_used = compute_logprobs_masked_vmap(
    #     transitions, used_mask, x_prev, x_curr, sigma_sqr
    # )

    ### OPTION 3: FORI_LOOP WITH COND, ACROSS COMPONENTS (NOTE: SLOW)
    # logprobs_used = compute_logprobs_masked_fori(
    #     transitions, used_mask, x_prev, x_curr, sigma_sqr
    # )

    # 3) The maximum over used transitions
    max_used = logprobs_used.max()  # single scalar

    # 4) If the max used is < threshold, we add new component (which itself will dispatch to adding either a bias or velocity component)
    def add_component_case(trans):
        return add_vel_or_bias_component(
            trans,
            x_prev,
            x_curr,
            used_mask,
            pos_thr,
            dt=dt,
            use_unused_counter=use_unused_counter,
            use_velocity=use_velocity,
            clip_value=clip_value,
        )

    def no_op_case(trans):
        return trans

    transitions = jax.lax.cond(
        max_used < logp_thr, add_component_case, no_op_case, transitions
    )

    used_mask = jnp.sum(jnp.abs(transitions), axis=(-1, -2)) > 0

    # recompute logprobs with new mask
    logprobs_all = compute_logprobs(
        transitions, x_prev, x_curr, sigma_sqr, use_velocity
    )
    logprobs_used = jnp.where(used_mask, logprobs_all, -jnp.inf)

    return transitions, used_mask, logprobs_used


def create_tmm(
    key,
    n_total_components,
    state_dim,
    dt=1.0,
    vu=0.1,
    use_bias=True,
    use_velocity=True,
    **kwargs,
):
    """
    n_total_components: int
    state_dim: int
    dt: float
    """

    transitions = jnp.zeros(
        (n_total_components, 2 * state_dim, 2 * state_dim + int(use_bias))
    )
    used_mask = jnp.zeros(n_total_components, dtype=bool)

    if use_velocity:
        # In case we use velocity, we initialize some sensible default components
        transitions = transitions.at[0].set(
            generate_default_dynamics_component(state_dim, dt=dt, use_bias=use_bias)
        )
        transitions = transitions.at[1].set(
            generate_default_keep_unused_component(
                state_dim, dt=dt, vu=vu, use_bias=use_bias
            )
        )
        transitions = transitions.at[2].set(
            generate_default_become_unused_component(
                state_dim, dt=dt, vu=vu, use_bias=use_bias
            )
        )
        transitions = transitions.at[3].set(
            generate_default_stop_component(state_dim, use_bias)
        )

        used_mask = used_mask.at[:4].set(True)
    else:
        transitions = transitions.at[0].set(
            generate_default_keep_unused_component(
                state_dim, dt=dt, vu=vu, use_bias=use_bias
            )
        )
        transitions = transitions.at[1].set(
            generate_default_become_unused_component(
                state_dim, dt=dt, vu=vu, use_bias=use_bias
            )
        )
        used_mask = used_mask.at[:2].set(True)

    return TMM(transitions=transitions, used_mask=used_mask)


def update_model(
    model,
    x_prev,
    x_curr,
    state_dim=2,
    sigma_sqr=2.0,
    logp_threshold=-0.001,
    position_threshold=0.5,
    dt=1.0,
    use_unused_counter=False,
    use_velocity=True,
    clip_value=5e-4,
    **kwargs,
):
    """
    transitions: (K_max, 2*state_dim, (2*state_dim)+1)
    x_prev: (2*state_dim,)
    x_curr: (2*state_dim,)
    sigma_sqr: float
    """
    logp_threshold_adjust = logp_threshold * (
        1.0 / 2 * sigma_sqr
    ) - 0.5 * 2 * state_dim * jnp.log(2 * jnp.pi * sigma_sqr)

    new_transitions, new_used_mask, logprobs = update_transitions(
        model.transitions,
        x_prev,
        x_curr,
        model.used_mask,
        sigma_sqr,
        logp_threshold_adjust,
        position_threshold,
        dt=dt,
        use_unused_counter=use_unused_counter,
        use_velocity=use_velocity,
        clip_value=clip_value,
    )

    model = eqx.tree_at(
        lambda x: (x.transitions, x.used_mask), model, (new_transitions, new_used_mask)
    )

    return model, logprobs
