from dataclasses import dataclass, field
from functools import partial
from typing import NamedTuple, List, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax.nn as nn
from jax.nn import softmax
from jaxtyping import Array

import equinox as eqx

from axiom.models import tmm as tmm_tools
from axiom.models import imm as imm_tools

from axiom.vi.models.hybrid_mixture_model import HybridMixture
from axiom.models.utils_hybrid import create_mm, train_step_fn


@dataclass(frozen=True)
class RMMConfig:
    """
    Configuration for the RMM
    """

    num_components_per_switch: int = 25
    num_switches: int = 100
    num_object_types: int = 32
    num_features: int = 5
    num_continuous_dims: int = 7
    interact_with_static: bool = False

    r_ell_threshold: float = -100
    i_ell_threshold: float = -500

    cont_scale_identity: float = 0.5
    cont_scale_switch: float = 25.0

    discrete_alphas: tuple[float] = field(
        default_factory=lambda: tuple([1e-4, 1e-4, 1e-4, 1e-4, 1.0, 1e-4])
    )

    r_interacting: float = 0.6
    r_interacting_predict: float = 0.6

    forward_predict: bool = False
    stable_r: bool = False
    relative_distance: bool = True

    absolute_distance_scale: bool = False

    reward_prob_threshold: float = 0.45

    color_precision_scale: float = 1.0
    color_only_identity: bool = False

    exclude_background: bool = True

    use_ellipses_for_interaction: bool = True

    # multiply velocity features with this scale factor for improved resolution
    velocity_scale: float = 10.0


class RMM(NamedTuple):
    model: HybridMixture
    used_mask: Array
    dirty_mask: Array
    max_switches: int = eqx.static_field()


def predict(
    rmm: RMM,
    c_sample: Array,
    d_sample: List[Array],
    key: Array = None,
    reward_prob_threshold: float = 0.45,
):
    # Overwrite the TMM switch observation with a uniform prior. We don't know the
    # switch value yet. Same for the reward.
    d_sample = d_sample[:-2] + [
        jnp.ones_like(d_sample[-i]) * 1 / (d_sample[-i].flatten().shape[0])
        for i in [2, 1]
    ]

    # also mask out the reward and tmm switch in ell calc
    w_disc = jnp.array([1.0] * len(rmm.model.discrete_likelihoods))
    w_disc = w_disc.at[-2:].set(0.0)

    # Do an e-step to infer the mixture cluster
    qz, c_ell, d_ell = rmm.model._e_step(c_sample, d_sample, w_disc)

    elogp = c_ell + d_ell
    elogp = elogp * rmm.used_mask[None] + (1 - rmm.used_mask[None]) * (-1e10)
    qz = softmax(elogp, rmm.model.mix_dims)

    # Get the distribution over the TMM switch of the inferred cluster
    p_tmm = rmm.model.discrete_likelihoods[-1].mean()[..., 0]
    if key is not None:
        key, *subkeys = jr.split(key, 3)
        mix_slot = jr.choice(subkeys[0], qz[0].shape[0], p=qz[0])
        p_tmm = (p_tmm[mix_slot] > 1e-4) * p_tmm[mix_slot]  # mask out noise
        tmm_slot = jr.choice(subkeys[1], p_tmm.shape[0], p=p_tmm)

    else:
        mix_slot = qz[0].argmax(-1)
        tmm_slot = p_tmm[mix_slot].argmax(-1)

    # Be optimistic about positive reward
    # and pessimistic about negative reward
    # i.e. if max likelihood positive reward probability > prob threshold
    # return 1.0
    # else return the weighted sum of the reward probabilities
    p_reward = rmm.model.discrete_likelihoods[-2].mean()[..., 0]
    p_reward = p_reward * (p_reward > reward_prob_threshold)
    max_likelihood_reward = p_reward[mix_slot]

    reward = jax.lax.cond(
        max_likelihood_reward[-1] > reward_prob_threshold,
        lambda: 1.0,
        lambda: jnp.dot(jnp.array([-1.0, 0.0, 1.0]), jnp.dot(qz[0], p_reward)),
    )

    return tmm_slot[None], reward[None], elogp, qz[0], mix_slot


def forward_default(x):
    x_tmm = x[:, :6]
    transitions = tmm_tools.generate_default_dynamics_component(3)[None]
    x_tmm = jax.vmap(lambda x_in: tmm_tools.forward(transitions, x_in))(x_tmm)[:, 0]
    return x.at[:, :6].set(x_tmm)


@jax.jit
def _in_ellipse(x, y, cx, cy, rx, ry):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1


@partial(jax.jit, static_argnames=["n", "r", "return_grid"])
def _object_interactions(x_t, i, j, n=3, r=1, return_grid=False):
    cx_i, cy_i, w_i, h_i = x_t[i, [0, 1, 6, 7]]
    cx_j, cy_j, w_j, h_j = x_t[j, [0, 1, 6, 7]]

    w_i, h_i, w_j, h_j = r * w_i, r * h_i, r * w_j, r * h_j

    # we sample a grid of nxn points in the bounding box of the ellipse
    px, py = jnp.meshgrid(
        jnp.linspace(cx_j - w_j, cx_j + w_j, n), jnp.linspace(cy_j - h_j, cy_j + h_j, n)
    )
    px, py = px.flatten(), py.flatten()

    # First determine which of the sampled points are in the object_1 ellipse
    out_1 = jax.vmap(lambda xii, yii: _in_ellipse(xii, yii, cx_j, cy_j, w_j, h_j))(
        px, py
    )
    # First determine which of the sampled points are in the object_2 ellipse
    out_2 = jax.vmap(lambda xii, yii: _in_ellipse(xii, yii, cx_i, cy_i, w_i, h_i))(
        px, py
    )

    out = out_1 & out_2

    is_interacting = (out.sum() > 0) * (i != j)

    # Now distance of the centroids.
    distance = jnp.array([cx_i - cx_j, cy_i - cy_j])

    # We could also correct the distance based on the shape of the object.
    # E.g. we offset the means with sigma in the direction
    # sig = jnp.array([w_i / 3 + w_j / 3, h_i / 3 + h_j / 3])
    # abs_dist = jnp.abs(distance) - sig
    # abs_dist = abs_dist * (abs_dist > 0)
    # distance = jnp.sign(distance) * abs_dist

    if return_grid:
        return (is_interacting, distance), (out, px, py)


def get_interacting_objects_ellipse(
    data,
    tracked_obj_mask,
    object_idx,
    r_interacting,
    forward_predict,
    exclude_background=True,
    interact_with_static=True,
):
    (interacting, distances), (grid, *_) = jax.vmap(
        lambda j: _object_interactions(
            data, object_idx, j, n=30, r=r_interacting, return_grid=True
        )
    )(jnp.arange(data.shape[0]))

    # predict if an interaction will occur when the objects keep moving in their current
    # direction.
    data_fwd = forward_default(data)
    (interacting_fwd, _), (grid_fwd, *_) = jax.vmap(
        lambda j: _object_interactions(
            data_fwd, object_idx, j, n=30, r=r_interacting, return_grid=True
        )
    )(jnp.arange(data.shape[0]))

    interacting = jax.lax.cond(
        forward_predict, lambda: interacting | interacting_fwd, lambda: interacting
    )

    # Sort based on overlap
    sort_metric = jax.lax.cond(
        forward_predict, lambda: grid + grid_fwd, lambda: grid
    ).sum(-1)
    # sort based on distance
    sort_metric = (distances**2).sum(-1)

    # mask out the the non present objects
    interacting = interacting * (data[:, 2] == 0)
    # mask out background # TODO: check if background is always 0
    if exclude_background:
        interacting = interacting.at[0].set(False)

    # mask out static objects if we do not want to consider them
    interacting_dynamic = interacting * tracked_obj_mask
    interacting_static = interacting * (1 - tracked_obj_mask)

    # pick the most interesting interacting object (i.e. preference on dynamic objects)
    # if there is a dynamic, pick it
    # otherwise if interact_with_static pick that one,
    # otherwise return -1
    other_idx = jax.lax.cond(
        interacting_dynamic.sum() > 0,
        lambda: (
            sort_metric * interacting_dynamic + (~interacting_dynamic) * 100
        ).argmin(),
        lambda: jax.lax.cond(
            (interacting_static.sum() > 0) & interact_with_static,
            lambda: interacting_static.argmax(),
            lambda: -1,
        ),
    )

    return (other_idx, distances)


@jax.jit
def interacting_squares(x1, y1, w1, h1, x2, y2, w2, h2):
    x_overlap = jnp.abs(x1 - x2) < w1 + w2
    y_overlap = jnp.abs(y1 - y2) < h1 + h2
    return jnp.logical_and(x_overlap, y_overlap)


def get_distance(x_t, i, j, o=0.1):
    cx_i, cy_i, w_i, h_i = x_t[i, [0, 1, 6, 7]]
    cx_j, cy_j, w_j, h_j = x_t[j, [0, 1, 6, 7]]
    d = jnp.array([cx_i - cx_j, cy_i - cy_j])
    interacting = interacting_squares(
        cx_i, cy_i, w_i + o, h_i + o, cx_j, cy_j, w_j, h_j
    )
    return interacting, d, d / jnp.array([w_i, h_i])


def get_interacting_objects_closest(
    data,
    tracked_obj_mask,
    object_idx,
    r_interacting=2,
    exclude_background=True,
    interact_with_static=True,
    absolute_distance_scale=False,
):
    # shape (n_objects, 2)
    interacting, a_distances, r_distances = jax.vmap(
        lambda j: get_distance(data, object_idx, j, o=r_interacting)
    )(jnp.arange(data.shape[0]))

    # Filter out self-distance & sort on absolute distance
    distances = a_distances.at[object_idx].set([100, 100])
    distances = distances * (interacting[:, None]) + (1 - interacting[:, None]) * 100
    dd = distances * (tracked_obj_mask[:, None]) + (1 - tracked_obj_mask[:, None]) * 100
    ds = distances * (1 - tracked_obj_mask[:, None]) + (tracked_obj_mask[:, None]) * 100
    if exclude_background:
        ds = ds.at[0].set([100, 100])

    threshold = 20_000
    other_idx = jax.lax.cond(
        (dd**2).sum(-1).min() < threshold,
        lambda: (dd**2).sum(-1).argmin(),
        lambda: jax.lax.cond(
            ((ds**2).sum(-1).min() < threshold) & interact_with_static,
            lambda: (ds**2).sum(-1).argmin(),
            lambda: -1,
        ),
    )

    if absolute_distance_scale:
        distances = a_distances
    else:
        distances = r_distances

    return other_idx, distances


def _to_distance_obs_hybrid(
    imm,
    data,
    object_idx,
    action,
    reward,
    tmm_switch,
    tracked_obj_mask,
    interact_with_static,
    max_switches,
    action_dim=6,
    object_identities=None,  # shape: (n_objects)
    num_object_classes=32,
    reward_dim=3,
    forward_predict=False,
    r_interacting=0.6,
    stable_r=False,
    relative_distance=False,
    color_only_identity=False,
    exclude_background=True,
    use_ellipses_for_interaction=False,
    velocity_scale=10,
    absolute_distance_scale=False,
    **kwargs,
):
    if tracked_obj_mask is None:
        tracked_obj_mask = jnp.array([True] * data.shape[0])

    if use_ellipses_for_interaction:
        other_idx, distances = get_interacting_objects_ellipse(
            data,
            tracked_obj_mask,
            object_idx,
            r_interacting,
            forward_predict,
            exclude_background=exclude_background,
            interact_with_static=interact_with_static,
        )
    else:
        other_idx, distances = get_interacting_objects_closest(
            data,
            tracked_obj_mask,
            object_idx,
            r_interacting,
            exclude_background=exclude_background,
            interact_with_static=interact_with_static,
            absolute_distance_scale=absolute_distance_scale,
        )

    if stable_r:
        # TODO: stable_r is a fixes that make the model more resilient to not so welldefined
        # values of r. i.e. if there is a default TMM switch predicted, then interactions
        # are not added!
        other_idx = jax.lax.cond(tmm_switch == 0, lambda: -1, lambda: other_idx)

    if object_identities is None:
        class_labels = imm_tools.infer_identity(
            imm,
            jnp.array(data[[object_idx, other_idx], -5:, None]),
            color_only_identity,
        )
        self_id = nn.one_hot(class_labels[0], num_classes=num_object_classes + 1)
        other_id = jax.lax.cond(
            (other_idx == -1) | (class_labels[0] == class_labels[1]),
            lambda: nn.one_hot(
                jnp.array(num_object_classes), num_classes=num_object_classes + 1
            ),
            lambda: nn.one_hot(class_labels[1], num_classes=num_object_classes + 1),
        )
    else:
        # Basically, if we already pass in the identities, this operation does not need
        # to be called over and over again.
        self_id = nn.one_hot(
            object_identities[object_idx], num_classes=num_object_classes + 1
        )
        other_id = nn.one_hot(
            jax.lax.cond(
                other_idx == -1,
                lambda: num_object_classes,
                lambda: object_identities[other_idx],
            ),
            num_classes=num_object_classes + 1,
        )

    # Continuous features: (x, y, u, vx, vy)
    c_feat = data[object_idx, :5]
    if relative_distance:
        d = distances[other_idx]
        d = jax.lax.cond(
            other_idx != -1,
            lambda: distances[other_idx],
            lambda: jnp.array([1.2, 1.2]) + np.random.rand(2) * 0.01,
        )
        c_feat = jnp.concatenate([c_feat, d], axis=0)

    c_feat = c_feat.at[3:5].set(c_feat[3:5] * velocity_scale)

    # Discrete features: (self_id)
    d_feat = [
        self_id,
        other_id,
        nn.one_hot((data[object_idx, 2] == 0).astype(jnp.int32), num_classes=2),
        nn.one_hot(action, num_classes=action_dim),
        # [-1, 0, 1] -> [0, 1, 2]
        nn.one_hot((reward + 1).astype(jnp.int32), num_classes=reward_dim),
        nn.one_hot(tmm_switch, num_classes=max_switches),
    ]

    return c_feat, d_feat


def create_rmm(
    key: Array,
    action_dim: int,
    num_components_per_switch: int,
    num_switches: int,
    num_object_types: int,
    num_continuous_dims: int = 5,
    reward_dim: int = 3,
    cont_scale_switch: float = 25.0,
    discrete_alphas=None,
    **kwargs,
):
    key, subkey = jr.split(key)
    switch_model = create_mm(
        subkey,
        num_components=num_components_per_switch * num_switches,
        continuous_dim=num_continuous_dims,
        discrete_dims=[
            num_object_types + 1,  # own identity
            num_object_types + 1,  # interacting identity
            2,  # used/unused
            action_dim,  # action
            reward_dim,  # reward
            num_switches,  # tmm switches
        ],
        discrete_alphas=discrete_alphas,
        cont_scale=cont_scale_switch,
        opt={"lr": 1.0, "beta": 0.0},
    )

    r_used_mask = jnp.zeros(switch_model.continuous_likelihood.mean.shape[0])

    rmm = RMM(
        model=switch_model,
        used_mask=r_used_mask,
        max_switches=num_switches,
        dirty_mask=jnp.zeros_like(r_used_mask),
    )
    return rmm


def mark_dirty(rmm, elogp, dx, r_ell_threshold):
    qz = softmax(elogp, rmm.model.mix_dims)

    z = qz[0, :].argmax()
    tmm_predict = (
        rmm.model.discrete_likelihoods[-1].alpha[z, :, 0].argmax(-1)
        == dx[-1][0, :, 0].argmax()
    )
    reward_predict = (
        rmm.model.discrete_likelihoods[-2].alpha[z, :, 0].argmax(-1)
        == dx[-2][0, :, 0].argmax()
    )

    # If well explained, and the predictions are wrong
    mark = (elogp.max() < r_ell_threshold) & (~tmm_predict | ~reward_predict)
    dirty_mask = jax.lax.cond(
        mark,
        lambda: rmm.dirty_mask.at[z].set(rmm.dirty_mask[z] + 1.0),
        lambda: rmm.dirty_mask,
    )
    rmm = eqx.tree_at(lambda x: x.dirty_mask, rmm, dirty_mask)
    return rmm


def infer_and_update(
    key: jnp.ndarray,
    rmm: RMM,
    imm: imm_tools.IMM,
    obs: jnp.ndarray,
    tmm_switch: int,
    object_idx: int,
    action: jnp.ndarray = None,
    reward: jnp.ndarray = None,
    r_ell_threshold: float = 1,
    i_ell_threshold: float = 1,
    tracked_obj_ids: Optional[jnp.ndarray] = None,
    interact_with_static: bool = True,
    num_switches: int = 100,
    num_features: int = 5,
    r_interacting: float = 0.6,
    forward_predict: bool = False,
    stable_r=False,
    relative_distance: bool = False,
    color_only_identity=False,
    exclude_background=True,
    use_ellipses_for_interaction=True,
    absolute_distance_scale=False,
    **kwargs,
) -> Tuple[RMM, jnp.ndarray]:
    # 2. Update  the switch model
    cx, dx = _to_distance_obs_hybrid(
        imm,
        obs,
        object_idx,
        action,
        reward,
        tmm_switch,
        tracked_obj_ids,
        interact_with_static,
        num_switches,
        action_dim=rmm.model.discrete_likelihoods[-3].alpha.shape[-2],
        num_object_classes=rmm.model.discrete_likelihoods[0].alpha.shape[1] - 1,
        r_interacting=r_interacting,
        forward_predict=forward_predict,
        stable_r=stable_r,
        relative_distance=relative_distance,
        color_only_identity=color_only_identity,
        exclude_background=exclude_background,
        use_ellipses_for_interaction=use_ellipses_for_interaction,
        absolute_distance_scale=absolute_distance_scale,
    )

    cx = cx[None, :, None]
    dx = jtu.tree_map(lambda x: x[None, :, None], dx)

    r_model, r_used_mask, elogp = train_step_fn(
        rmm.model, rmm.used_mask, cx, dx, logp_thr=r_ell_threshold
    )
    rmm = eqx.tree_at(lambda x: x.model, rmm, r_model)
    rmm = eqx.tree_at(lambda x: x.used_mask, rmm, r_used_mask)

    ### Debug
    # rmm = mark_dirty(rmm, elogp, dx, r_ell_threshold)
    ###

    return rmm, elogp


def rollout_sample(
    key: Array,
    rmm: RMM,
    imm: imm_tools.IMM,
    tmm: tmm_tools.TMM,
    x_input: Array,
    actions: Array,
    tracked_obj_ids: Array,
    num_features: int = 0,
    num_samples: int = 1,
    position_noise=0.0,
    velocity_noise=0.0,
    interact_with_static=False,
    r_interacting_predict=0.6,
    num_switches: int = 100,
    r_ell_threshold: float = 1,
    forward_predict: bool = False,
    reward_prob_threshold: float = 0.45,
    relative_distance: bool = False,
    color_only_identity=False,
    **kwargs,
):
    # Compute the object identities once.
    object_identities = imm_tools.infer_identity(
        imm, x_input[..., None], color_only_identity
    )

    x_input = x_input[None, ...]
    actions = actions[None, ...]

    # repeat num_samples
    x_input = jnp.repeat(x_input, num_samples, axis=0)
    actions = jnp.repeat(actions, num_samples, axis=0)

    # vmap over rollout
    def rollout_fn(x_input, actions, key):
        return rollout(
            rmm,
            tmm,
            x_input,
            actions,
            tracked_obj_ids,
            num_features,
            key,
            position_noise,
            velocity_noise,
            interact_with_static=interact_with_static,
            r_interacting_predict=r_interacting_predict,
            num_switches=num_switches,
            object_identities=object_identities,
            r_ell_threshold=r_ell_threshold,
            forward_predict=forward_predict,
            reward_prob_threshold=reward_prob_threshold,
            relative_distance=relative_distance,
            color_only_identity=color_only_identity,
        )

    key, subkey = jr.split(key, 2)
    sample_keys = jr.split(subkey, num_samples)
    return jax.vmap(rollout_fn)(x_input, actions, sample_keys)


def rollout(
    rmm: RMM,
    imm: imm_tools.IMM,
    tmm: tmm_tools.TMM,
    x_input: Array,
    actions: Array,
    tracked_obj_ids: Array,
    num_features: int = 0,
    key: Array = None,
    position_noise: float = 0.0,
    velocity_noise: float = 0.0,
    interact_with_static: bool = False,
    r_ell_threshold: float = 1,
    num_switches: int = 100,
    object_identities=None,
    r_interacting_predict: float = 0.6,
    forward_predict: bool = False,
    reward_prob_threshold: float = 0.45,
    stable_r=False,
    relative_distance: bool = False,
    color_only_identity=False,
    exclude_background=True,
    use_ellipses_for_interaction=True,
    absolute_distance_scale=False,
    **kwargs,
) -> Array:
    if key is not None:
        keys = jr.split(key, actions.shape[0] * 2).reshape(actions.shape[0], 2, 2)
    else:
        keys = [None] * actions.shape[0]

    def step_fn(x_t, input):
        action_t, subkeys = input

        def _no_op(object_idx):
            return x_t[object_idx, : x_t.shape[1] - num_features], 0, 0.0, 0, 0.0

        def _predict(object_idx):
            c_obs, d_obs = _to_distance_obs_hybrid(
                imm,
                x_t,
                object_idx,
                action_t,
                tmm_switch=10,  # we pass in a dummy value (gets overwritten in predict)
                reward=jnp.array(
                    0
                ),  # we pass in a dummy value (gets overwritten in predict)
                tracked_obj_mask=tracked_obj_ids,
                interact_with_static=interact_with_static,
                max_switches=num_switches,
                action_dim=rmm.model.discrete_likelihoods[-3].alpha.shape[-2],
                object_identities=object_identities,
                num_object_classes=rmm.model.discrete_likelihoods[0].alpha.shape[1] - 1,
                r_interacting=r_interacting_predict,
                forward_predict=forward_predict,
                stable_r=stable_r,
                relative_distance=relative_distance,
                color_only_identity=color_only_identity,
                exclude_background=exclude_background,
                use_ellipses_for_interaction=use_ellipses_for_interaction,
                absolute_distance_scale=absolute_distance_scale,
            )
            c_obs = c_obs[None, :, None]
            d_obs = jtu.tree_map(lambda d: d[None, :, None], d_obs)

            # Compute the TMM switching slot using the rMM
            switch_slot, pred_reward, ell, qz, r_cluster = predict(
                rmm,
                c_obs,
                d_obs,
                key=None if subkeys is None else subkeys[0],
                reward_prob_threshold=reward_prob_threshold,
            )

            # Use the TMM to predict the next location
            x_t_model = x_t[object_idx, : x_t.shape[1] - num_features]
            tmm_prediction = tmm_tools.forward(tmm.transitions, x_t_model)[
                switch_slot[0]
            ]

            # optionally add some noise to tmm prediction
            if subkeys is not None:
                mask = jnp.array(
                    [
                        position_noise,
                        position_noise,
                        0,
                        velocity_noise,
                        velocity_noise,
                        0,
                    ]
                )
                tmm_prediction += mask * jr.normal(
                    subkeys[1], shape=tmm_prediction.shape
                )

            ell = rmm.used_mask * ell[0] + (1 - rmm.used_mask) * (-1e20)

            # do we want info gain bc of expecting to create new component?
            ig = 1 / rmm.model.prior.alpha - 1 / rmm.model.prior.alpha.sum()
            info_gain = jnp.dot(ig, qz)

            # also add info gain if we expect a new cluster to be discovered
            # info gain as if we have a first dirichlet count
            info_gain_unused = 1.0 - 1 / rmm.model.prior.alpha.sum()
            info_gain = jax.lax.cond(
                (ell.max(-1) < r_ell_threshold),
                lambda: info_gain_unused,
                lambda: info_gain,
            )

            # but penalize info gain if states are beyond the frame limits
            info_gain = jax.lax.cond(
                (
                    (tmm_prediction[0] < -1.1)
                    | (tmm_prediction[0] > 1.1)
                    | (tmm_prediction[1] > 1.1)
                    | (tmm_prediction[1] < -1.1)
                ),
                lambda: -1.0,
                lambda: info_gain,
            )

            return tmm_prediction, switch_slot[0], pred_reward[0], r_cluster, info_gain

        def _inner_step_fn(_, object_idx):
            x, tmm_slot, pred_reward, rmm_slot, pred_infogain = jax.lax.cond(
                tracked_obj_ids[object_idx], _predict, _no_op, object_idx
            )

            return None, (x, tmm_slot, rmm_slot, pred_reward, pred_infogain)

        # NOTE: scan is much faster than vmap here
        _, (pred_x_tmm, tmm_slots, rmm_slots, pred_rewards, pred_infogains) = (
            jax.lax.scan(_inner_step_fn, None, jnp.arange(tracked_obj_ids.shape[0]))
        )

        pred_x = jnp.zeros_like(x_t)
        pred_x = pred_x.at[:, : pred_x_tmm.shape[1]].set(pred_x_tmm)
        pred_x = pred_x.at[:, pred_x_tmm.shape[1] :].set(
            x_t[:, x_t.shape[1] - num_features :]
        )

        # Use reward from most salient object (i.e. that predicts most extreme reward value)
        pred_agg_reward = pred_rewards[jnp.abs(pred_rewards).argmax(axis=0)]
        pred_agg_infogain = pred_infogains.mean()

        return pred_x, (
            pred_x,
            tmm_slots,
            rmm_slots,
            pred_agg_reward,
            pred_agg_infogain,
        )

    _, (predictions, tmm_slots, rmm_slots, predicted_rewards, info_gains) = (
        jax.lax.scan(step_fn, x_input, (actions, keys))
    )

    return (
        predictions,
        tmm_slots,
        rmm_slots,
        info_gains[:, None],
        predicted_rewards[:, None],
    )


def _find_pairs(key, rmm, n_samples=None):
    def fix_vals(c_ell, d_ell, idx):
        return c_ell.at[idx].set(-1e20), d_ell.at[idx].set(-1e20)

    cxm, dxm = rmm.model.get_means_as_data()

    indices = jnp.arange(rmm.used_mask.shape[0])[rmm.used_mask > 0]

    if n_samples is not None:
        indices = indices[
            jr.choice(
                key,
                jnp.arange(len(indices)),
                shape=(min(n_samples * 2, len(indices) - 1),),
                replace=False,
            )
        ]

    _, cont_ell, disc_ell = rmm.model._e_step(
        cxm[indices],
        jtu.tree_map(lambda d: d[indices], dxm),
    )

    c_ell_fixed, d_ell_fixed = jax.vmap(fix_vals)(cont_ell, disc_ell, indices)
    qz_argmax = softmax(c_ell_fixed + d_ell_fixed, axis=rmm.model.mix_dims).argmax(-1)

    pairs = jnp.stack([indices, qz_argmax], axis=-1)

    tmm_switches = dxm[-1][..., 0].argmax(-1)
    rewards = dxm[-2][..., 0].argmax(-1)
    ids = dxm[0][..., 0].argmax(-1)
    other_ids = dxm[1][..., 0].argmax(-1)

    def filter_fn(mask, i):
        is_good = rewards[pairs[i][0]] == rewards[pairs[i][1]]
        # Same TMM switch
        is_good = is_good & (tmm_switches[pairs[i][0]] == tmm_switches[pairs[i][1]])
        # Same object
        is_good = is_good & (ids[pairs[i][0]] == ids[pairs[i][1]])
        # Same interaction
        is_good = is_good & (other_ids[pairs[i][0]] == other_ids[pairs[i][1]])
        mask = mask.at[i].set(is_good)
        return mask, None

    mask = jnp.zeros(pairs.shape[0]).astype(bool)
    mask, _ = jax.lax.scan(filter_fn, mask, jnp.arange(pairs.shape[0]))

    return pairs[mask][:n_samples]


@jax.jit
def consider_merge(c_data, d_data, elbo_before, mixture, used_mask, idx_1, idx_2):
    mixture_after = jtu.tree_map(lambda x: x, mixture)
    mixture_after._merge_clusters(idx_1, idx_2)
    mask_after = used_mask.at[idx_2].set(0)

    elbo_after = compute_elbo(mixture_after, c_data, d_data)

    def replace_fn():
        return mixture_after, mask_after, elbo_after, True

    def keep_fn():
        return mixture, used_mask, elbo_before, False

    return jax.lax.cond(
        (used_mask[idx_1] > 0) & (used_mask[idx_2] > 0),
        lambda: jax.lax.cond(elbo_after >= elbo_before, replace_fn, keep_fn),
        keep_fn,
    )


@jax.jit
def compute_elbo(mm, c_data, d_data):
    """
    NOTE: not the full elbo. This only accounts for the likelihoods on the TMM switch
    and the reward, and only considers the prior complexities. As these are the only
    ones we want to optimize for in BMR
    """
    mask = mm.prior.alpha > mm.prior.prior_alpha

    # Only account for the TMM and reward likelihoods!

    # First calculate predicted qz's
    qz_predict, *_ = mm._e_step(
        c_data, d_data, w_disc=jnp.ones(len(d_data)).at[-2:].set(0)
    )

    # Then calculate ELL for the TMM and reward likelihoods
    *_, d_ell = mm._e_step(c_data, d_data, w_disc=jnp.zeros(len(d_data)).at[-2:].set(1))

    # Remove unused components
    d_ell = d_ell * mask[None]

    # And get ELL for inferred qz as elbo contrib
    d_ell = jax.vmap(lambda i, j: d_ell[i, j])(
        jnp.arange(d_ell.shape[0]), qz_predict.argmax(-1)
    )

    _elbo_contrib = d_ell.mean()
    _elbo = _elbo_contrib - mm.prior.kl_divergence()

    return _elbo


def run_bmr(key, rmm, n_samples, pairs=None, cxm=None, dxm=None):
    if cxm is None:
        cxm, dxm = rmm.model.get_means_as_data()
        cxm = cxm[rmm.used_mask > 0]
        dxm = jtu.tree_map(lambda d: d[rmm.used_mask > 0], dxm)

    initial_elbo = compute_elbo(rmm.model, cxm, dxm)

    def step_fn(carry, pair):
        mixture, used_mask, elbo = carry
        mixture, used_mask, elbo, merged = consider_merge(
            cxm, dxm, elbo, mixture, used_mask, *pair
        )
        return (mixture, used_mask, elbo), (elbo, used_mask.sum(), merged)

    if pairs is None:
        pairs = _find_pairs(key, rmm, n_samples)

    (mixture, used_mask, _), (elbo_hist, num_components_hist, merged) = jax.lax.scan(
        step_fn,
        (rmm.model, rmm.used_mask, initial_elbo),
        pairs,
    )

    new_rmm = eqx.tree_at(lambda x: x.model, rmm, mixture)
    new_rmm = eqx.tree_at(lambda x: x.used_mask, new_rmm, used_mask)

    return new_rmm, elbo_hist, num_components_hist, pairs[merged]
