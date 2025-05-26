import equinox as eqx

import jax
from jax import jit, lax, vmap
from jax import random as jr, numpy as jnp
from jax import tree_util as jtu
from jax.scipy.linalg import block_diag

from functools import partial

from dataclasses import asdict

from axiom.models import (
    smm as smm_tools,
    hsmm as hsmm_tools,
    tmm as tmm_tools,
    rmm as rmm_tools,
    imm as imm_tools,
)
import axiom.planner as mppi


def init(key, config, observation, action_dim):
    # SMM
    key, subkey = jr.split(key)
    smm_model = hsmm_tools.create_hierarch_smm(subkey, layer_configs=config.smm)

    used, moving = [], []
    for l in range(len(config.smm)):
        # check which slots are actually used and contain moving objects
        used.append(jnp.zeros(config.smm[l].num_slots))
        moving.append(jnp.zeros(config.smm[l].num_slots))

    ## TMM
    key, subkey = jr.split(key)
    tmm_model = tmm_tools.create_tmm(subkey, **asdict(config.tmm))

    ## IMM
    key, subkey = jr.split(key)
    imm_model = imm_tools.create_imm(
        subkey,
        **asdict(config.imm),
    )

    ## RMM
    key, subkey = jr.split(key)
    rmm_model = rmm_tools.create_rmm(
        subkey,
        action_dim=action_dim,
        **asdict(config.rmm),
    )

    # initialize the model using the first observation
    obs = smm_tools.format_single_frame(
        observation,
        offset=smm_model.stats["offset"],
        stdevs=smm_model.stats["stdevs"],
    )
    smm_model, py, qx, qz, u, _ = hsmm_tools.initialize_hierarch_smm(
        subkey, smm_model, init_inputs=obs, layer_configs=config.smm
    )

    num_tracked_steps = []
    objects_mean, objects_sigma, used, x, moving = [], [], [], [], []
    for l in range(len(config.smm)):
        if l == 0:
            objects_mean_l, objects_sigma_l = (
                py[l].mean[0, 0, :, :, 0],  # shape (num_slots_layer_l, 5)
                py[l].sigma[0, 0],  # shape (num_slots_layer_l, 5, 5)
            )

        else:
            # compute assignment mask normalized over the input slots (over the N_tokens dimension)
            safe_denom = qz[l][0].sum(0, keepdims=True) + 1e-8
            qz_normalized_over_inputs = qz[l][0] / safe_denom
            # (num_slots_layer_l-1, 3, 1) , (num_slots_layer_l-1, 1, num_slots_layer_l) --> (3, num_slots_layer_l)
            macro_slot_average_color = (
                py[l - 1].mean[0, 0, :, 2:, :] * qz_normalized_over_inputs[:, None, :]
            ).sum(0)
            # (num_slots_layer_l-1, 3, 3) , (num_slots_layer_l-1, 1, 1, num_slots_layer_l) --> (3,3, num_slots_layer_l)
            macro_slot_average_color_sigma = (
                jnp.expand_dims(py[l - 1].sigma[0, 0, :, 2:, 2:], -1)
                * qz_normalized_over_inputs[:, None, None, :]
            ).sum(0)

            macro_slot_position = py[l].mean[0, 0, :, :2, 0]
            macro_slot_shape = py[l].sigma[0, 0, :, :2, :2]

            objects_mean_l = jnp.concatenate(
                [
                    macro_slot_position,
                    jnp.permute_dims(macro_slot_average_color, axes=(1, 0)),
                ],
                axis=-1,
            )

            objects_sigma_l = vmap(block_diag)(
                macro_slot_shape,
                jnp.permute_dims(macro_slot_average_color_sigma, axes=(2, 0, 1)),
            )
        objects_mean.append(objects_mean_l)
        objects_sigma.append(objects_sigma_l)
        used.append(0.01 * u[l])

        # Initialize x with zeros and fill in positions, first velocity will be 0
        x_l = jnp.zeros(
            (
                objects_mean_l.shape[0],
                2 * (2 + int(config.use_unused_counter)) + config.rmm.num_features,
            )
        )
        x_l = x_l.at[:, :2].set(objects_mean_l[:, :2])
        if config.use_unused_counter:
            # Counter is at 0 if used, positive otherwise
            x_l = x_l.at[:, 2].set((1 - u[l]) * config.tmm.vu)

        # Let's assume this is 5 for now, sigma x, sigma y, r, g, b
        safe_shape_variances = objects_sigma_l[:, jnp.arange(2), jnp.arange(2)] + 1e-8
        shape = jnp.sqrt(safe_shape_variances) * 3
        color = objects_mean_l[:, 2:]
        x_l = x_l.at[:, -config.rmm.num_features : -config.rmm.num_features + 2].set(
            shape
        )
        x_l = x_l.at[:, -config.rmm.num_features + 2 :].set(color)
        x.append(x_l)
        # track moving average of the component's velocity when used
        abs_vels_l = jnp.sqrt(
            x_l[:, 2 + int(config.use_unused_counter)] ** 2
            + x_l[:, 3 + int(config.use_unused_counter)] ** 2
        )
        moving_l = 0.01 * abs_vels_l * u[l]
        moving.append(moving_l)

        num_tracked_steps.append(jnp.zeros(config.smm[l].num_slots, dtype=jnp.int32))

    tracked_obj_ids = [
        jnp.array([False] * config.smm[l].num_slots) for l in range(len(config.smm))
    ]
    # initial carry
    num_plan_steps = 1 if config.planner is None else config.planner.num_steps
    fg_mask = qz[0][0].sum(0)
    fg_mask = fg_mask < fg_mask.max()
    fg_mask = [fg_mask] + [
        jnp.ones(config.smm[l].num_slots, dtype=bool) for l in range(1, len(config.smm))
    ]
    initial_carry = {
        "smm_model": smm_model,
        "imm_model": imm_model,
        "tmm_model": tmm_model,
        "rmm_model": rmm_model,
        "qx": qx,
        "used": used,
        "moving": moving,
        "tracked_obj_ids": tracked_obj_ids,
        "num_tracked_steps": num_tracked_steps,
        "foreground_mask": fg_mask,
        "x": x,
        "object_colors": [objects_mean[l][:, 2:] for l in range(len(config.smm))],
        "key": key,
        "mppi_probs": jnp.full((num_plan_steps, action_dim), 1.0 / action_dim),
        "current_plan": jnp.zeros(num_plan_steps, jnp.int32),
    }

    return initial_carry


@partial(jit, static_argnames=["config", "num_tracked", "update", "remap_color"])
def step_fn(
    carry, config, obs, reward, action, num_tracked=0, update=True, remap_color=False
):
    """Version of step_fn that uses a hierarchical SMM model"""
    # unpacking the carry dictionary
    smm_model = carry["smm_model"]
    imm_model = carry["imm_model"]
    tmm_model = carry["tmm_model"]
    rmm_model = carry["rmm_model"]
    qx = carry["qx"]
    used_prev = carry["used"]
    moving_prev = carry["moving"]
    num_tracked_steps_prev = carry["num_tracked_steps"]
    prev_x = carry["x"]
    key = carry["key"]

    """" Update the SMM model """
    obs = smm_tools.format_single_frame(
        obs,
        offset=smm_model.stats["offset"],
        stdevs=smm_model.stats["stdevs"],
    )

    key, subkey = jr.split(key)
    hsmm_mask = carry["foreground_mask"][0]

    if num_tracked == 0:
        slots_to_pass_up = [jnp.where(hsmm_mask, size=hsmm_mask.shape[0] - 1)[0]]
    else:
        slots_to_pass_up = [jnp.where(carry["tracked_obj_ids"][0], size=num_tracked)[0]]
    (
        smm_model,
        py,
        qx,
        qz_for_vis,
        used_smm,
        smm_eloglike_for_vis,
    ) = hsmm_tools.infer_and_update(
        subkey,
        smm_model,
        obs,
        prev_qx=qx,
        layer_configs=config.smm,
        slots_to_pass_up=slots_to_pass_up,
    )

    (
        objects_mean,
        objects_sigma,
        used,
        unused_mask,
        x,
        moving,
        tracked_obj_ids,
        num_tracked_steps,
    ) = ([], [], [], [], [], [], [], [])
    for l in range(len(config.smm)):
        if l == 0:
            objects_mean_l, objects_sigma_l = (
                py[l].mean[0, 0, :, :, 0],  # shape (num_slots_layer_l, 5)
                py[l].sigma[0, 0],  # shape (num_slots_layer_l, 5, 5)
            )

        else:
            # compute assignment mask normalized over the input slots (over the N_tokens dimension)
            safe_denom = qz_for_vis[l][0].sum(0, keepdims=True) + 1e-8
            # compute assignment mask normalized over the input slots (over the N_tokens dimension)
            qz_normalized_over_inputs = qz_for_vis[l][0] / safe_denom
            # (num_slots_layer_l-1, 3, 1) , (num_slots_layer_l-1, 1, num_slots_layer_l) --> (3, num_slots_layer_l)
            macro_slot_average_color = (
                py[l - 1].mean[0, 0, slots_to_pass_up[l - 1], 2:, :]
                * qz_normalized_over_inputs[:, None, :]
            ).sum(0)
            # (num_slots_layer_l-1, 3, 3) , (num_slots_layer_l-1, 1, 1, num_slots_layer_l) --> (3,3, num_slots_layer_l)
            macro_slot_average_color_sigma = (
                jnp.expand_dims(
                    py[l - 1].sigma[0, 0, slots_to_pass_up[l - 1], 2:, 2:], -1
                )
                * qz_normalized_over_inputs[:, None, None, :]
            ).sum(0)

            macro_slot_position = py[l].mean[0, 0, :, :2, 0]
            macro_slot_shape = py[l].sigma[0, 0, :, :2, :2]

            objects_mean_l = jnp.concatenate(
                [
                    macro_slot_position,
                    jnp.permute_dims(macro_slot_average_color, axes=(1, 0)),
                ],
                axis=-1,
            )

            objects_sigma_l = vmap(block_diag)(
                macro_slot_shape,
                jnp.permute_dims(macro_slot_average_color_sigma, axes=(2, 0, 1)),
            )

        objects_mean.append(objects_mean_l)
        objects_sigma.append(objects_sigma_l)

        used_l = 0.99 * used_prev[l] + 0.01 * used_smm[l]
        used.append(used_l)

        # mask needed to pass on prev_x later
        unused_mask_l = 1 - used_smm[l][:, None]
        unused_mask.append(unused_mask_l)

        prev_x_l = prev_x[l]

        x_l = jnp.zeros_like(prev_x_l)

        # Repeat previous value of positions if unused
        x_l = x_l.at[:, :2].set(
            unused_mask_l * prev_x_l[:, :2]
            + (1 - unused_mask_l) * objects_mean_l[:, :2]
        )

        if config.use_unused_counter:
            # Increment if used & reset to 0 if now used
            # Scale by vu to avoid becoming dominant in the TMM ELL term
            count = unused_mask_l[:, 0] * config.tmm.vu
            x_l = x_l.at[:, 2].set(prev_x_l[:, 2] * unused_mask_l[:, 0] + count)

        # append velocity to get (num_obj, 4) state representations
        velocity_l = (
            x_l[:, : 2 + int(config.use_unused_counter)]
            - prev_x_l[:, : 2 + int(config.use_unused_counter)]
        )

        velocity_l = jnp.where(
            jnp.abs(velocity_l) < config.tmm.clip_value, 0.0, velocity_l
        )

        x_l = x_l.at[
            :,
            2 + int(config.use_unused_counter) : x_l.shape[1] - config.rmm.num_features,
        ].set(velocity_l)

        safe_shape_variances = objects_sigma_l[:, jnp.arange(2), jnp.arange(2)] + 1e-8
        shape = jnp.sqrt(safe_shape_variances) * 3
        color = objects_mean_l[:, 2:]
        x_l = x_l.at[:, -config.rmm.num_features : -config.rmm.num_features + 2].set(
            shape
        )
        x_l = x_l.at[:, -config.rmm.num_features + 2 :].set(color)

        # Repeat previous value of shape and colors if unused
        x_l = x_l.at[:, -config.rmm.num_features :].set(
            unused_mask_l * prev_x_l[:, -config.rmm.num_features :]
            + (1 - unused_mask_l) * x_l[:, -config.rmm.num_features :]
        )

        # track moving average of the component's velocity when used
        abs_vels_l = jnp.sqrt(
            x_l[:, 2 + int(config.use_unused_counter)] ** 2
            + x_l[:, 3 + int(config.use_unused_counter)] ** 2
        )

        # in case of large velocities (i.e. teleportations),
        # set the velocity part to 0
        mask_l = (jnp.abs(abs_vels_l) < config.tmm.position_threshold)[:, None]
        zeros_for_vel = (
            jnp.ones(2 * (2 + int(config.use_unused_counter)) + config.rmm.num_features)
            .at[
                2
                + int(config.use_unused_counter) : 2
                * (2 + int(config.use_unused_counter))
            ]
            .set(0)
        )
        x_l = x_l * mask_l + x_l * (1 - mask_l) * zeros_for_vel
        if config.use_unused_counter:
            # We can't have negative velocities for unused
            x_l = x_l.at[:, 5].set(0)

        x.append(x_l)

        moving_l = (
            0.99 * moving_prev[l] * used_smm[l]
            + (1 - used_smm[l]) * moving_prev[l]
            + 0.01 * abs_vels_l * used_smm[l]
        )

        moving.append(moving_l)

        # Determine if the object meets the instantaneous tracking criteria in this step
        meets_thresholds_l = (
            (moving_l > config.moving_threshold[l])
            & (used_l > config.used_threshold[l])
            & (
                x_l[:, 2] < config.max_steps_tracked_unused * config.tmm.vu
            )  # stop tracking if currently unused for more than max_steps_tracked_unused steps
        )

        # Update the consecutive tracked steps counter
        # Increment the counter if thresholds are met, otherwise reset to 0
        num_tracked_steps_l = (num_tracked_steps_prev[l] + 1) * meets_thresholds_l
        num_tracked_steps.append(num_tracked_steps_l)

        # Track object
        tracked_obj_ids_l = num_tracked_steps_l >= config.min_track_steps[l]
        if l == 0:
            # Don't track the bg slot even if it moves
            tracked_obj_ids_l = tracked_obj_ids_l.at[0].set(False)

        tracked_obj_ids.append(tracked_obj_ids_l)

        # once tracked, always tracked?
        # if a slot becomes unused for a while we might no longer want to track it
        # however, if slots only get assigned a fraction of the time (i.e. bullets or rare objects)
        # they might never get tracked status?
        # | carry["tracked_obj_ids"]

    # compute a foreground mask, using the zero-th layer assignments
    fg_mask = qz_for_vis[0][0].sum(0)
    fg_mask = fg_mask < fg_mask.max()
    fg_mask = [fg_mask] + [
        jnp.ones(config.smm[l].num_slots, dtype=bool) for l in range(1, len(config.smm))
    ]

    # if update=False, don't update tmm/rmm model

    """" Update the TMM model """

    # which layer of SMM latents do you want to use for the dynamics
    dyn_layer_id = config.layer_for_dynamics

    def tmm_over_objects(carry, k):
        tmm_model = carry

        def _no_op(_model):
            return _model, jnp.full(config.tmm.n_total_components, 0.0)

        def _update_with_k(_model):
            return tmm_tools.update_model(
                _model,
                prev_x[dyn_layer_id][
                    k, : prev_x[dyn_layer_id].shape[1] - config.rmm.num_features
                ],
                x[dyn_layer_id][
                    k, : x[dyn_layer_id].shape[1] - config.rmm.num_features
                ],
                **asdict(config.tmm),
            )

        tmm_model_updated, logprobs_k = lax.cond(
            update & tracked_obj_ids[dyn_layer_id][k], _update_with_k, _no_op, tmm_model
        )

        return tmm_model_updated, logprobs_k

    tmm_model, logprobs = lax.scan(
        tmm_over_objects, tmm_model, jnp.arange(config.smm[dyn_layer_id].num_slots)
    )
    switches = jnp.argmax(logprobs, axis=-1)

    """" Update the rMM and iMM model """
    if remap_color:
        # we are triggered that colors might be remapped ... try to reassign object identities based
        # on other features, e.g. shape
        def _infer_remapped_color_id(rmm_model, k):
            identity = imm_tools.infer_remapped_color_identity(
                imm_model, obs=x[dyn_layer_id], object_idx=k, **asdict(config.rmm)
            )
            return imm_model, identity

        imm_model, identities = lax.scan(
            _infer_remapped_color_id,
            imm_model,
            jnp.arange(config.smm[dyn_layer_id].num_slots),
        )

        mask = (x[dyn_layer_id][:, 2] == 0) & fg_mask[dyn_layer_id][:]
        identities = identities * mask[:, None]

        def do_remap(rmm_model):
            # TODO update model, remapping old slots on the new data
            model_updated = jax.tree_map(lambda x: x, imm_model.model)

            # first reset the slots to remap to prior params
            slots_to_wipe = identities.sum(axis=0)
            model_updated.prior.posterior_params = jtu.tree_map(
                lambda post, pri: pri * slots_to_wipe + post * (1 - slots_to_wipe),
                model_updated.prior.posterior_params,
                model_updated.prior.prior_params,
            )

            # and then update i_model
            object_features = x[dyn_layer_id][
                :, None, x[dyn_layer_id].shape[-1] - config.rmm.num_features :, None
            ]
            object_features = object_features.at[:, :, 2:, :].set(
                object_features[:, :, 2:, :] * 100
            )

            def remap_model(model, k):
                def _update_model(model, k):
                    model._m_step_keep_unused(
                        object_features[k], [], identities[k][None, :]
                    )
                    return model, None

                def _do_nothing(model, k):
                    return model, None

                model = lax.cond(mask[k], _update_model, _do_nothing, model, k)
                return model

            # and update rmm_model
            model_updated, _ = lax.scan(
                remap_model,
                model_updated,
                jnp.arange(config.smm[dyn_layer_id].num_slots),
            )
            imm_model = eqx.tree_at(lambda x: x.model, imm_model, model_updated)
            return rmm_model

        def do_nothing(imm_model):
            return imm_model

        # if we can remap all objects, we do it, else we wait
        # TODO we have trouble inferring the correct slot if we already remapped few objects previously
        # in practice this waits until all objects are back in view
        imm_model = jax.lax.cond(
            (imm_model.model.prior.alpha > 0.1).sum()
            == ((identities > 0.1).sum(axis=1) > 0.1).sum(),
            do_remap,
            do_nothing,
            imm_model,
        )
    else:

        def update_i_model(imm_model, k):
            def _update_model(imm):
                imm = imm_tools.infer_and_update_identity(
                    imm,
                    obs=prev_x[dyn_layer_id],
                    object_idx=k,
                    **asdict(config.imm),
                )
                return imm

            is_visible = (prev_x[dyn_layer_id][k][2] == 0) & fg_mask[dyn_layer_id][k]

            if not update:
                is_visible = jnp.zeros_like(is_visible, dtype=bool)

            imm_model_updated = lax.cond(
                is_visible,  # is_visible if config.rmm.interact_with_static else is_tracked,
                _update_model,
                lambda x: x,
                imm_model,
            )

            return imm_model_updated, None

        # Fit the identity model on background components first
        imm_model, _ = lax.scan(
            update_i_model, imm_model, jnp.arange(config.smm[dyn_layer_id].num_slots)
        )

    def rmm_over_objects(carry, k):
        key, rmm_model = carry
        key, subkey = jr.split(key)

        def _no_op(model):
            return model, 0

        def _update_model(model):
            model, logprobs_k_rmm = rmm_tools.infer_and_update(
                subkey,
                model,
                imm_model,
                obs=prev_x[dyn_layer_id],
                tmm_switch=switches[k],
                object_idx=k,
                tracked_obj_ids=tracked_obj_ids[dyn_layer_id],
                action=action,
                reward=reward,
                **asdict(config.rmm),
            )
            return model, logprobs_k_rmm.argmax(-1)[0]

        rmm_model_updated, rmm_slot = lax.cond(
            update & tracked_obj_ids[dyn_layer_id][k], _update_model, _no_op, rmm_model
        )

        return (key, rmm_model_updated), rmm_slot

    key, subkey = jr.split(key)
    (_, rmm_model), rmm_switches = lax.scan(
        rmm_over_objects,
        (subkey, rmm_model),
        jnp.arange(config.smm[dyn_layer_id].num_slots),
    )

    # store some records
    records = {}
    records["decoded_mu"] = objects_mean
    records["decoded_sigma"] = objects_sigma
    records["qz"] = qz_for_vis
    records["smm_eloglike"] = smm_eloglike_for_vis
    records["switches"] = switches
    records["rmm_switches"] = rmm_switches
    records["tracked_obj_ids"] = tracked_obj_ids
    records["x"] = x
    records["moving"] = moving

    next_carry = {
        "smm_model": smm_model,
        "tmm_model": tmm_model,
        "imm_model": imm_model,
        "rmm_model": rmm_model,
        "qx": qx,
        "used": used,
        "moving": moving,
        "tracked_obj_ids": tracked_obj_ids,
        "num_tracked_steps": num_tracked_steps,
        "foreground_mask": fg_mask,
        "x": x,
        "object_colors": [objects_mean[l][:, 2:] for l in range(len(config.smm))],
        "key": key,
        "mppi_probs": carry["mppi_probs"],
        "current_plan": carry["current_plan"],
    }

    return next_carry, records


@partial(jit, static_argnames=["config", "action_dim"])
def plan_fn(key, carry, config, action_dim):
    imm_model = carry["imm_model"]
    tmm_model = carry["tmm_model"]
    rmm_model = carry["rmm_model"]

    probs = carry["mppi_probs"]
    current_plan = carry["current_plan"]

    x = carry["x"][config.layer_for_dynamics]
    tracked_obj_ids = carry["tracked_obj_ids"][config.layer_for_dynamics]

    def rollout_fn(k, x, actions):

        def rollout_action_seq(actions, key):
            object_identities = imm_tools.infer_identity(
                imm_model,
                x[..., None],
                config.imm.color_only_identity,
            )

            return rmm_tools.rollout(
                rmm_model,
                imm_model,
                tmm_model,
                x,
                actions,
                tracked_obj_ids,
                key=key,
                **asdict(config.rmm),
                object_identities=object_identities,
            )

        # planner gives actions as [num_steps, num_samples, action_dim]
        # whereas we vmap over num_samples
        actions = jnp.transpose(actions, (1, 0, 2))

        # pred_xs should be shape (num_policies, num_timesteps, num_slots, data_dim)
        pred_xs, pred_switches, pred_rmm_switches, info_gains, pred_rewards = vmap(
            rollout_action_seq
        )(actions[:, :, 0], jr.split(k, actions.shape[0]))

        # Planner expects predictions transposed appropriately.
        pred_xs = jnp.transpose(pred_xs, (1, 0, 2, 3))
        pred_switches = jnp.transpose(pred_switches, (1, 0, 2))
        pred_rmm_switches = jnp.transpose(pred_rmm_switches, (1, 0, 2))
        pred_rewards = jnp.transpose(pred_rewards, (1, 0, 2))
        info_gains = jnp.transpose(info_gains, (1, 0, 2))

        return pred_xs, pred_switches, pred_rmm_switches, pred_rewards, info_gains

    action, info = mppi.plan(
        x,
        rollout_fn,
        action_dim=action_dim,
        key=key,
        probs=probs,
        current_plan=current_plan,
        **asdict(config.planner),
    )

    carry["mppi_probs"] = info["probs"]
    carry["current_plan"] = info["current_plan"]
    return action, carry, info


def reduce_fn_rmm(key, rmm_model, cxm=None, dxm=None, n_samples=2000, n_pairs=2000):
    key, subkey = jr.split(key)
    cxm_new, dxm_new = rmm_model.model.sample(subkey, n_samples)
    if cxm is None:
        cxm, dxm = cxm_new, dxm_new
    else:
        # Basically also keep optimizing for the old data points that were sampled
        cxm = jnp.concatenate([cxm, cxm_new], axis=0)
        dxm = jtu.tree_map(
            lambda d1, d2: jnp.concatenate([d1, d2], axis=0), dxm, dxm_new
        )

    key, subkey = jr.split(key)
    rmm_model, _, _, merged_pairs = rmm_tools.run_bmr(
        subkey, rmm_model, n_pairs, cxm=cxm, dxm=dxm
    )
    return rmm_model, merged_pairs, cxm, dxm
