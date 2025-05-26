import sys
import rich

import mediapy
import csv

from tqdm import tqdm

import wandb
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

import defaults

from gameworld.envs import create_gameworld_env
from gameworld.perturbed_envs import (
    create_gameworld_env as create_gameworld_env_perturbed,
)

from axiom import infer as ax
from axiom import visualize as vis


def main(config):
    key = jr.PRNGKey(config.seed)
    np.random.seed(config.seed)

    if config.precision_type == "float64":
        jax.config.update("jax_enable_x64", True)

    # Create environment.
    if config.perturb is None:
        env = create_gameworld_env(config.game)
    else:
        env = create_gameworld_env_perturbed(
            config.game, perturb=config.perturb, perturb_step=config.perturb_step
        )

    observations = []
    rewards = []
    expected_utility = []
    expected_info_gain = []
    num_components = []

    # reset
    obs, _ = env.reset()
    obs = obs.astype(np.uint8)
    observations.append(obs)
    reward = 0

    # initialize
    key, subkey = jr.split(key)
    carry = ax.init(subkey, config, obs, env.action_space.n)

    bmr_buffer = None, None

    # main loop
    for t in tqdm(range(config.num_steps)):
        # action selection
        key, subkey = jr.split(key)
        action, carry, plan_info = ax.plan_fn(subkey, carry, config, env.action_space.n)

        best = jnp.argsort(plan_info["rewards"][:, :, 0].sum(0))[-1]
        expected_utility.append(
            plan_info["expected_utility"][:, best, :].mean(-1).sum(0)
        )
        expected_info_gain.append(
            plan_info["expected_info_gain"][:, best, :].mean(-1).sum(0)
        )
        num_components.append(carry["rmm_model"].used_mask.sum())

        # step env
        obs, reward, done, truncated, info = env.step(action)
        obs = obs.astype(np.uint8)
        observations.append(obs)
        rewards.append(reward)

        # wandb.log({"reward": reward})

        # update models
        update = True
        remap_color = False
        if (
            config.remap_color
            and config.perturb is not None
            and t + 1 >= config.perturb_step
            and t < config.perturb_step + 20
        ):
            update = False
            remap_color = True

        carry, rec = ax.step_fn(
            carry,
            config,
            obs,
            jnp.array(reward),
            action,
            num_tracked=0,
            update=update,
            remap_color=remap_color,
        )

        # log stuff
        observations.append(obs)

        if done:
            obs, _ = env.reset()
            obs = obs.astype(np.uint8)
            observations.append(obs)
            reward = 0

            carry, rec = ax.step_fn(
                carry,
                config,
                obs,
                jnp.array(reward),
                jnp.array(0),
                num_tracked=0,
                update=False,
            )

        if (t + 1) % config.prune_every == 0:
            key, subkey = jr.split(key)
            new_rmm, pairs, *bmr_buffer = ax.reduce_fn_rmm(
                subkey,
                carry["rmm_model"],
                *bmr_buffer,
                n_samples=config.bmr_samples,
                n_pairs=config.bmr_pairs,
            )
            carry["rmm_model"] = new_rmm

    # Write results to file: a csv file iwth the rewards adn a video of the gameplay
    with open(f"{config.game.lower()}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Step",
                "Reward",
                "Average Reward",
                "Cumulative Reward",
                "Expected Utility",
                "Expected Info Gain",
                "Num Components",
            ]
        )
        for i in range(len(rewards)):
            writer.writerow(
                [
                    i,
                    rewards[i],
                    jnp.mean(jnp.array(rewards[max(0, i - 1000) : max(i, 1)])),
                    sum(jnp.array(rewards[max(0, i - 1000) : i])),
                    expected_utility[i],
                    expected_info_gain[i],
                    num_components[i],
                ]
            )

    with mediapy.set_show_save_dir("."):
        mediapy.show_videos({f"{config.game.lower()}": observations}, fps=30)

    # Do wandb logging after the job to avoid performance impact
    wandb.init(
        reinit=True,
        group=config.group,
        project=config.project,
        config=config,
        resume="allow",
        id=config.id + "-" + config.game,
        name=config.name + "-" + config.game,
    )

    for i in range(len(rewards)):
        wandb.log(
            {
                "reward": rewards[i],
                "reward_1k_avg": jnp.mean(
                    jnp.array(rewards[max(0, i - 1000) : max(i, 1)])
                ),
                "cumulative_reward": sum(jnp.array(rewards[max(0, i - 1000) : i])),
                "expected_utility": expected_utility[i],
                "expected_info_gain": expected_info_gain[i],
                "num_components": num_components[i],
            }
        )

    # finally log a sample of final gameplay
    logs = {
        "play": wandb.Video(
            np.asarray(observations)[-1000:].transpose(0, 3, 1, 2),
            fps=30,
            format="mp4",
        ),
        "rmm": wandb.Image(
            vis.plot_rmm(carry["rmm_model"], carry["imm_model"], colorize="cluster")
        ),
        "plan": wandb.Image(
            vis.plot_plan(
                observations[-2],
                plan_info,
                carry["tracked_obj_ids"][config.layer_for_dynamics],
                carry["smm_model"].stats,
                topk=1,
            )
        ),
        "identities": wandb.Image(vis.plot_identity_model(carry["imm_model"])),
    }
    if config.perturb is not None:
        logs["perturb"] = wandb.Video(
            np.asarray(observations)[
                config.perturb_step - 100 : config.perturb_step + 100
            ].transpose(0, 3, 1, 2),
            fps=30,
            format="mp4",
        )
    wandb.log(logs)


if __name__ == "__main__":
    config = defaults.parse_args(sys.argv[1:])
    rich.print(config)
    main(config)
