import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import jax.lax as lax

from typing import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerConfig:
    num_steps: int = 24
    num_policies: int = 512
    num_samples_per_policy: int = 1
    topk_ratio: float = 0.1  # fraction of the top samples to calc probs
    random_ratio: float = 0.5  # fraction of the samples to draw randomly
    alpha: float = 1.0
    temperature: float = 10.0
    normalize: bool = True
    iters: int = 1
    gamma: float = 0.99
    repeat_prob: float = 0.0
    info_gain: float = 1.0
    lazy_reward: bool = False
    sample_action: bool = False


def _smooth_actions(key, num_steps, num_samples, action_dim, bias_prob=0.6):
    # Initialize the action sequence array
    key, _key = jr.split(key)
    action_sequences = jr.randint(
        _key, shape=(num_steps, num_samples), minval=0, maxval=action_dim
    )
    key, _key = jr.split(key)
    rv = jr.beta(_key, 1.0, 1.0, shape=(num_steps, num_samples))
    idx = rv < bias_prob

    def step_fn(carry, xs):
        actions = carry
        idx, rand_actions = xs
        new_actions = jnp.where(idx, actions, rand_actions)

        return new_actions, actions

    _, action_sequences = lax.scan(
        step_fn, action_sequences[0], (idx, action_sequences)
    )
    return action_sequences


def plan(
    state,
    rollout_fn: Callable,
    action_dim: int,
    key: jr.PRNGKey,
    probs: jnp.ndarray = None,
    current_plan: jnp.ndarray = None,
    num_steps: int = 10,
    num_policies: int = 128,
    num_samples_per_policy: int = 5,
    topk_ratio: float = 0.1,
    random_ratio: float = 0.5,
    alpha: float = 1.0,
    temperature: float = 10,
    normalize: bool = True,
    iters: int = 1,
    gamma: float = 0.99,
    repeat_prob: float = 0.0,
    info_gain: float = 0.0,
    lazy_reward: bool = False,
    sample_action: bool = False,
    **kwargs,
):
    topk = int(num_policies * topk_ratio)
    num_random_samples = int(num_policies * random_ratio)

    if probs is None:
        probs = jnp.full((num_steps, action_dim), 1.0 / action_dim)
    else:
        probs = jnp.roll(probs, -1, axis=0)
        probs = probs.at[-1, :].set(1 / probs.shape[-1])

    if current_plan is not None:
        current_plan = jnp.roll(current_plan, -1, axis=0)

    for i in range(iters):
        # sample from the given probs
        key, subkey = jr.split(key)
        actions = jr.categorical(
            subkey,
            jnp.expand_dims(jnp.log(probs + 1e-16), -2),
            shape=(num_steps, num_policies),
        )

        # overwrite final x actions with random smoothed actions
        if num_random_samples > 0:
            key, subkey = jr.split(key)
            random_actions = _smooth_actions(
                subkey, num_steps, num_random_samples, action_dim, repeat_prob
            )
            actions = actions.at[:, -num_random_samples:].set(random_actions)

        # if we have a current plan, keep that in the sample set always
        if current_plan is not None:
            actions = actions.at[:, 0].set(current_plan)

        # force constant action policies in first k items after current plan
        for k in range(action_dim):
            actions = actions.at[:, k + 1].set(k)

        # draw a number of samples for each policy
        repeated_actions = actions.repeat(num_samples_per_policy, axis=1)

        key, subkey = jr.split(key)
        states, switches, rmm_switches, expected_utility, expected_info_gain = (
            rollout_fn(subkey, state, repeated_actions[..., None])
        )

        # average utility/info gain over policy samples
        expected_utility = expected_utility.reshape(
            num_steps, num_policies, num_samples_per_policy, 1
        ).mean(axis=2)
        expected_info_gain = expected_info_gain.reshape(
            num_steps, num_policies, num_samples_per_policy, 1
        ).mean(axis=2)

        # reshape to have num_steps, num_policies, num_samples_per_policy
        states = states.reshape(
            num_steps,
            num_policies,
            num_samples_per_policy,
            states.shape[-2],
            states.shape[-1],
        )
        switches = switches.reshape(
            num_steps, num_policies, num_samples_per_policy, switches.shape[-1]
        )
        rmm_switches = rmm_switches.reshape(
            num_steps, num_policies, num_samples_per_policy, rmm_switches.shape[-1]
        )

        rewards = expected_utility + info_gain * expected_info_gain

        if lazy_reward:
            # prefer lazy policies with action sequences that are biased towards noop at end
            lazy_rewards = (-jnp.arange(rewards.shape[1]) * 1e-2)[None, :]
            lazy_rewards * (actions > 0)
            rewards += lazy_rewards

        probs, _ = _refit(
            probs, actions, rewards, topk, alpha, temperature, gamma, normalize
        )

    idx = rewards.sum(0)[:, 0].argsort()[-1]
    new_plan = actions[:, idx]

    if not sample_action:
        action = new_plan[0]
    else:
        # sample from the probs or rather pick the top rewarding action sequence?
        key, subkey = jr.split(key)
        action = jr.choice(subkey, probs.shape[-1], p=probs[0])

    return action, {
        "states": states,
        "switches": switches,
        "rmm_switches": rmm_switches,
        "rewards": rewards,
        "actions": actions,
        "probs": probs,
        "current_plan": new_plan,
        "expected_utility": expected_utility,
        "expected_info_gain": expected_info_gain,
    }


def _refit(
    probs, actions, rewards, topk, alpha, temperature=10, gamma=0.99, normalize=True
):
    T, A = probs.shape

    num_steps = rewards.shape[0]
    discounts = gamma ** jnp.arange(num_steps)
    discounted_rewards = rewards * jnp.expand_dims(discounts, (-1, -2))
    cum_rewards = discounted_rewards.sum(0).squeeze(-1)[None]

    if normalize:
        cum_rewards = min_max_normalization(cum_rewards)

    topk_indices = jnp.argsort(-cum_rewards, axis=-1)[..., :topk]
    a_top = jnp.take_along_axis(actions, topk_indices, axis=-1)
    r_top = jnp.take_along_axis(cum_rewards, topk_indices, axis=-1)

    # match moment
    w = jnn.softmax(temperature * r_top, axis=-1)[..., None]
    one_hot_actions = jax.nn.one_hot(a_top, A)
    counts = jnp.sum(w * one_hot_actions, axis=1)
    probs_new = counts / counts.sum(axis=-1, keepdims=True)
    probs_updated = alpha * probs_new + (1 - alpha) * probs
    return probs_updated, r_top


def min_max_normalization(v: jnp.ndarray, dim: int = -1):
    v_max = v.max(dim, keepdims=True)
    v_min = v.min(dim, keepdims=True)

    v_norm = jnp.where(v_max != v_min, (v - v_min) / (v_max - v_min + 1e-16), v)
    return v_norm
