import os
import argparse
import wandb
import yaml
from typing import Sequence
from pathlib import Path

import dataclasses
from dataclasses import dataclass, fields

from axiom.models import smm, tmm, rmm, imm
from axiom import planner


DATA_DIR = os.environ.get("DATA_DIR", default=str(Path(__file__).parent / "data"))
os.makedirs(DATA_DIR, exist_ok=True)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    id: str
    group: str
    seed: int
    game: str
    num_steps: int
    smm: smm.SMMConfig | Sequence[smm.SMMConfig]
    imm: imm.IMMConfig
    tmm: tmm.TMMConfig
    rmm: rmm.RMMConfig
    planner: planner.PlannerConfig
    moving_threshold: float | Sequence[float] = 1e-2
    used_threshold: float | Sequence[float] = 0.2
    min_track_steps: Sequence[int] = (1, 1)
    max_steps_tracked_unused: int = 10
    prune_every: int = 500
    use_unused_counter: bool = True
    project: str = "axiom"
    precision_type: str = "float32"
    layer_for_dynamics: int = 0  # which layer to use for dynamics
    warmup_smm: bool = False  # warmup SMM
    num_warmup_steps: int = 50
    velocity_clip_value: float = 7.5e-4  # clip abs velocities below this value to 0
    perturb: str = None  # which env perturbation to apply - defaults to None
    perturb_step: int = 5000  # when to apply the perturbation
    remap_color: bool = False  # remap color of the objects on color perturbation
    bmr_samples: int = 2000  # number of samples to use for Bayesian model reduction
    bmr_pairs: int = 2000  # number of pairs to use for Bayesian model reduction


def parse_floats(s):
    try:
        return [float(item) for item in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "All values must be floats, separated by commas."
        )


def expand_used_threshold(values: Sequence[float], n_layers: int) -> list[float]:
    """
    If you pass a single value, returns:
       [v, v/2, v/4, ..., v/(2**(n_layers-1))]
    If you pass exactly n_layers values, returns them unchanged.
    Otherwise, errors out.
    """
    if n_layers > 1:
        if len(values) == 1:
            v0 = values[0]
            return [v0 / (2**l) for l in range(n_layers)]
        elif len(values) == n_layers:
            return list(values)
        else:
            raise ValueError(
                f"used_threshold must have length 1 or {n_layers}, got {len(values)}"
            )
    else:
        if len(values) != 1:
            raise ValueError(
                f"For n_smm_layers=1, used_threshold must have length 1, got {len(values)}"
            )
        return list(values)


def expand_layer_values(
    values: Sequence[float], name: str, n_layers: int
) -> list[float]:
    """
    Given a list of floats and the number of SMM layers, return a list of length n_layers:
      - If len(values) == 1, repeat that value for every layer
      - If len(values) == n_layers, return it unchanged
      - Otherwise, error out
    """
    if n_layers > 1:
        if len(values) == 1:
            return values * n_layers
        elif len(values) == n_layers:
            return list(values)
        else:
            raise ValueError(
                f"{name} must have length 1 or {n_layers}, got {len(values)}"
            )
    else:
        if len(values) != 1:
            raise ValueError(
                f"For n_smm_layers=1, {name} must have length 1, got {len(values)}"
            )
        return list(values)


def create_smm_configs(args):
    """
    Process layer-by-layer SMM arguments and create SMM configs.

    Supports two modes:
    1. Single values for each parameter will be automatically expanded for multi-layer SMMs
    2. Lists of values with length matching n_smm_layers will be used directly
    """
    # Get the layerwise parameters
    num_slots_arg = args.num_slots
    input_dim_arg = args.input_dim
    dof_offset_arg = args.dof_offset
    scale_arg = args.scale
    threshold_arg = args.smm_eloglike_threshold

    # Lists to store the expanded parameters
    num_slots = []
    input_dim = []
    dof_offset = []
    scale = []
    smm_eloglike_threshold = []

    # Check if we need to expand single values to multiple layers
    if args.n_smm_layers > 1:
        # num_slots: If a single value is provided, halve it for each layer
        if len(num_slots_arg) == 1:
            num_slots = [num_slots_arg[0] // (2**l) for l in range(args.n_smm_layers)]
        else:
            if len(num_slots_arg) != args.n_smm_layers:
                raise ValueError(
                    f"num_slots must have length 1 or {args.n_smm_layers}, got {len(num_slots_arg)}"
                )
            num_slots = num_slots_arg

        # input_dim: If a single value is provided, use it for the first layer and 2 for subsequent layers
        if len(input_dim_arg) == 1:
            input_dim = [input_dim_arg[0]] + [4] * (args.n_smm_layers - 1)
        else:
            if len(input_dim_arg) != args.n_smm_layers:
                raise ValueError(
                    f"input_dim must have length 1 or {args.n_smm_layers}, got {len(input_dim_arg)}"
                )
            input_dim = input_dim_arg

        # dof_offset: If a single value is provided, use it for the first layer and 2.0 for subsequent layers
        if len(dof_offset_arg) == 1:
            dof_offset = [dof_offset_arg[0]] + [2.0] * (args.n_smm_layers - 1)
        else:
            if len(dof_offset_arg) != args.n_smm_layers:
                raise ValueError(
                    f"dof_offset must have length 1 or {args.n_smm_layers}, got {len(dof_offset_arg)}"
                )
            dof_offset = dof_offset_arg

        # scale: If a single value is provided, use it for the first layer and [0.075, 0.075] for subsequent layers
        if len(scale_arg) == 1:
            scale = [scale_arg[0]] + [[0.075, 0.075, 0.025, 0.025]] * (
                args.n_smm_layers - 1
            )

        else:
            if len(scale_arg) != args.n_smm_layers:
                raise ValueError(
                    f"scale must have length 1 or {args.n_smm_layers}, got {len(scale_arg)}"
                )
            scale = scale_arg

        # threshold: If a single value is provided, repeat it for all layers
        if len(threshold_arg) == 1:
            smm_eloglike_threshold = threshold_arg * args.n_smm_layers
        else:
            if len(threshold_arg) != args.n_smm_layers:
                raise ValueError(
                    f"smm_eloglike_threshold must have length 1 or {args.n_smm_layers}, got {len(threshold_arg)}"
                )
            smm_eloglike_threshold = threshold_arg
    else:
        # For a single layer, check if the lists have the correct length
        if len(num_slots_arg) != 1:
            raise ValueError(
                f"For n_smm_layers=1, num_slots must have length 1, got {len(num_slots_arg)}"
            )
        if len(input_dim_arg) != 1:
            raise ValueError(
                f"For n_smm_layers=1, input_dim must have length 1, got {len(input_dim_arg)}"
            )
        if len(dof_offset_arg) != 1:
            raise ValueError(
                f"For n_smm_layers=1, dof_offset must have length 1, got {len(dof_offset_arg)}"
            )
        if len(scale_arg) != 1:
            raise ValueError(
                f"For n_smm_layers=1, scale must have length 1, got {len(scale_arg)}"
            )
        if len(threshold_arg) != 1:
            raise ValueError(
                f"For n_smm_layers=1, smm_eloglike_threshold must have length 1, got {len(threshold_arg)}"
            )

        # Use the provided values directly
        num_slots = num_slots_arg
        input_dim = input_dim_arg
        dof_offset = dof_offset_arg
        scale = scale_arg
        smm_eloglike_threshold = threshold_arg

    # Create SMM configs for each layer
    smm_configs = []
    for l in range(args.n_smm_layers):
        config_l = smm.SMMConfig(
            num_slots=num_slots[l],
            eloglike_threshold=smm_eloglike_threshold[l],
            input_dim=input_dim[l],
            slot_dim=2 if l == 0 else 4,
            scale=tuple(scale[l]),
            dof_offset=dof_offset[l],
        )
        smm_configs.append(config_l)

    return tuple(smm_configs)


def get_defaults(parser):
    parser.add_argument("--name", type=str, default="axiom")
    parser.add_argument("--uid", type=str, default=None)
    parser.add_argument("--group", type=str, default="axiom")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--game", type=str, default="Explode")
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--precision_type", type=str, default="float32")

    # unused counter, ties to state dim & data dim
    parser.add_argument("--no_unused_counter", action="store_true")

    # max steps tracked unused
    parser.add_argument(
        "--max_steps_tracked_unused", type=int, default=10
    )  # max steps tracked unused

    # do Bayesian model reduction every n steps
    parser.add_argument("--prune_every", type=int, default=500)

    # Use perturbed game versions
    parser.add_argument(
        "--perturb",
        type=str,
        default=None,
        help="Name of the perturbation to apply (e.g. player_shape).",
    )
    parser.add_argument(
        "--perturb_step",
        type=int,
        default=5000,
        help="Global step at which to fire the perturbation.",
    )
    parser.add_argument("--remap_color", action="store_true", required=False)

    parser.add_argument("--warmup_smm", action="store_true", required=False)
    parser.add_argument("--num_warmup_steps", type=int, default=50)

    # smm params

    parser.add_argument("--n_smm_layers", type=int, default=1)
    parser.add_argument("--layer_for_dynamics", type=int, default=0)

    # Flat lists
    parser.add_argument("--num_slots", type=int, nargs="+", default=[32])
    parser.add_argument("--input_dim", type=int, nargs="+", default=[5])
    parser.add_argument("--dof_offset", type=float, nargs="+", default=[10.0])
    parser.add_argument(
        "--smm_eloglike_threshold", type=float, nargs="+", default=[5.7]
    )
    parser.add_argument("--moving_threshold", type=float, nargs="+", default=[0.003])
    parser.add_argument("--used_threshold", type=float, nargs="+", default=[0.02])
    parser.add_argument("--min_track_steps", type=int, nargs="+", default=[1])

    # Nested list for `scale`
    parser.add_argument(
        "--scale",
        type=parse_floats,
        nargs="+",
        default=[[0.075, 0.075, 0.75, 0.75, 0.75]],
        help="List of comma-separated float lists for each layer, e.g., --scale 0.1,0.1 0.2,0.2",
    )

    # tmm params
    parser.add_argument("--n_total_components", type=int, default=500)
    parser.add_argument("--state_dim", type=int, default=3)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--no_bias", action="store_true")
    parser.add_argument("--sigma_sqr", type=float, default=2.0)
    parser.add_argument("--logp_threshold", type=float, default=-0.00001)
    parser.add_argument("--position_threshold", type=float, default=0.2)
    parser.add_argument("--no_velocity_tmm", action="store_true")
    parser.add_argument("--velocity_clip_value", type=float, default=7.5e-4)

    # imm params
    parser.add_argument("--num_object_types", type=int, default=32)
    parser.add_argument("--cont_scale_identity", type=float, default=0.5)
    parser.add_argument("--i_ell_threshold", type=float, default=-500)
    parser.add_argument("--color_scale_identity", type=float, default=1.0)
    parser.add_argument("--color_only_identity", action="store_true")

    # rmm params
    parser.add_argument("--num_components_per_switch", type=int, default=10)
    parser.add_argument("--interact_with_static", action="store_true", required=False)
    parser.add_argument("--cont_scale_switch", type=float, default=75.0)
    parser.add_argument(
        "--discrete_alphas",
        type=float,
        nargs="+",
        default=[1e-4, 1e-4, 1e-4, 1e-4, 1.0, 1e-4],
    )
    parser.add_argument("--reward_prob_threshold", type=float, default=0.45)
    parser.add_argument("--r_ell_threshold", type=float, default=-10)

    parser.add_argument("--fixed_r", action="store_true", required=False)
    parser.add_argument("--r_interacting", type=float, default=0.075)
    parser.add_argument("--r_interacting_predict", type=float, default=0.075)
    parser.add_argument("--velocity_scale", type=float, default=10.0)
    parser.add_argument(
        "--relative_distance_scale", action="store_true", required=False
    )

    # planner params
    parser.add_argument("--random_actions", action="store_true")
    parser.add_argument("--planning_horizon", type=int, default=32)
    parser.add_argument("--planning_rollouts", type=int, default=512)
    parser.add_argument("--num_samples_per_rollout", type=int, default=3)
    parser.add_argument("--planning_iterations", type=int, default=1)
    parser.add_argument("--repeat_prob", type=float, default=0.0)
    parser.add_argument("--info_gain", type=float, default=0.1)
    parser.add_argument("--sample_action", action="store_true")

    # bmr params
    parser.add_argument("--bmr_samples", type=int, default=2000)
    parser.add_argument("--bmr_pairs", type=int, default=2000)

    return parser


def from_dict(data_clazz, d):
    try:
        fieldtypes = {f.name: f.type for f in fields(data_clazz)}
        return data_clazz(**{f: from_dict(fieldtypes[f], d[f]) for f in d})
    except:
        return d  # Not a dataclass field


def load_config_raw(config_path):
    config = yaml.safe_load(open(os.path.join(DATA_DIR, config_path), "r"))
    return from_dict(ExperimentConfig, config)


def load_config(config_path, seed=0):
    config = load_config_raw(config_path)
    # allow to override seed
    config.seed = seed
    # and generate new run id
    config.id = wandb.util.generate_id()
    return config


def parse_args(args=None):
    if args is None:
        args = []

    # for compatibility with the old scripts
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser = get_defaults(parser)
    args = parser.parse_args(args=args)

    if args.config is not None:
        config = load_config(args.config, args.seed)
        return config

    if args.layer_for_dynamics >= args.n_smm_layers:
        raise ValueError(
            f"Invalid layer_for_dynamics index: {args.layer_for_dynamics} "
            f"(number of layers: {args.n_smm_layers})"
        )

    # Expand layerwise parameters if needed
    smm_configs = create_smm_configs(args)

    # Expand moving_threshold and used_threshold to per-layer lists
    moving_thresholds = expand_layer_values(
        args.moving_threshold, "moving_threshold", args.n_smm_layers
    )
    used_thresholds = expand_used_threshold(args.used_threshold, args.n_smm_layers)

    min_track_steps = expand_layer_values(
        args.min_track_steps, "min_track_steps", args.n_smm_layers
    )

    tmm_config = tmm.TMMConfig(
        n_total_components=args.n_total_components,
        state_dim=args.state_dim,
        dt=args.dt,
        use_bias=not args.no_bias,
        sigma_sqr=args.sigma_sqr,
        logp_threshold=args.logp_threshold,
        position_threshold=args.position_threshold,
        use_unused_counter=not args.no_unused_counter,
        use_velocity=not args.no_velocity_tmm,
        clip_value=args.velocity_clip_value,
    )

    num_features = 5
    data_dim = 2 * (2 * (2 + int(not args.no_unused_counter)) + num_features) + 1

    if len(args.discrete_alphas) != 6:
        if len(args.discrete_alphas) == 1:
            args.discrete_alphas = args.discrete_alphas * 6
        else:
            raise ValueError(
                f"Invalid number of discrete alphas (defaults to 6): {len(args.discrete_alphas)}"
            )

    if args.fixed_r:
        r_interacting = args.r_interacting
        r_interacting_predict = args.r_interacting_predict
        use_relative_distance = False
        num_continuous_dims = 5
        stable_r = True
        forward_predict = True
        use_ellipses_for_interaction = True
        absolute_distance_scale = True
    else:
        r_interacting = args.r_interacting
        r_interacting_predict = args.r_interacting
        use_relative_distance = True
        absolute_distance_scale = not args.relative_distance_scale
        num_continuous_dims = 7
        stable_r = False
        forward_predict = False
        use_ellipses_for_interaction = False

    rmm_config = rmm.RMMConfig(
        num_components_per_switch=args.num_components_per_switch,
        num_switches=tmm_config.n_total_components,
        interact_with_static=args.interact_with_static,
        cont_scale_switch=args.cont_scale_switch,
        r_interacting=r_interacting,
        r_interacting_predict=r_interacting_predict,
        discrete_alphas=tuple(args.discrete_alphas),
        forward_predict=forward_predict,
        reward_prob_threshold=args.reward_prob_threshold,
        stable_r=stable_r,
        num_continuous_dims=num_continuous_dims,
        relative_distance=use_relative_distance,
        r_ell_threshold=args.r_ell_threshold,
        exclude_background=(args.n_smm_layers == 1),
        use_ellipses_for_interaction=use_ellipses_for_interaction,
        velocity_scale=args.velocity_scale,
        absolute_distance_scale=absolute_distance_scale,
    )

    imm_config = imm.IMMConfig(
        num_object_types=args.num_object_types,
        i_ell_threshold=args.i_ell_threshold,
        cont_scale_identity=args.cont_scale_identity,
        color_precision_scale=args.color_scale_identity,
        color_only_identity=args.color_only_identity,
    )

    if args.random_actions:
        planner_config = None
    else:
        planner_config = planner.PlannerConfig(
            num_steps=args.planning_horizon,
            num_policies=args.planning_rollouts,
            num_samples_per_policy=args.num_samples_per_rollout,
            iters=args.planning_iterations,
            repeat_prob=args.repeat_prob,
            info_gain=args.info_gain,
            sample_action=args.sample_action,
        )

    config = ExperimentConfig(
        name=args.name,
        id=wandb.util.generate_id() if args.uid is None else args.uid,
        group=args.group,
        seed=args.seed,
        game=args.game,
        num_steps=args.num_steps,
        precision_type=args.precision_type,
        use_unused_counter=not args.no_unused_counter,
        max_steps_tracked_unused=args.max_steps_tracked_unused,
        smm=smm_configs,
        imm=imm_config,
        tmm=tmm_config,
        rmm=rmm_config,
        planner=planner_config,
        moving_threshold=tuple(moving_thresholds),
        used_threshold=tuple(used_thresholds),
        min_track_steps=tuple(min_track_steps),
        prune_every=args.prune_every,
        layer_for_dynamics=args.layer_for_dynamics,
        warmup_smm=args.warmup_smm,
        num_warmup_steps=args.num_warmup_steps,
        perturb=args.perturb,
        perturb_step=args.perturb_step,
        remap_color=args.remap_color,
        bmr_samples=args.bmr_samples,
        bmr_pairs=args.bmr_pairs,
    )
    return config
