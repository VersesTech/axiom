import io

import jax
import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from scipy.stats import norm

from axiom.models import rmm as rmm_tools
from axiom.models import imm as imm_tools


def fig2img(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, facecolor="white", format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    plt.close(fig)
    return im[:, :, :3]


def draw_ellipse(position, covariance, ax=None, nsigs=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    if covariance.shape == (2, 2):
        U, s, Vt = jnp.linalg.svd(covariance)
        angle = jnp.degrees(jnp.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * jnp.sqrt(s)
    else:
        angle = 0
        width, height = 2 * jnp.sqrt(covariance)

    kwargs["angle"] = angle
    kwargs["edgecolor"] = "black"
    kwargs["facecolor"] = kwargs.get("color", None)
    del kwargs["color"]
    if nsigs is None:
        nsigs = list(range(1, 4))

    for nsig in nsigs:
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, **kwargs))


def draw_ellipses(means, covars, ax=None, nsigs=None, zorder=1, scatter=True, **kwargs):
    ax = ax or plt.gca()
    nsig = nsigs or range(1, 4)

    U, s, Vt = jax.vmap(jnp.linalg.svd)(covars)
    vals = 2 * jax.vmap(jnp.sqrt)(s)
    widths, heights = vals[:, 0], vals[:, 1]
    angles = jax.vmap(lambda u: jnp.degrees(jnp.arctan2(u[1, 0], u[0, 0])))(U)

    colors = kwargs.get("colors", [None] * means.shape[0])

    if kwargs.get("edgecolors", None) is not None:
        edgecolors = kwargs["edgecolors"]
        del kwargs["edgecolors"]
    else:
        edgecolors = [kwargs.get("edgecolor", "black")] * means.shape[0]

    if kwargs.get("colors", None) is not None:
        del kwargs["colors"]

    for color, angle, position, width, height, edgecolor in zip(
        colors, angles, means, widths, heights, edgecolors
    ):
        kwargs["edgecolor"] = edgecolor

        if color is not None:
            kwargs["facecolor"] = color

        kwargs["angle"] = angle
        for nsig in nsigs:
            ax.add_patch(
                Ellipse(position, nsig * width, nsig * height, **kwargs, zorder=zorder)
            )

    if scatter:
        ax.scatter(
            means[:, 0],
            means[:, 1],
            color=colors,
            marker=".",
            alpha=kwargs.get("alpha", 1.0),
        )


def add_colorbar(fig, ax, plot_im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(plot_im, cax=cax, orientation="vertical")


def transform_mvn(scale, offset, mean, cova):
    A = jnp.diag(scale)
    new_mean = A.dot(mean) + offset
    new_cova = jnp.dot(A, jnp.dot(cova, A.T))
    return new_mean, new_cova


def plot_elbo_smm(elbo, width, height):
    fig, ax = plt.subplots(figsize=((2.1 / height) * width, 2.1))
    ax.imshow(elbo.reshape(height, width))
    idx = jnp.argmin(elbo)
    ax.scatter(idx % width, idx // width, color="red", marker="x")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig2img(fig)


def plot_qz_smm(qz, width, height):
    fig, ax = plt.subplots(figsize=((2.1 / height) * width, 2.1))
    cmap = plt.get_cmap("tab20", qz.shape[-1])
    assignments = cmap(qz[0].argmax(-1).reshape(height, width))[:, :, :3]
    ax.imshow(assignments)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig2img(fig)


def make_empty_figure(width, height, figsize=(2.1, 2.1)):
    fig, ax = plt.subplots(figsize=((figsize[0] / height) * width, figsize[1]))
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig2img(fig)


def plot_rollouts_with_switch_colors(
    xs_traj, pred_switches, offsets, stdevs, width, height, tracked_obj_idx
):
    """
    This function takes a rolled out trajectory of object states and predicted switch states from an RMM / TMM /  SMM model (where the rolling out is done using the TMM
    and RMM, and the encoding / projecting to image coordinates is done using the SMM), and plots the positions of the objects in the scene over time, colored by the switch index.
    """

    T = xs_traj.shape[0]

    # temporarily reshape decoded_xs_traj to be (T*len(tracked_obj_idx), 2) so we can vmap the transformation
    xs_traj = xs_traj[:, tracked_obj_idx, :2].reshape((T * len(tracked_obj_idx), 2))
    pred_pos = jax.vmap(lambda mu: stdevs.flatten()[:2] * mu + offsets.flatten()[:2])(
        xs_traj
    )

    pred_pos = pred_pos.reshape((T, len(tracked_obj_idx), 2))

    pred_switches_only_tracked = pred_switches[:, tracked_obj_idx]

    # make a 2-D trajectory and scatter plot of the positions of each object, colored by the switch index
    fig, ax = plt.subplots(figsize=((2.1 / height) * width, 2.1))
    for obj_idx in range(len(tracked_obj_idx)):
        _map = {
            int(k): v
            for v, k in enumerate(jnp.unique(pred_switches_only_tracked[:, obj_idx]))
        }
        ax.plot(
            pred_pos[:, obj_idx, 0],
            pred_pos[:, obj_idx, 1],
            label=f"Slot {tracked_obj_idx[obj_idx]}",
            zorder=-1,
        )
        ax.scatter(
            pred_pos[:, obj_idx, 0],
            pred_pos[:, obj_idx, 1],
            c=[_map[int(i)] for i in pred_switches_only_tracked[:, obj_idx]],
            marker=".",
            cmap="tab20",
        )

    ax.set_aspect("equal")
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])
    ax.legend(fontsize=6)

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.show()
    return fig2img(fig)


def plot_decoded_positions_with_switch_colors(
    decoded_mu_hist, offsets, stdevs, width, height, switch_hist, tracked_obj_idx
):
    """ "
    This function takes a history of decoded slot-means (obtained via marginalizing out <\log p(y|x, A, \Sigma)>_Q(A, Sigma, x)
    from an SMM model, and plots the positions of the objects in the scene over time, colored by the switch index.
    """
    T = decoded_mu_hist.shape[0]

    # temporarily reshape decoded_mu_hist to be (T*len(tracked_obj_idx), 4) so we can vmap the transformation
    decoded_mu_hist = decoded_mu_hist[:, tracked_obj_idx].reshape(
        (T * len(tracked_obj_idx),) + decoded_mu_hist.shape[2:]
    )
    transformed_positions = jax.vmap(
        lambda mu: stdevs.flatten() * mu + offsets.flatten()
    )(decoded_mu_hist)

    pos_hist = transformed_positions[..., :2]
    pos_hist = pos_hist.reshape((T, len(tracked_obj_idx), 2))

    switch_hist_only_tracked = switch_hist[:, tracked_obj_idx]

    # make a 2-D trajectory and scatter plot of the positions of each object, colored by the switch index
    fig, ax = plt.subplots(figsize=((2.1 / height) * width * 1.5, 2.1))
    for obj_idx in range(len(tracked_obj_idx)):
        _map = {
            int(k): v
            for v, k in enumerate(jnp.unique(switch_hist_only_tracked[:, obj_idx]))
        }
        ax.plot(
            pos_hist[:, obj_idx, 0],
            pos_hist[:, obj_idx, 1],
            label=f"Slot {tracked_obj_idx[obj_idx]}",
            zorder=-1,
        )
        ax.scatter(
            pos_hist[:, obj_idx, 0],
            pos_hist[:, obj_idx, 1],
            c=[_map[int(i)] for i in switch_hist_only_tracked[:, obj_idx]],
            marker=".",
            cmap="tab20",
        )

    ax.set_aspect("equal")
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])
    fig.legend(loc="center right", fontsize="small", edgecolor="white")

    ax.axis("off")
    plt.subplots_adjust(left=0, right=2.0 / 3, bottom=0, top=1)
    # plt.show()
    return fig2img(fig)


def plot_qx_smm(decoded_mu, decoded_sigma, offsets, stdevs, width, height, qz=None):
    mu, cova = jax.vmap(
        lambda mu, si: transform_mvn(
            stdevs.flatten(),
            offsets.flatten(),
            mu,
            si,
        )
    )(decoded_mu, decoded_sigma)

    pos, col = mu[..., :2], mu[..., 2:].clip(0, 255.0) / 255.0

    # just pick the first 3 dims randomly as a color dim now
    if col.shape[-1] > 3:
        col = col[..., :3].clip(0, 1)

    cova = cova[..., :2, :2]

    if qz is not None:
        assignments = qz[0, ...].argmax(-1)
        heights = [(assignments == i).sum() for i in range(qz.shape[-1])]
        indices = jnp.argwhere(jnp.asarray(heights) != 0)[:, 0]
    else:
        indices = jnp.arange(cova.shape[0])

    fig, ax = plt.subplots(figsize=((2.1 / height) * width, 2.1))
    for i in indices:
        draw_ellipse(
            pos[i],
            covariance=cova[i],
            alpha=0.125,
            color=tuple(col[i].tolist()),
            ax=ax,
        )

    ax.scatter(pos[indices, 0], pos[indices, 1], c=col[indices])
    ax.set_aspect("equal")
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    return fig2img(fig)


def plot_rmm(
    rmm: rmm_tools.RMM,
    imm: imm_tools.IMM = None,
    width=160,
    height=210,
    colorize="switch",
    indices=None,
    return_ax=False,
    highlight_idcs=None,
    fig_ax=None,
    scatter=True,
    color_only_identity=False,
):
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=((8.4 / height) * width, 8.4))
    else:
        fig, ax = fig_ax
    ax.set_xlim([-1, 1])
    ax.set_ylim([1, -1])
    ax.set_xticks([])
    ax.set_yticks([])

    cmap = plt.get_cmap("viridis")
    tab20 = plt.get_cmap("tab20")

    if indices is None:
        indices = rmm.used_mask > 0

    mu = rmm.model.continuous_likelihood.mean[indices, :2, 0]
    si = rmm.model.continuous_likelihood.expected_sigma()[indices, :2, :2]

    edgecolors = None
    if colorize == "switch":
        sw = rmm.model.discrete_likelihoods[4].mean()[indices, ..., 0]
        colors = sw.argmax(-1)
        colors = colors / colors.max()
        colors = tab20(colors)
    elif colorize == "reward":
        rw = rmm.model.discrete_likelihoods[3].mean()[indices, ..., 0]
        rw = rw.argmax(-1) / rw.shape[-1]
        c = {
            -1: (1, 0, 0),  # Red
            0: (1, 1, 0),  # Yellow
            1: (0, 1, 0),  # Green
        }
        cmap = LinearSegmentedColormap.from_list("custom_cmap", [c[-1], c[0], c[1]])
        colors = cmap(rw)
    elif colorize == "cluster":
        identities = (
            rmm.model.discrete_likelihoods[0].mean()[indices, ..., 0].argmax(-1)
        )

        interact_identities = (
            rmm.model.discrete_likelihoods[1].mean()[indices, ..., 0].argmax(-1)
        )
        colors = (
            (
                (
                    imm.model.continuous_likelihood.mean[
                        identities, (2 - 2 * int(color_only_identity)) :, 0
                    ]
                    / 100
                )
                * 0.5
                + 0.5
            )
            .clip(0, 1)
            .tolist()
        )

        edgecolors = (
            (
                (
                    imm.model.continuous_likelihood.mean[
                        interact_identities, (2 - 2 * int(color_only_identity)) :, 0
                    ]
                    / 100
                )
                * 0.5
                + 0.5
            )
            .clip(0, 1)
            .tolist()
        )
    elif colorize == "infogain":
        ig = 1 / rmm.model.prior.alpha - 1 / rmm.model.prior.alpha.sum()
        ig = ig[indices]
        colors = cmap(ig)
    else:
        colors = cmap(jnp.ones(mu.shape[0]))

    draw_ellipses(
        mu,
        si,
        ax,
        alpha=0.65,
        colors=colors,
        linewidth=2,
        nsigs=[3],
        scatter=scatter,
        edgecolors=edgecolors if edgecolors is not None else colors,
    )

    if highlight_idcs is not None:
        for j, highlight_idx in enumerate(highlight_idcs):
            i = jnp.where(indices == highlight_idx)[0]
            if len(i) > 0:
                i = i[0]
                draw_ellipses(
                    mu[i : i + 1, :2, 0],
                    si[i : i + 1],
                    ax,
                    alpha=0.10,
                    colors=["black"],
                    edgecolor=tab20(2 * i),
                    linewidth=2,
                    nsigs=[3],
                    zorder=2,
                    scatter=scatter,
                    edgecolors=edgecolors,
                )
            else:
                plt.scatter([0], [0], marker="x", color=tab20(2 * j), s=50)

    if return_ax:
        return fig, ax
    return fig2img(fig)


def plot_reward_clusters(rmm, return_ax=False):
    fig, ax = plt.subplots(1, 3, figsize=(6, 2))
    for i, a in enumerate(ax.flatten()):
        select = rmm.used_mask > 0
        select = select & (
            rmm.model.discrete_likelihoods[-2].alpha[:, :, 0].argmax(-1) == (i)
        )
        x = rmm.model.continuous_likelihood.mean[select, :2, :]

        a.scatter(x[:, 0], x[:, 1])
        a.set_title(f"Reward {i - 1}")
        a.set_xlim(-1, 1)
        a.set_ylim(1, -1)
        a.set_xticks([])
        a.set_yticks([])

    plt.suptitle("Cluster means associated with reward")
    plt.subplots_adjust(top=0.7)
    if return_ax:
        return fig, ax
    return fig2img(fig)


def plot_reward_cluster_ellipses(
    rmm, imm, return_ax=False, fig_ax=None, color_only_identity=False
):
    if fig_ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(6, 2))
    else:
        fig, ax = fig_ax
    for i, a in enumerate(ax.flatten()):
        select = rmm.used_mask > 0
        select = select & (
            rmm.model.discrete_likelihoods[-2].alpha[:, :, 0].argmax(-1) == (i)
        )

        x = rmm.model.continuous_likelihood.mean[select, :2, 0]
        si = rmm.model.continuous_likelihood.expected_sigma()[select, :2, :2]

        objects = rmm.model.discrete_likelihoods[0].alpha[select, :, 0].argmax(-1)
        colors = (
            imm.model.continuous_likelihood.mean[
                objects, (2 - 2 * int(color_only_identity)) :, 0
            ]
            / 100
        )
        colors = (colors * 0.5 + 0.5).clip(0, 1).tolist()

        other_objects = rmm.model.discrete_likelihoods[1].alpha[select, :, 0].argmax(-1)
        other_colors = (
            imm.model.continuous_likelihood.mean[
                other_objects, (2 - 2 * int(color_only_identity)) :, 0
            ]
            / 100
        )
        other_colors = (other_colors * 0.5 + 0.5).clip(0, 1).tolist()

        draw_ellipses(x, si, a, colors=colors, edgecolors=other_colors, nsigs=[3])
        a.scatter(x[:, 0], x[:, 1], marker=".", color="black", s=2)

        # a.scatter(x[:, 0], x[:, 1])
        a.set_title(f"Reward {i - 1}")
        a.set_xlim(-2, 2)
        a.set_ylim(2, -2)
        a.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], color="black", alpha=0.5)
        a.set_xticks([])
        a.set_yticks([])

    if fig_ax is None:
        plt.suptitle("Cluster means associated with reward")
        plt.subplots_adjust(top=0.7)
    if return_ax:
        return fig, ax
    return fig2img(fig)


def plot_identity_model(imm, return_ax=False, color_only_identity=False):
    num_object_types = imm.model.continuous_likelihood.mean.shape[0]

    # ensure that the number of subplots are compatible with the total number of object types
    if num_object_types <= 8:
        n_rows, n_cols = 1, num_object_types
    else:
        n_rows = int(np.ceil(np.sqrt(num_object_types)))
        n_cols = int(np.ceil(num_object_types / n_rows))
        while (n_rows * n_cols) < num_object_types:
            n_rows += 1
            n_cols = int(np.ceil(num_object_types / n_rows))

    fig, ax = plt.subplots(n_rows, n_cols)

    for object_label, a in enumerate(ax.flatten()[:num_object_types]):
        if not color_only_identity:
            width, height = imm.model.continuous_likelihood.mean[object_label, :2, 0]
            co = (
                imm.model.continuous_likelihood.mean[object_label, 2:, 0] / 100
            ) * 0.5 + 0.5
        else:
            width, height = 0.3, 0.3
            co = (
                imm.model.continuous_likelihood.mean[object_label, :, 0] / 100
            ) * 0.5 + 0.5

        a.add_patch(
            Ellipse(
                (0.0, 0.0),
                width,
                height,
                facecolor=co.clip(0, 1).tolist(),
                edgecolor="black",
            )
        )
        a.set_xlim([-0.25, 0.25])
        a.set_ylim([-0.25, 0.25])
        a.set_title(f"{bool(imm.used_mask[object_label])}", fontsize=8)
        a.set_xticks([])
        a.set_yticks([])

    # Hide unused subplots
    for a in ax.flatten()[num_object_types:]:
        a.axis("off")

    plt.suptitle("Learned shapes")
    if return_ax:
        return fig, ax
    return fig2img(fig)


def str_to_col_triplet(c):
    h = c.lstrip("#")
    return jnp.array([int(h[i : i + 2], 16) for i in (0, 2, 4)])


def col_triplet_to_str(c):
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


def hue_shift(color, amount=jnp.pi / 2):
    if isinstance(color, str):
        color = str_to_col_triplet(color)
    if color.dtype is not jnp.floating:
        color = color / 255
    hsv = jnp.array(matplotlib.colors.rgb_to_hsv(color))
    low = hsv.at[0].set((hsv[0] - amount) % (1.0))
    high = hsv.at[0].set((hsv[0] + amount) % (1.0))
    low_rgb = (matplotlib.colors.hsv_to_rgb(low) * 255).astype(jnp.int32)
    high_rgb = (matplotlib.colors.hsv_to_rgb(high) * 255).astype(jnp.int32)
    return col_triplet_to_str(low_rgb), col_triplet_to_str(high_rgb)


def smm_states_to_coords(positions, stats):
    return positions * stats["stdevs"][None, None, :2] + stats["offset"][None, None, :2]


def time_plot(ax, states, rewards, colors, stats=None, alpha=1.0):
    """plot a trajectory

    :param states: oc states of shape [T, N objects, 4]
    :param rewards: reward sequence for this rollout of shape [T]
    :param colors: the colors for each object
    :param alpha: the alpha scale of the plotted points

    The end points of the reward plotting points can be controlled by changing the
    `CMAP_LOWER` and `CMAP_UPPER` globals of this module.
    """
    time_horizon = len(states)
    if stats is None:
        states = states[:, :, :2]
        # shape is now T, Object, 2
    else:
        # SMM DATA
        states = smm_states_to_coords(states[..., :2], stats)

    if len(colors) == 0:
        colors = ["#FFFFFF"] * states.shape[-2]

    for i in range(states.shape[-2]):
        state = states[:, i]
        color = colors[i]
        if not isinstance(color, str):
            color = col_triplet_to_str(color)
        lc, hc = hue_shift(color, amount=0.02)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "SMM-plot", [lc, color, hc]
        )
        alpha_range = alpha * (jnp.linspace(1, 0, time_horizon) ** 2)
        color = cmap(rewards[i])
        alpha_range = jnp.broadcast_to(alpha_range[:, None], (time_horizon, 4))
        alpha_range = alpha_range.at[:, :3].set(color[:3])
        ax.scatter(state[:, 0], state[:, 1], c=alpha_range)
    ax.set_xlim(0, 210)
    ax.set_ylim(160, 0)
    return ax


def plot_rollouts(obs, states, rewards, colors, horizon=-1, stats=None, title=None):
    """Plot a rollout in OC position coordinates overlayed on an RGB frame.

    :param obs: The RGB buffer from env.render() to show as a background.
    :param states: The OC states [T, B, O, 4].
    :param rewards: The rewards [T, B, 1].
    :param colors: The color to use for each object.
    :param horizon: Optional truncation of the rollout.
    :param stats: Pass in a SMM stats dictionary if the model uses SMM.

    :return frame: A frame of 160x210 pixels with the resulting plot.

    The end points of the reward plotting points can be controlled by changing the
    `CMAP_LOWER` and `CMAP_UPPER` globals of this module.
    """

    # plot results in a 160 x 210 image
    my_dpi = 50
    fig = plt.figure(figsize=(160 / my_dpi, 210 / my_dpi), dpi=my_dpi, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    if title is not None:
        ax.text(
            0.5,
            0.97,
            title,
            verticalalignment="top",
            horizontalalignment="center",
            transform=ax.transAxes,
            color="white",
            fontsize=16,
        )

    ax.imshow(obs)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    # Clip rewards into a sensible range for cmap
    rewards = jnp.clip(rewards, -1, 1)
    for i in range(states.shape[1]):
        time_plot(
            ax,
            states[:horizon, i],
            rewards[:, i, 0],
            colors,
            stats=stats,
            alpha=0.9 / states.shape[1],
        )
    ax.set_xlim(0, 160)
    ax.set_ylim(210, 0)
    frame = fig2img(fig)
    ax.clear()
    return frame


def plot_plan(
    obs,
    plan_info,
    tracked_obj_ids,
    stats,
    decoded_mu=None,
    topk=5,
    descending=True,
    indices=None,
):
    tracked_obj_ids = tracked_obj_ids
    obj_ids = jnp.argwhere(tracked_obj_ids).squeeze()
    if obj_ids.shape == ():
        obj_ids = obj_ids[None]

    if decoded_mu is None:
        if plan_info["states"].shape[-1] > 6:
            colors = (plan_info["states"][0, 0, 0, obj_ids, -3:] * 128 + 128).astype(
                jnp.uint8
            )
        else:
            colors = []
    else:
        colors = (decoded_mu[obj_ids, 2:] * 128 + 128).astype(jnp.uint8)

    rewards = plan_info["rewards"]
    if indices is not None:
        idx = indices
    else:
        idx = jnp.argsort(rewards.sum(axis=0)[:, 0], descending=descending)[:topk]
    states_topk = plan_info["states"][:, idx]
    rewards_topk = rewards[:, idx]
    utility_topk = plan_info["expected_utility"][:, idx]
    info_gain_topk = plan_info["expected_info_gain"][:, idx]

    r = rewards_topk.sum(axis=0).mean()
    u = utility_topk.sum(axis=0).mean()
    ig = info_gain_topk.sum(axis=0).mean()
    title = f"r: {r:.1f}, u: {u:.1f}, ig: {ig:.1f}"

    return plot_rollouts(
        obs,
        states_topk[:, :, 0, obj_ids, :],
        rewards_topk,
        colors,
        stats=stats,
        title=title,
    )


def rollout_samples_lineplot(x_pred, x_gt, object_ids, color_vals=None, rewards=None):
    if len(x_pred.shape) == 3:
        # this function expects a batch of rollouts
        x_pred = x_pred[None, ...]

    fig, ax = plt.subplots(len(object_ids), 6, figsize=(18, 9), sharex=True)
    if len(object_ids) == 1:
        ax = [ax]

    labels = ["x", "y", "u", "v_x", "v_y"]

    if color_vals is not None:
        normalize = plt.Normalize(min(color_vals), max(color_vals))
        colormap = plt.cm.viridis
        colors = [colormap(normalize(v)) for v in color_vals]

    for o, obj_idx in enumerate(object_ids):
        for i, a in enumerate(ax[o][:-1]):
            a.set_title(f"${labels[i]}$")
            if x_gt is not None:
                a.plot(
                    x_gt[:, obj_idx, i],
                    color="black",
                    linestyle="dashed",
                    label="ground truth",
                )

            for k in range(x_pred.shape[0]):
                c = colors[k] if color_vals is not None else "orange"

                a.plot(x_pred[k, :, obj_idx, i], zorder=-1, color=c)
            if i == 0:
                # x coordinate
                a.set_ylim(-1, 1)
            if i == 1:
                # y coordinate
                a.set_ylim(1, -1)

    if rewards is not None:
        for o, obj_idx in enumerate(object_ids):
            ax[o][-1].plot(rewards)

    plt.subplots_adjust(
        bottom=0.10, top=0.9, hspace=0.25, wspace=0.25, left=0.05, right=0.99
    )

    frame = fig2img(fig)
    plt.close()
    return frame


def rollout_lineplot(x_pred, x_gt, switch_pred, switch_gt, actions, obj_idx):
    switches = jnp.concatenate(
        [switch_pred[:, obj_idx].flatten(), switch_gt[:, obj_idx].flatten()], axis=0
    )
    _map = {int(switch): i for i, switch in enumerate(jnp.unique(switches))}

    fig, ax = plt.subplots(3, 6, figsize=(18, 9), sharex=True)
    labels = ["x", "y", "u", "v_x", "v_y", "v_u"]
    for i, a in enumerate(ax[0]):
        a.set_title(f"${labels[i]}$")
        a.plot(x_gt[:, obj_idx, i], zorder=-1, color="black", linestyle="dashed")
        a.scatter(
            jnp.arange(x_gt.shape[0]),
            x_gt[:, obj_idx, i],
            c=[_map[int(j)] for j in switch_gt[:, obj_idx]],
            cmap="tab20",
        )
        a.set_ylim(-1, 1)

    for i, a in enumerate(ax[1]):
        a.set_title(f"${labels[i]}$")
        a.plot(
            x_gt[:, obj_idx, i],
            zorder=-1,
            color="black",
            linestyle="dashed",
            label="ground truth",
        )
        a.scatter(
            jnp.arange(x_gt.shape[0]),
            x_pred[:, obj_idx, i],
            c=[_map[int(j)] for j in switch_pred[:, obj_idx]],
            cmap="tab20",
        )
        a.plot(x_pred[:, obj_idx, i], zorder=-1, color="orange", label="predicted")
        a.set_ylim(-1, 1)

    for i, a in enumerate(ax[2]):
        a.scatter(
            jnp.arange(actions.shape[0]), actions, c=actions, cmap="tab20c", marker="o"
        )
        a.plot(jnp.arange(actions.shape[0]), actions, zorder=-1)
        a.set_xlabel("time")

    plt.suptitle(f"Tracked object {obj_idx}", fontsize=20)
    fig.legend(
        *ax[1][-1].get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        fontsize=12,
        edgecolor="white",
    )
    plt.subplots_adjust(
        bottom=0.10, top=0.9, hspace=0.25, wspace=0.25, left=0.05, right=0.99
    )

    ax[0][0].set_ylabel("color=inferred switch", fontsize=15)
    ax[1][0].set_ylabel("color=predicted switch", fontsize=15)
    ax[2][0].set_ylabel("color=Action", fontsize=15)

    frame = fig2img(fig)
    plt.close()
    return frame


def line_plots_before_and_after_reduce(
    config,
    rmm_before,
    rmm_after,
    tmm_model,
    x_sequence,
    a_sequence,
    s_sequence,
    tracked_object_ids,
):
    x_before, s_before, *_ = rmm_tools.rollout(
        rmm_before + rmm_before * config.rmm.shared * (config.rmm.num_slots - 1),
        tmm_model,
        x_sequence[0],
        a_sequence,
        tracked_object_ids,
        num_features=config.rmm.num_features,
    )

    x_after, s_after, *_ = rmm_tools.rollout(
        rmm_after + rmm_after * config.rmm.shared * (config.rmm.num_slots - 1),
        tmm_model,
        x_sequence[0],
        a_sequence,
        tracked_object_ids,
        num_features=config.rmm.num_features,
    )

    labels = ["x", "y", "u", "v_x", "v_y", "v_u"]
    ims = dict({})

    tracked_object_idx = jnp.where(tracked_object_ids)[0]
    for obj_idx in tracked_object_idx:
        switches = jnp.concatenate(
            [
                s_before[:, obj_idx].flatten(),
                s_after[:, obj_idx].flatten(),
                s_sequence[:, obj_idx],
            ],
            axis=0,
        )
        _map = {int(switch): i for i, switch in enumerate(jnp.unique(switches))}

        fig, ax = plt.subplots(2, 6, figsize=(18, 6), sharex=True)
        for i, a in enumerate(ax[0]):
            a.set_title(f"${labels[i]}$")
            a.plot(x_before[:, obj_idx, i], color="tab:blue", label="before")
            a.plot(x_after[:, obj_idx, i], color="tab:orange", label="after")
            a.plot(
                x_sequence[:, obj_idx, i],
                color="black",
                linestyle="dotted",
                label="ground truth",
            )
            a.scatter(
                jnp.arange(x_before.shape[0]),
                x_before[:, obj_idx, i],
                c=[_map[int(j)] for j in s_before[:, obj_idx]],
                cmap="tab20",
            )

        for i, a in enumerate(ax[1]):
            a.set_title(f"${labels[i]}$")
            a.plot(x_before[:, obj_idx, i], color="tab:blue", label="before")
            a.plot(x_after[:, obj_idx, i], color="tab:orange", label="after")
            a.plot(
                x_sequence[:, obj_idx, i],
                color="black",
                linestyle="dotted",
                label="ground truth",
            )
            a.scatter(
                jnp.arange(x_after.shape[0]),
                x_after[:, obj_idx, i],
                c=[_map[int(j)] for j in s_after[:, obj_idx]],
                cmap="tab20",
            )

        ax[0][0].set_ylabel("color=predicted switch before")
        ax[1][0].set_ylabel("color=predicted switch after")
        plt.suptitle(f"Rollout before/after pruning (object {obj_idx})", fontsize=20)
        fig.legend(
            *ax[1][-1].get_legend_handles_labels(),
            loc="lower center",
            ncol=3,
            fontsize=12,
            edgecolor="white",
        )
        plt.subplots_adjust(
            bottom=0.10, top=0.9, hspace=0.25, wspace=0.25, left=0.05, right=0.99
        )

        ims[f"rollout_before_after_reduce_{obj_idx}"] = fig2img(fig)
        plt.close()
    return ims


def plot_rmm_detail(mixture, index, dobs=None):
    if dobs is not None:
        _, ell = rmm_tools._e_step(mixture, dobs)
        ell = f"ELL: {ell[0, index]:.2f}"
        dobs = dobs[:, :, 0]
    else:
        ell = ""

    mu = mixture.likelihood.mean[index]
    si = mixture.likelihood.expected_sigma()[index]

    fig, all_axes = plt.subplots(2, 12, figsize=(2.5 * 12, 2.5 * 2))

    labs = [
        "$x$",
        "$y$",
        "$u$",
        "$v_x$",
        "$v_y$",
        "$v_u$",
        "$\\text{shape}_x$",
        "$\\text{shape}_y$",
        "red",
        "green",
        "blue",
    ]
    labs = labs + [f"other {lab}" for lab in labs]
    labs = labs + ["action"]
    labs = [l + f" ({i})" for i, l in enumerate(labs)]

    for i in range(2):
        axes = all_axes[i]

        for j in range(len(axes)):
            ax = axes[j]

            idx = i * len(axes) + j
            if idx > len(labs) - 1:
                ax.axis("off")
                ax.legend(
                    *axes[j - 1].get_legend_handles_labels(),
                    edgecolor="white",
                    loc="center",
                )
                break

            mu_i = mu[None, idx, 0]
            si_i = jnp.sqrt(si[None, idx, idx])
            x_axis = jnp.linspace(-1, 1, 10000) * si_i * 3 + mu_i
            ax.plot(
                x_axis,
                norm.pdf(x_axis, mu_i, si_i),
                color="darkblue",
                label="Component marginal",
            )
            ax.spines[["right", "top"]].set_visible(False)

            logpdf = ""
            if dobs is not None:
                logpdf = norm.pdf(dobs[0, idx], mu_i, si_i)
                logpdf = f"\n{logpdf[0]:.2f}"

            ax.set_title(labs[idx] + logpdf)
            if dobs is not None:
                ax.scatter(
                    dobs[0, idx],
                    0,
                    marker="o",
                    color="tab:orange",
                    label="Sample",
                )

                if len(dobs) > 1:
                    ax.scatter(
                        dobs[1:, idx],
                        dobs[1:, idx] * 0,
                        marker=".",
                        color="black",
                        label="buffer",
                        zorder=-1,
                    )

        if idx > len(labs):
            break
    plt.subplots_adjust(wspace=0.5, hspace=0.75, top=0.8)
    plt.suptitle(f"Component {index} {ell}", fontsize=20)
    plt.show()


def plot_rmm_detail(model, index, c_obs=None, d_obs=None, predict=True):
    if c_obs is not None:
        w_disc = jnp.array([1.0] * len(model.discrete_likelihoods))
        if predict:
            w_disc = w_disc.at[-2:].set(0.0)
        _, c_ell, d_ell = model._e_step(c_obs, d_obs, w_disc)
        elogp_str = f"elogp: {c_ell[index]:.2f} {d_ell[index]:.2f}"
    else:
        elogp_str = ""

    n_continuous = model.continuous_likelihood.shape[1]
    n_discrete = len(model.discrete_likelihoods)

    n_col = n_continuous + n_discrete
    fig, ax = plt.subplots(1, n_col, figsize=(2 * n_col, 2))

    cont_labels = [
        "$x$",
        "$y$",
        "$u$",
        "$v_x$",
        "$v_y$",
        "$dx$",
        "$d_y$",
    ]

    disc_labels = [
        "self id",
        "other id",
        "used",
        "action",
        "reward",
        "tmm switch",
    ]

    mu = model.continuous_likelihood.mean[index]
    si = model.continuous_likelihood.expected_sigma()[index]
    for i in range(n_continuous):
        mu_i = mu[None, i, 0]
        si_i = jnp.sqrt(si[None, i, i])
        x_axis = jnp.linspace(-1, 1, 10000) * si_i * 3 + mu_i
        ax[i].plot(
            x_axis,
            norm.pdf(x_axis, mu_i, si_i),
            color="darkblue",
            label="Component marginal",
        )
        # ax[i].set_xlim([-1, 1])

        if c_obs is not None:
            ax[i].scatter(
                c_obs[i, 0],
                norm.pdf(c_obs[i, 0], mu_i, si_i),
                color="tab:orange",
                zorder=2,
            )

    for i in range(n_discrete):
        mu = model.discrete_likelihoods[i].mean()[index, :, 0]
        ax[i + n_continuous].bar(jnp.arange(mu.shape[0]), mu, color="darkblue")

        ax[i + n_continuous].text(
            mu.argmax(),
            1.05 * mu.max(),
            f"{mu.argmax()}",
            ha="center",
            va="bottom",
            color="darkblue",
            fontsize=10,
        )

        ax[i + n_continuous].set_ylim([0, 1.2 * mu.max()])

        if d_obs is not None:
            o = d_obs[i][:, 0].argmax()
            ax[i + n_continuous].scatter(
                o, mu.flatten()[o], color="tab:orange", marker="o"
            )

    [a.spines[["right", "top"]].set_visible(False) for a in ax.flatten()]
    [a.set_title(cont_labels[i]) for i, a in enumerate(ax.flatten()[:n_continuous])]
    [a.set_title(disc_labels[i]) for i, a in enumerate(ax.flatten()[n_continuous:])]
    plt.suptitle(f"Component {index} [{elogp_str}]", fontsize=15)
    plt.subplots_adjust(wspace=0.35, top=0.75)
    plt.show()


def plot_convolved_rewards_over_time(
    reward_timeseries,
    size=1000,
    random_agent_baseline=None,
):
    raw_rewards = jnp.array(reward_timeseries)

    r = jnp.concatenate([jnp.zeros(size), raw_rewards])
    r = jnp.convolve(r, jnp.ones(size) / size, mode="valid")

    if random_agent_baseline is not None:
        rr = jnp.array(random_agent_baseline)
        rr = jnp.concatenate([jnp.zeros(size), rr])
        rr = jnp.convolve(rr, jnp.ones(size) / size, mode="valid")

    fig, ax = plt.subplots(1, 2, figsize=(6, 2.0), sharex=True)
    [a.grid("on") for a in ax]
    [a.set_facecolor("whitesmoke") for a in ax]

    ax[0].plot(raw_rewards, alpha=1, color="crimson")
    ax[0].set_title("True Rewards")

    if random_agent_baseline is not None:
        ax[1].plot(rr, color="tab:blue", label="random")
    ax[1].plot(r, color="crimson", label="agent")
    ax[1].set_title(f"Reward ({size}-step moving-average)")
    ax[1].legend(fontsize=8)

    [a.set_xlabel("# observations") for a in ax]

    plt.suptitle("Reward timeseries)")
    plt.subplots_adjust(top=0.75, wspace=0.35)

    frame = fig2img(fig)
    return frame


def plot_obs_and_info(obs, x, title=None, indices=None):
    my_dpi = 50
    fig = plt.figure(figsize=(160 / my_dpi, 210 / my_dpi), dpi=my_dpi, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    if title is not None:
        ax.text(
            0.5,
            0.97,
            title,
            verticalalignment="top",
            horizontalalignment="center",
            transform=ax.transAxes,
            color="white",
            fontsize=16,
        )

    if obs is not None:
        ax.imshow(obs, alpha=1.0, extent=[-1, 1, 1, -1], aspect="auto")
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_xlim(-1, 1)
    ax.set_ylim(1, -1)

    if indices is None:
        indices = jnp.arange(x.shape[0])

    for i in indices:
        position = x[i, :2]
        # convert shape back to covar
        covar = (x[i, 6:8] / 3) ** 2
        color = jnp.clip((x[i, 8:11] + 1) / 2, min=0, max=1)
        if x[i, 2] == 0:
            draw_ellipse(
                position,
                covariance=covar,
                alpha=0.25,
                color=tuple(color.tolist()),
                ax=ax,
            )

    frame = fig2img(fig)
    return frame


def generate_report(
    rewards,
    random_rewards,
    num_components,
    rmm_before,
    rmm_after,
    imm,
    color_only_identity=False,
):
    def make_plot(rewards, random_rewards, fig_ax=None):
        size = 1000

        r = jnp.array(rewards)[-10_000:]
        r = jnp.concatenate([jnp.zeros(size), r])
        r = jnp.convolve(r, jnp.ones(size) / size, mode="valid")

        if random_rewards is not None:
            rr = jnp.array(random_rewards)
            rr = jnp.concatenate([jnp.zeros(size), rr])
            rr = jnp.convolve(rr, jnp.ones(size) / size, mode="valid")

        # get a running average of the rewards for the last `size` episodes
        if fig_ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(6, 2.0), sharex=True)
        else:
            fig, ax = fig_ax
        [a.grid("on") for a in ax]
        [a.set_facecolor("whitesmoke") for a in ax]

        # ax[0].plot(jnp.array(random_rewards), alpha=0.25, color="tab:blue")
        ax[0].plot(jnp.array(rewards)[-10_000:], alpha=1, color="crimson")
        ax[0].set_title("Reward")

        if random_rewards is not None:
            ax[1].plot(rr, color="tab:blue")
        ax[1].plot(r, color="crimson")
        ax[1].set_title("Reward (1K average)")

        if fig_ax is None:
            [a.set_xlabel("# observations") for a in ax]
            plt.subplots_adjust(top=0.75, wspace=0.35)
            plt.show()

    def plot_interacting(rmm, ax, default=True, interacting=False):
        num_types = rmm.model.discrete_likelihoods[1].alpha.shape[1] - 1
        if not default and not interacting:
            select = rmm.used_mask > 0
            select = select & (
                rmm.model.discrete_likelihoods[1].alpha[:, :, 0].argmax(-1) == num_types
            )
            select = select & (
                rmm.model.discrete_likelihoods[-1].alpha[:, :, 0].argmax(-1) > 0
            )
            ax.set_title("non interacting\nnon default")
        elif not default and interacting:
            select = rmm.used_mask > 0
            select = select & (
                rmm.model.discrete_likelihoods[1].alpha[:, :, 0].argmax(-1) < num_types
            )
            select = select & (
                rmm.model.discrete_likelihoods[-1].alpha[:, :, 0].argmax(-1) > 0
            )
            ax.set_title("interacting\nnon default")
        elif default and not interacting:
            select = rmm.used_mask > 0
            select = select & (
                rmm.model.discrete_likelihoods[1].alpha[:, :, 0].argmax(-1) == num_types
            )
            select = select & (
                rmm.model.discrete_likelihoods[-1].alpha[:, :, 0].argmax(-1) == 0
            )
            ax.set_title("not interacting\ndefault")
        elif default and interacting:
            select = rmm.used_mask > 0
            select = select & (
                rmm.model.discrete_likelihoods[1].alpha[:, :, 0].argmax(-1) < num_types
            )
            select = select & (
                rmm.model.discrete_likelihoods[-1].alpha[:, :, 0].argmax(-1) == 0
            )
            ax.set_title("interacting\ndefault")

        plot_rmm(
            rmm,
            imm,
            width=20,
            height=20,
            colorize="cluster",
            indices=jnp.where(select)[0],
            return_ax=True,
            fig_ax=(None, ax),
            color_only_identity=color_only_identity,
        )

    fig, ax = plt.subplots(3, 7, figsize=(7 / 5 * 12, 6))
    make_plot(rewards, random_rewards, fig_ax=(fig, ax[0, :2]))
    ax[0, 2].plot(num_components, color="crimson")
    ax[0, 2].set_facecolor("whitesmoke")
    ax[0, 2].set_title("Num components")
    ax[0, 2].grid("on")
    [a.axis("off") for a in ax[0, 3:]]

    plot_reward_cluster_ellipses(
        rmm=rmm_before,
        imm=imm,
        fig_ax=(fig, ax[1, :3]),
        return_ax=True,
        color_only_identity=color_only_identity,
    )
    ax[1, 0].set_ylabel("Before BMR")
    plot_interacting(rmm_before, ax[1, 3], default=True, interacting=True)
    plot_interacting(rmm_before, ax[1, 4], default=False, interacting=True)
    plot_interacting(rmm_before, ax[1, 5], default=True, interacting=False)
    plot_interacting(rmm_before, ax[1, 6], default=False, interacting=False)

    plot_reward_cluster_ellipses(
        rmm=rmm_after,
        imm=imm,
        fig_ax=(fig, ax[2, :3]),
        return_ax=True,
        color_only_identity=color_only_identity,
    )
    ax[2, 0].set_ylabel("After BMR")
    plot_interacting(rmm_after, ax[2, 3], default=True, interacting=True)
    plot_interacting(rmm_after, ax[2, 4], default=False, interacting=True)
    plot_interacting(rmm_after, ax[2, 5], default=True, interacting=False)
    plot_interacting(rmm_after, ax[2, 6], default=False, interacting=False)

    plt.subplots_adjust(wspace=0.35, hspace=0.5)
    plt.show()
