import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch
from IPython.display import HTML, display


def animate_bar_model(model, trajectory, interval=30, save_animation=False):
    """Animate bars, applied point forces, and applied planar torques."""
    bars = model.bars
    forces = model.external_forces
    n_bars = len(bars)
    n_q = len(model.q)
    trajectory = np.asarray(trajectory, dtype=float)

    if n_bars == 0 or n_q != 3 * n_bars:
        raise ValueError(
            "Expected a planar bar_joint_model with (x, y, theta) per bar"
        )
    if (
        trajectory.ndim != 2
        or trajectory.shape[0] == 0
        or trajectory.shape[1] < n_q
    ):
        raise ValueError(
            "trajectory must be a non-empty 2D array containing model states"
        )

    # Each bar contributes (x, y, theta) to the configuration vector.
    q_frames = trajectory[:, :n_q].reshape(-1, n_bars, 3)
    lengths = np.asarray([bar.length for bar in bars], dtype=float)
    directions = np.stack(
        (np.cos(q_frames[:, :, 2]), np.sin(q_frames[:, :, 2])), axis=-1
    )
    endpoints = (
        q_frames[:, :, None, :2]
        + directions[:, :, None, :]
        * lengths[None, :, None, None]
        * np.array([-0.5, 0.5])[None, None, :, None]
    )

    x_min, x_max = endpoints[..., 0].min(), endpoints[..., 0].max()
    y_min, y_max = endpoints[..., 1].min(), endpoints[..., 1].max()
    plot_span = max(x_max - x_min, y_max - y_min, 1.0)
    padding = 0.25 * plot_span

    force_vectors = np.asarray(
        [(force.fx, force.fy) for force in forces], dtype=float
    ).reshape(-1, 2)
    force_magnitudes = np.linalg.norm(force_vectors, axis=1)
    max_force = force_magnitudes.max() if len(force_magnitudes) else 0.0
    force_scale = 0.18 * plot_span / max_force if max_force > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    bar_lines = [
        ax.plot([], [], lw=3, color=f"C{i % 10}", solid_capstyle="round")[0]
        for i in range(n_bars)
    ]
    joint_artist = ax.scatter([], [], s=35, color="black", zorder=4)
    support_artist = ax.scatter(
        [], [], marker="+", s=90, color="black", zorder=5
    )

    force_artist = None
    if forces:
        force_artist = ax.quiver(
            np.zeros(len(forces)),
            np.zeros(len(forces)),
            np.zeros(len(forces)),
            np.zeros(len(forces)),
            angles="xy",
            scale_units="xy",
            scale=1,
            color="tab:red",
            width=0.006,
            zorder=6,
        )

    torque_artists = []

    def point_on_bar(q, bar_index, s):
        x, y, theta = q[3 * bar_index : 3 * bar_index + 3]
        direction = np.array([np.cos(theta), np.sin(theta)])
        return np.array([x, y]) + s * bars[bar_index].length * direction

    def update(frame):
        q = q_frames[frame].reshape(-1)

        for i, line in enumerate(bar_lines):
            line.set_data(
                endpoints[frame, i, :, 0],
                endpoints[frame, i, :, 1],
            )

        joints = [
            point_on_bar(q, joint.bar_index[0], joint.s[0])
            for joint in model.joints
        ]
        supports = [
            point_on_bar(q, support.bar_index, support.s)
            for support in model.supports
        ]
        joint_artist.set_offsets(np.asarray(joints).reshape(-1, 2))
        support_artist.set_offsets(np.asarray(supports).reshape(-1, 2))

        load_points = np.asarray(
            [
                point_on_bar(q, force.bar_index, force.s)
                for force in forces
            ]
        ).reshape(-1, 2)

        if force_artist is not None:
            force_artist.set_offsets(load_points)
            force_artist.set_UVC(
                force_vectors[:, 0] * force_scale,
                force_vectors[:, 1] * force_scale,
            )

        # Redraw torque arrows at their moving points of application.
        for artist in torque_artists:
            artist.remove()
        torque_artists.clear()

        torque_radius = 0.045 * plot_span
        for point, force in zip(load_points, forces):
            torque = force.torque
            if torque == 0:
                continue

            # Positive generalized torque follows increasing theta: counterclockwise.
            sign = 1 if torque > 0 else -1
            angles = np.linspace(
                0.25 * np.pi,
                0.25 * np.pi + sign * 1.5 * np.pi,
                65,
            )
            vertices = np.column_stack(
                (
                    point[0] + torque_radius * np.cos(angles),
                    point[1] + torque_radius * np.sin(angles),
                )
            )
            codes = np.full(len(vertices), Path.LINETO, dtype=np.uint8)
            codes[0] = Path.MOVETO

            arrow = FancyArrowPatch(
                path=Path(vertices, codes),
                arrowstyle="-|>",
                mutation_scale=12,
                color="tab:purple",
                linewidth=1.8,
                zorder=7,
            )
            ax.add_patch(arrow)
            torque_artists.append(arrow)

            label = ax.text(
                point[0] + torque_radius,
                point[1] + torque_radius,
                f"M={torque:g}",
                color="tab:purple",
                fontsize=9,
                zorder=8,
            )
            torque_artists.append(label)

        artists = (*bar_lines, joint_artist, support_artist, *torque_artists)
        if force_artist is not None:
            artists += (force_artist,)
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=len(q_frames),
        interval=interval,
        blit=False,
    )

    if save_animation:
        anim.save("animation.gif", writer="pillow", fps=30, dpi=150)
    else:
        display(HTML(anim.to_jshtml()))

    plt.close(fig)
