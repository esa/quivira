import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch
from IPython.display import HTML, display


def animate_bar_model(
    model,
    trajectory,
    interval=30,
    save_animation=False,
    torque_values=None,
    force_values=None,
):
    """Animate bars, applied point forces, and planar torque arrows.

    Force values are ``(fx, fy)`` pairs and torque values are scalars, each
    ordered to match ``model.external_forces``. Numeric model values are used
    by default; symbolic values default to positive x force and positive
    counterclockwise torque.
    """
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

    if force_values is None:
        force_vectors = []
        for force in forces:
            try:
                force_vectors.append((float(force.fx), float(force.fy)))
            except (TypeError, ValueError):
                force_vectors.append((1.0, 0.0))
        force_vectors = np.asarray(force_vectors, dtype=float).reshape(-1, 2)
    else:
        try:
            force_vectors = np.asarray(force_values, dtype=float).reshape(-1, 2)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "force_values must contain numeric (fx, fy) pairs for each force"
            ) from error
        if force_vectors.shape[0] != len(forces):
            raise ValueError(
                "force_values must have one (fx, fy) pair for each force"
            )

    if torque_values is None:
        torque_values = []
        for force in forces:
            try:
                torque_values.append(float(force.torque))
            except (TypeError, ValueError):
                torque_values.append(1.0)
    else:
        try:
            torque_values = np.asarray(
                torque_values, dtype=float
            ).reshape(-1)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "torque_values must contain numeric torques for each force"
            ) from error
        if torque_values.size != len(forces):
            raise ValueError(
                "torque_values must have one value for each force"
            )

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
    fixed_support_artist = ax.scatter(
        [], [], marker="+", s=90, color="black", zorder=5
    )
    support_artists = []
    for support in model.supports:
        if support.dof_in_x and not support.dof_in_y:
            line = ax.plot([], [], color="black", lw=2, zorder=5)[0]
        elif support.dof_in_y and not support.dof_in_x:
            line = ax.plot([], [], color="black", lw=2, zorder=5)[0]
        else:
            line = None
        support_artists.append(line)
    support_line_artists = [
        artist for artist in support_artists if artist is not None
    ]

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
        joint_artist.set_offsets(np.asarray(joints).reshape(-1, 2))

        fixed_supports = []
        support_half_length = 0.035 * plot_span
        for support, artist in zip(model.supports, support_artists):
            point = point_on_bar(q, support.bar_index, support.s)
            if not support.dof_in_x and not support.dof_in_y:
                fixed_supports.append(point)
            elif artist is not None:
                if support.dof_in_x:
                    artist.set_data(
                        [point[0] - support_half_length,
                         point[0] + support_half_length],
                        [point[1], point[1]],
                    )
                else:
                    artist.set_data(
                        [point[0], point[0]],
                        [point[1] - support_half_length,
                         point[1] + support_half_length],
                    )
        fixed_support_artist.set_offsets(
            np.asarray(fixed_supports).reshape(-1, 2)
        )

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
        for point, torque in zip(load_points, torque_values):
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
        artists = (
            *bar_lines,
            joint_artist,
            fixed_support_artist,
            *support_line_artists,
            *torque_artists,
        )
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
