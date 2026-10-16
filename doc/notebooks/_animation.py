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
    bar_linewidth=2.5,
    joint_size=0.11,
    support_size=0.25,
    force_length=0.65,
    torque_radius=0.22,
    figsize=(9, 7),
):
    """Animate a planar bar model with symbols sized relative to its bars.

    joint_size, support_size, force_length and torque_radius are fractions
    of the median bar length. Marker sizes and line widths are converted to
    screen units so that they remain legible across lattice sizes.
    """
    bars = model.bars
    forces = model.external_forces
    n_bars = len(bars)
    n_q = len(model.q)
    trajectory = np.asarray(trajectory, dtype=float)

    if n_bars == 0 or n_q != 3 * n_bars:
        raise ValueError("Expected a planar bar_joint_model with (x, y, theta) per bar")
    if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] < n_q:
        raise ValueError("trajectory must be a non-empty 2D array containing model states")

    q_frames = trajectory[:, :n_q].reshape(-1, n_bars, 3)
    lengths = np.asarray([bar.length for bar in bars], dtype=float)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("All bar lengths must be positive and finite")
    typical_length = float(np.median(lengths))

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
    plot_span = max(x_max - x_min, y_max - y_min, typical_length)
    padding = max(0.04 * plot_span, 0.35 * typical_length)

    if force_values is None:
        vectors = []
        for force in forces:
            try:
                vectors.append((float(force.fx), float(force.fy)))
            except (TypeError, ValueError):
                vectors.append((1.0, 0.0))
        force_vectors = np.asarray(vectors, dtype=float).reshape(-1, 2)
    else:
        try:
            force_vectors = np.asarray(force_values, dtype=float).reshape(-1, 2)
        except (TypeError, ValueError) as error:
            raise ValueError("force_values must contain numeric (fx, fy) pairs") from error
        if len(force_vectors) != len(forces):
            raise ValueError("force_values must have one (fx, fy) pair for each force")

    if torque_values is None:
        torques = []
        for force in forces:
            try:
                torques.append(float(force.torque))
            except (TypeError, ValueError):
                torques.append(1.0)
        torques = np.asarray(torques, dtype=float)
    else:
        try:
            torques = np.asarray(torque_values, dtype=float).reshape(-1)
        except (TypeError, ValueError) as error:
            raise ValueError("torque_values must contain numeric torques") from error
        if len(torques) != len(forces):
            raise ValueError("torque_values must have one value for each force")

    force_magnitudes = np.linalg.norm(force_vectors, axis=1)
    max_force = force_magnitudes.max() if len(force_magnitudes) else 0.0
    force_scale = force_length * typical_length / max_force if max_force > 0 else 0.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.canvas.draw()

    # Convert model lengths to display points, and cap symbols to avoid
    # overwhelming dense lattices or disappearing on very large ones.
    pixels_per_unit = min(
        ax.bbox.width / (ax.get_xlim()[1] - ax.get_xlim()[0]),
        ax.bbox.height / (ax.get_ylim()[1] - ax.get_ylim()[0]),
    )
    points_per_unit = pixels_per_unit * 72.0 / fig.dpi
    bar_width = float(np.clip(bar_linewidth, 0.5, 4.0))
    joint_diameter = float(np.clip(joint_size * typical_length * points_per_unit, 1.2, 9.0))
    support_diameter = float(np.clip(support_size * typical_length * points_per_unit, 3.0, 15.0))
    joint_area = joint_diameter ** 2
    support_area = support_diameter ** 2
    support_half_length = support_size * typical_length / 2.0
    torque_r = torque_radius * typical_length

    bar_lines = [
        ax.plot([], [], lw=bar_width, color=f"C{i % 10}",
                solid_capstyle="round", zorder=2)[0]
        for i in range(n_bars)
    ]
    joint_artist = ax.scatter([], [], s=joint_area, color="black", zorder=4)
    fixed_support_artist = ax.scatter(
        [], [], marker="+", s=support_area, linewidths=1.5,
        color="black", zorder=5
    )
    support_artists = []
    for support in model.supports:
        if support.dof_in_x != support.dof_in_y:
            line = ax.plot([], [], color="black", lw=1.5, zorder=5)[0]
        else:
            line = None
        support_artists.append(line)
    support_line_artists = [a for a in support_artists if a is not None]

    force_artist = None
    if forces:
        force_artist = ax.quiver(
            np.zeros(len(forces)), np.zeros(len(forces)),
            np.zeros(len(forces)), np.zeros(len(forces)),
            angles="xy", scale_units="xy", scale=1,
            color="tab:red", width=0.004, headwidth=4,
            headlength=5, zorder=6,
        )

    torque_artists = []

    def point_on_bar(q, bar_index, s):
        x, y, theta = q[3 * bar_index: 3 * bar_index + 3]
        return np.array([x, y]) + s * lengths[bar_index] * np.array(
            [np.cos(theta), np.sin(theta)]
        )

    def update(frame):
        q = q_frames[frame].reshape(-1)
        for i, line in enumerate(bar_lines):
            line.set_data(endpoints[frame, i, :, 0], endpoints[frame, i, :, 1])

        joints = [
            point_on_bar(q, joint.bar_index[0], joint.s[0])
            for joint in model.joints
        ]
        joint_artist.set_offsets(np.asarray(joints, dtype=float).reshape(-1, 2))

        fixed_supports = []
        for support, artist in zip(model.supports, support_artists):
            point = point_on_bar(q, support.bar_index, support.s)
            if not support.dof_in_x and not support.dof_in_y:
                fixed_supports.append(point)
            elif artist is not None:
                if support.dof_in_x:
                    artist.set_data(
                        [point[0] - support_half_length, point[0] + support_half_length],
                        [point[1], point[1]],
                    )
                else:
                    artist.set_data(
                        [point[0], point[0]],
                        [point[1] - support_half_length, point[1] + support_half_length],
                    )
        fixed_support_artist.set_offsets(
            np.asarray(fixed_supports, dtype=float).reshape(-1, 2)
        )

        load_points = np.asarray(
            [point_on_bar(q, force.bar_index, force.s) for force in forces],
            dtype=float,
        ).reshape(-1, 2)
        if force_artist is not None:
            force_artist.set_offsets(load_points)
            force_artist.set_UVC(
                force_vectors[:, 0] * force_scale,
                force_vectors[:, 1] * force_scale,
            )

        for artist in torque_artists:
            artist.remove()
        torque_artists.clear()
        for point, torque in zip(load_points, torques):
            if torque == 0:
                continue
            sign = 1 if torque > 0 else -1
            angles = np.linspace(
                0.25 * np.pi, 0.25 * np.pi + sign * 1.5 * np.pi, 65
            )
            vertices = np.column_stack((
                point[0] + torque_r * np.cos(angles),
                point[1] + torque_r * np.sin(angles),
            ))
            codes = np.full(len(vertices), Path.LINETO, dtype=np.uint8)
            codes[0] = Path.MOVETO
            arrow = FancyArrowPatch(
                path=Path(vertices, codes), arrowstyle="-|>",
                mutation_scale=float(np.clip(0.7 * support_diameter, 3, 12)),
                color="tab:purple", linewidth=1.5, zorder=7,
            )
            ax.add_patch(arrow)
            torque_artists.append(arrow)

        artists = (
            *bar_lines, joint_artist, fixed_support_artist,
            *support_line_artists, *torque_artists,
        )
        if force_artist is not None:
            artists += (force_artist,)
        return artists

    anim = FuncAnimation(
        fig, update, frames=len(q_frames), interval=interval, blit=False
    )
    if save_animation:
        anim.save("animation.gif", writer="pillow", fps=30, dpi=150)
    else:
        display(HTML(anim.to_jshtml()))
    plt.close(fig)
