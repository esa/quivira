"""Bar-and-joint model for constrained rigid-link systems.

This module stores a compact set of bars, joints, supports, and point loads for
forming symbolic Lagrange equations of motion in a quivira model.
"""

import numpy as np
import heyoka as hy
import quivira as qv

class bar:
    """Rigid bar segment with mass, geometric length, and initial pose.

    The bar stores the center position, orientation, and inertial properties that
    are used to assemble the generalized coordinates of the model.
    """

    x0 = float
    y0 = float
    theta0 = float
    length = float
    mass = float
    inertia = float

    def __init__(self, length, mass):
        """Initialize the bar geometry and inertial properties.

        Args:
            length (:class:`float`): Bar length.

            mass (:class:`float`): Bar mass.
        """
        self.x0 = None
        self.y0 = None
        self.theta0 = None
        self.xd0 = 0
        self.yd0 = 0
        self.thetad0 = 0
        self.length = length
        self.mass = mass
        self.inertia = self.mass * self.length**2 / 12.0

    @classmethod
    def from_point_pair(cls, start, end, mass_per_length=1.0):
        """Construct a bar from two endpoints.

        Args:
            start (:class:`tuple`): Cartesian coordinates ``(x, y)`` of the first
                end.

            end (:class:`tuple`): Cartesian coordinates ``(x, y)`` of the second
                end.

            mass_per_length (:class:`float`, optional): Mass per unit length.
                Default is 1.0.

        Returns:
            :class:`bar`: New bar positioned between the endpoints.
        """
        start_x, start_y = start
        end_x, end_y = end

        length = np.sqrt((end_x - start_x)**2 +(end_y - start_y)**2)
        theta0 = np.arctan2(end_y - start_y, end_x - start_x)

        mass = mass_per_length * length
        x0 = (start_x + end_x) / 2.0
        y0 = (start_y + end_y) / 2.0

        new_bar = cls(length, mass)
        new_bar.set_init_bar_attitude(x0, y0, theta0)

        return new_bar

    def set_init_bar_attitude(self, x0, y0, theta0):
        """Set the initial center position and angle of the bar.

        Args:
            x0 (:class:`float`): Initial center x coordinate.

            y0 (:class:`float`): Initial center y coordinate.

            theta0 (:class:`float`): Initial orientation angle.
        """
        self.x0 = x0
        self.y0 = y0
        self.theta0 = theta0

    def set_init_bar_velocities(self, xd0, yd0, thetad0):
        """Set the initial translational and angular velocities of the bar.

        Args:
            xd0 (:class:`float`): Initial x velocity of the center.

            yd0 (:class:`float`): Initial y velocity of the center.

            thetad0 (:class:`float`): Initial angular velocity.
        """
        self.xd0 = xd0
        self.yd0 = yd0
        self.thetad0 = thetad0

class joint:
    """Constraint tying multiple points on different bars to a common location.

    The joint stores the list of bars and local coordinates that must coincide in
    space at the same time instance.
    """
    # u, v, are symbolic expressions for the point on the bar in local coordinates
    bar_index = []
    s = []

    def __init__(self, bar_index, s):
        """Initialize a joint constraint between multiple bar points.

        Args:
            bar_index (:class:`list`): Indices of the bars in the joint.

            s (:class:`list`): Local coordinates on each bar that must coincide.

        Raises:
            ValueError: Raised when the arrays do not have the same length or
                when fewer than two bars are given.
        """
        if len(bar_index) != len(s):
            raise ValueError("bar_index and s must have the same length")

        if len(bar_index) < 2:
            raise ValueError("A joint must connect at least two bars")
    
        self.bar_index = bar_index
        self.s = s

class support:
    """Support condition constraining the motion of a point on a bar.

    A support can fix translation along the x and/or y directions while leaving the
    remaining kinematic degrees of freedom free.
    """
    # u, v, are symbolic expressions for the point on the bar in local coordinates
    bar_index = int
    s = float
    x = float
    y = float
    dof_in_x = bool
    dof_in_y = bool

    def __init__(self, bar_index, s, x, y, dof_in_x=False, dof_in_y=False):
        """Initialize a support constraint.

        Args:
            bar_index (:class:`int`): Index of the constrained bar.

            s (:class:`float`): Local coordinate of the constrained point.

            x (:class:`float`): Target x position for the support.

            y (:class:`float`): Target y position for the support.

            dof_in_x (:class:`bool`, optional): If ``True``, x motion is free.
                Default is ``False``.

            dof_in_y (:class:`bool`, optional): If ``True``, y motion is free.
                Default is ``False``.
        """
        self.bar_index = bar_index
        self.s = s
        self.x = x
        self.y = y
        self.dof_in_x = dof_in_x
        self.dof_in_y = dof_in_y

class external_force:
    """External point force applied to a bar at a local coordinate."""
    bar_index = int
    s = float
    fx = float
    fy = float
    torque = float

    def __init__(self, bar_index, s, fx, fy, torque = 0):
        """Initialize an external force applied at a point on a bar.

        Args:
            bar_index (:class:`int`): Index of the bar receiving the force.

            s (:class:`float`): Local coordinate of the application point.

            fx (:class:`float`): Force component along the x axis.

            fy (:class:`float`): Force component along the y axis.
        """
        self.bar_index = bar_index
        self.s = s
        self.fx = fx
        self.fy = fy
        self.torque = torque

class bar_joint_model:
    """Assembly of articulated bars with joints, supports, and external loads.

    The model collects the geometric and force data needed to generate a
    Lagrangian formulation and the constraint equations for the system.
    """

    def __init__(self, tolerance = 1e-10):
        """Initialize the model and reset the internal state.

        Args:
            tolerance (:class:`float`, optional): Numerical tolerance used when
                checking constraint satisfaction. Default is ``1e-10``.
        """
        self.tolerance = tolerance 
        self.clear()

    def clear(self):
        """Remove all bars, joints, supports, and forces from the model."""
        self.g = 0.0
        self.bars = []
        self.joints = []
        self.supports = []
        self.external_forces = []

        self.q = []
        self.qd = []

        self.lagrangian = None
        self.F = np.asarray([], dtype=object)
        self.Q_nc = np.asarray([], dtype=object)

    def set_g(self, g):
        """Set the gravitational acceleration for the model.

        Args:
            g (:class:`float`): Gravitational acceleration in the global y
                direction.
        """
        self.g = float(g)

    # def build_from_point_pairs(self, point_pairs):
    #     self.clear()

    #     for start, end in point_pairs:
    #         self.add_bar_from_point_pair(start, end)

    def add_bar(self, length, mass):
        """Append a new bar to the assembly.

        Args:
            length (:class:`float`): Bar length.

            mass (:class:`float`): Bar mass.
        """
        self.bars.append(bar(length, mass))

    def set_init_bar_attitude(self, bar_index, x0, y0, theta0):
        """Set the initial center position and orientation of a bar.

        Args:
            bar_index (:class:`int`): Index of the target bar.

            x0 (:class:`float`): Initial x coordinate of the bar center.

            y0 (:class:`float`): Initial y coordinate of the bar center.

            theta0 (:class:`float`): Initial angle of the bar.
        """
        self.bars[bar_index].set_init_bar_attitude(x0, y0, theta0)

    def set_init_bar_velocities(self, bar_index, xd0, yd0, thetad0):
        """Set the initial translational and angular velocities of a bar.

        Args:
            bar_index (:class:`int`): Index of the target bar.

            xd0 (:class:`float`): Initial x velocity of the center.

            yd0 (:class:`float`): Initial y velocity of the center.

            thetad0 (:class:`float`): Initial angular velocity.
        """
        self.bars[bar_index].set_init_bar_velocities(xd0, yd0, thetad0)

    def add_bar_at_attitude(self, x0, y0, theta0, length, mass):
        """Append a bar and assign its initial attitude in one step.

        Args:
            x0 (:class:`float`): Initial x coordinate of the bar center.

            y0 (:class:`float`): Initial y coordinate of the bar center.

            theta0 (:class:`float`): Initial angle of the bar.

            length (:class:`float`): Bar length.

            mass (:class:`float`): Bar mass.
        """
        self.add_bar(length, mass)
        self.set_init_bar_attitude(-1, x0, y0, theta0)

    def add_bar_from_point_pair(self, start, end, mass_per_length=1.0):
        """Append a bar constructed from its endpoints.

        Args:
            start (:class:`tuple`): Cartesian coordinates ``(x, y)`` of the first
                end.

            end (:class:`tuple`): Cartesian coordinates ``(x, y)`` of the second
                end.

            mass_per_length (:class:`float`, optional): Distributed mass per unit
                length. Default is 1.0.
        """
        self.bars.append(bar.from_point_pair(start, end, mass_per_length))

    def add_joint(self, bar_index, s):
        """Add a joint linking selected points across bars.

        Args:
            bar_index (:class:`list`): Indices of the bars participating in the
                joint.

            s (:class:`list`): Local coordinates of the coincident points on each
                bar.
        """
        self.joints.append(joint(bar_index, s))

    def infer_joints(self):
        """Infer joints from intersections of bars in their initial poses.

        Detects intersections anywhere along the bars, groups bars meeting
        at the same point, and avoids duplicate joints.

        Returns:
            int: Total number of joints after inference.
        """
        tol = self.tolerance

        def cross2(a, b):
            # Calculate the 2D cross product of vectors a and b.
            # It is zero when the vectors are parallel and is used below to calculate the intersection of two lines.
            return a[0] * b[1] - a[1] * b[0]

        # Calculate the start and end of each bar.
        segments = []

        for i, b in enumerate(self.bars):
            # We require initial positions of bars to infer the joints
            if any(v is None for v in (b.x0, b.y0, b.theta0)):
                raise ValueError(f"Bar {i} has no initial attitude, cannot infer joints")

            # Check the stored bar length.
            length = b.length
            if not np.isfinite(length) or length <= tol:
                raise ValueError(f"Bar {i} has invalid or zero length")

            # Calculate the start point and direction vector.
            start = np.asarray(self._initial_point_on_bar(i, -0.5), dtype=float)
            direction = length * np.array([
                np.cos(b.theta0),
                np.sin(b.theta0),
            ])

            segments.append((start, direction, length))

        # Each group represents one joint location.
        # "members" maps bar indices to local s coordinates.
        groups = []

        def register(point, members):
            matches = [
                k for k, group in enumerate(groups)
                if np.linalg.norm(group["point"] - point) <= tol
            ]

            if matches:
                group = groups[matches[0]]

                # Merge existing groups at this location.
                for k in reversed(matches[1:]):
                    other = groups.pop(k)
                    members.extend(other["members"].items())
            else:
                group = {"point": point, "members": {}}
                groups.append(group)

            for i, s in members:
                previous = group["members"].get(i)

                if previous is not None:
                    if abs(previous - s) * self.bars[i].length > tol:
                        raise ValueError(
                            f"Conflicting joint positions on bar {i}"
                        )

                group["members"][i] = s

        # Include manually added joints to avoid duplicates.
        for old in self.joints:
            point = np.asarray(
                self._initial_point_on_bar(
                    old.bar_index[0], old.s[0]
                ),
                dtype=float,
            )
            register(point, list(zip(old.bar_index, old.s)))

        # Check every unique pair of bars.
        for i in range(len(segments)):
            p, r, length_i = segments[i]

            for j in range(i + 1, len(segments)):
                q, v, length_j = segments[j]

                delta = q - p
                denominator = cross2(r, v)

                if abs(denominator) <= tol * length_i * length_j:
                    # Parallel bars: check whether they are collinear.
                    if abs(cross2(delta, r)) > tol * length_i:
                        continue

                    t0 = np.dot(delta, r) / length_i**2
                    t1 = t0 + np.dot(v, r) / length_i**2

                    lo = max(0.0, min(t0, t1))
                    hi = min(1.0, max(t0, t1))

                    if hi < lo - tol / length_i:
                        continue

                    if (hi - lo) * length_i > tol:
                        raise ValueError(
                            f"Bars {i} and {j} overlap along a segment; "
                            "a unique joint location cannot be inferred"
                        )

                    # Collinear bars touching at one endpoint.
                    t = np.clip(0.5 * (lo + hi), 0.0, 1.0)
                    point = p + t * r
                    u = np.dot(point - q, v) / length_j**2

                    if not -tol / length_j <= u <= 1 + tol / length_j:
                        continue

                    u = np.clip(u, 0.0, 1.0)

                else:
                    # Intersection of the two infinite lines.
                    t = cross2(delta, v) / denominator
                    u = cross2(delta, r) / denominator

                    # Intersection must lie on both finite segments.
                    if not (
                        -tol / length_i <= t <= 1 + tol / length_i
                        and -tol / length_j <= u <= 1 + tol / length_j
                    ):
                        continue

                    t = np.clip(t, 0.0, 1.0)
                    u = np.clip(u, 0.0, 1.0)

                    point_i = p + t * r
                    point_j = q + u * v

                    if np.linalg.norm(point_i - point_j) > tol:
                        continue

                    point = 0.5 * (point_i + point_j)

                # Convert endpoint-based parameters to COM-based s.
                register(
                    point,
                    [(i, float(t - 0.5)), (j, float(u - 0.5))]
                )

        # Rebuild the joints, with one joint per location.
        self.joints = []

        for group in groups:
            self.add_joint(
                list(group["members"]),
                list(group["members"].values()),
            )

        return len(self.joints)


    def add_support(self, bar_index, s, dof_in_x, dof_in_y, x=None, y=None):
        """Add a support constraint on a point of a bar.

        Args:
            bar_index (:class:`int`): Index of the constrained bar.

            s (:class:`float`): Local coordinate of the constrained point.

            dof_in_x (:class:`bool`): If ``True``, x motion is unconstrained.

            dof_in_y (:class:`bool`): If ``True``, y motion is unconstrained.

            x (:class:`float`, optional): Support x position. If omitted, the
                initial bar position is used. Default is ``None``.

            y (:class:`float`, optional): Support y position. If omitted, the
                initial bar position is used. Default is ``None``.

        Raises:
            ValueError: Raised if the support position is requested before the bar
                has an initial state.
        """
        if x == None or y == None:
            if self.bars[bar_index].x0 == None or self.bars[bar_index].y0 == None:
                raise ValueError("Bar " + str(bar_index) + " does not have an initial state yet, can't set support")
        
            x, y = self._initial_point_on_bar(bar_index, s)

        self.supports.append(support(bar_index, s, x, y, dof_in_x, dof_in_y))

    def add_external_force(self, bar_index, s, fx, fy, torque=0.0):
        """Add a point load on a bar.

        Args:
            bar_index (:class:`int`): Index of the bar receiving the force.

            s (:class:`float`): Local coordinate of the force application point.

            fx (:class:`float`): Force component along the global x axis.

            fy (:class:`float`): Force component along the global y axis.

            torque (:class:`float`): Torque mathematically positive
        """
        self.external_forces.append(external_force(bar_index, s, fx, fy, torque))

    def build_lagrange_eom(self, return_multipliers=False, return_W=False):
        """Assemble the symbolic equations of motion for the model.

            This method builds the system-level Lagrangian, constraint residuals, and
            external-force terms, then returns the equivalent symbolic first-order
            ODE system for numerical integration. When ``return_multipliers`` or
            ``return_W`` is enabled, the corresponding constraint quantities are
            included in the return value.

            Args:
                return_multipliers (:class:`bool`, optional): If ``True``, also
                    return the symbolic Lagrange multiplier vector associated with
                    the algebraic constraints. Default is ``False``.

                return_W (:class:`bool`, optional): If ``True``, also return the
                    Delassus matrix :math:`W = J M^{-1} J^T`, where ``J`` is the
                    constraint Jacobian and ``M`` is the mass matrix. Default is
                    ``False``.

            Returns:
                :class:`list` or :class:`tuple`: The ODE system as pairs
                ``(state_var, rhs_expr)``. If neither option is enabled, the
                function returns the ODE system directly. If
                ``return_multipliers`` is ``True``, it returns
                ``(ode_system, lambda_mult)``; if ``return_W`` is ``True``, it
                returns ``(ode_system, W)``; and if both are ``True``, it returns
                ``(ode_system, lambda_mult, W)``. The multiplier entries follow the
                ordering of the constraint residuals in ``self.F``.
            """
        self._build_state_variables()
        self._build_lagrangian()
        self._build_constraints()
        self._build_external_forces()
        return qv.lagrange_eom(self.lagrangian, self.F, self.q, self.qd, self.Q_nc, return_multipliers=return_multipliers, return_W=return_W)

    def set_initial_state(self, initial_state):
        """Set the initial bar poses and velocities from a flattened state vector.

        Args:
            initial_state (:class:`list` or :class:`numpy.ndarray`): State vector
                in the same order as :meth:`get_initial_state`, where all
                generalized coordinates are listed first and all generalized
                velocities follow.

        Raises:
            ValueError: Raised when the state length does not match the number of
                bars or when an individual bar state is malformed.

        Returns:
            list: The stacked initial generalized coordinates and velocities.
        """
        flat_state = np.asarray(initial_state, dtype=float).ravel()

        if flat_state.size == 6 * len(self.bars):
            q_size = 3 * len(self.bars)
            for i in range(len(self.bars)):
                q_offset = 3 * i
                qd_offset = q_size + 3 * i

                x0, y0, theta0 = flat_state[q_offset : q_offset + 3]
                xd0, yd0, thetad0 = flat_state[qd_offset : qd_offset + 3]

                self.bars[i].x0 = float(x0)
                self.bars[i].y0 = float(y0)
                self.bars[i].theta0 = float(theta0)
                self.bars[i].xd0 = float(xd0)
                self.bars[i].yd0 = float(yd0)
                self.bars[i].thetad0 = float(thetad0)
            self.check_constraint_fulfillment(self.get_initial_state())
            return

        if len(initial_state) == len(self.bars):
            for i, state in enumerate(initial_state):
                state_array = np.asarray(state, dtype=float).ravel()
                if state_array.size != 6:
                    raise ValueError(
                        f"Initial state for bar {i} must contain [x0, y0, theta0, xd0, yd0, thetad0]"
                    )

                x0, y0, theta0, xd0, yd0, thetad0 = state_array

                self.bars[i].x0 = float(x0)
                self.bars[i].y0 = float(y0)
                self.bars[i].theta0 = float(theta0)
                self.bars[i].xd0 = float(xd0)
                self.bars[i].yd0 = float(yd0)
                self.bars[i].thetad0 = float(thetad0)

            self.check_constraint_fulfillment(self.get_initial_state())
            return

        raise ValueError(
            f"Expected state length {6 * len(self.bars)}, got {len(initial_state)}"
        )

    def get_initial_state(self):
        """Return the stacked initial generalized coordinates and velocities.

        Returns:
            list: State vector containing ``[x0, y0, theta0, xd0, yd0, thetad0]``
            for each bar.
        """
        q0 = []
        for bar in self.bars:
            q0.extend([bar.x0, bar.y0, bar.theta0])

        qd0 = []
        for bar in self.bars:
            qd0.extend([bar.xd0, bar.yd0, bar.thetad0])
        return q0 + qd0

    def check_constraint_fulfillment(self, state):
        if len(state) != 6 * len(self.bars):
            raise ValueError(f"Expected state of length {6 * len(self.bars)}, got {len(state)}")

        n = len(self.bars)

        # Check joints
        for joint_index, joint in enumerate(self.joints):
            ref_bar = joint.bar_index[0]
            ref_s = joint.s[0]

            ref_x, ref_y = self._current_point_on_bar_from_state(state, ref_bar, ref_s)

            ref_theta = state[3 * ref_bar + 2]
            ref_xd, ref_yd, ref_thetad = state[3 * n + 3 * ref_bar : 3 * n + 3 * ref_bar + 3]
            ref_vx = ref_xd - ref_s * self.bars[ref_bar].length * np.sin(ref_theta) * ref_thetad
            ref_vy = ref_yd + ref_s * self.bars[ref_bar].length * np.cos(ref_theta) * ref_thetad

            for bar_index, s in zip(joint.bar_index[1:], joint.s[1:]):
                px, py = self._current_point_on_bar_from_state(state, bar_index, s)

                theta = state[3 * bar_index + 2]
                xd, yd, thetad = state[3 * n + 3 * bar_index : 3 * n + 3 * bar_index + 3]
                vx = xd - s * self.bars[bar_index].length * np.sin(theta) * thetad
                vy = yd + s * self.bars[bar_index].length * np.cos(theta) * thetad

                if abs(px - ref_x) > self.tolerance or abs(py - ref_y) > self.tolerance:
                    raise ValueError(f"State violates joint {joint_index}: bar {bar_index} does not coincide with bar {ref_bar}")

                if abs(vx - ref_vx) > self.tolerance or abs(vy - ref_vy) > self.tolerance:
                    raise ValueError(f"State violates velocity constraint at joint {joint_index}: bar {bar_index} does not move with bar {ref_bar}")

        # Check supports
        for support_index, support in enumerate(self.supports):
            px, py = self._current_point_on_bar_from_state(state, support.bar_index, support.s)

            theta = state[3 * support.bar_index + 2]
            xd, yd, thetad = state[3 * n + 3 * support.bar_index : 3 * n + 3 * support.bar_index + 3]
            vx = xd - support.s * self.bars[support.bar_index].length * np.sin(theta) * thetad
            vy = yd + support.s * self.bars[support.bar_index].length * np.cos(theta) * thetad

            if not support.dof_in_x and abs(px - support.x) > self.tolerance:
                raise ValueError(f"State violates support {support_index} in x: got {px}, expected {support.x}")

            if not support.dof_in_y and abs(py - support.y) > self.tolerance:
                raise ValueError(f"State violates support {support_index} in y: got {py}, expected {support.y}")

            if not support.dof_in_x and abs(vx) > self.tolerance:
                raise ValueError(f"State violates velocity constraint at support {support_index} in x: got {vx}, expected 0")

            if not support.dof_in_y and abs(vy) > self.tolerance:
                raise ValueError(f"State violates velocity constraint at support {support_index} in y: got {vy}, expected 0")

    def print_model(self):
        """Print a human-readable summary of the assembled model."""
        print("Bars:")
        for i, bar in enumerate(self.bars):
            print(f"  Bar {i}: length={bar.length}, mass={bar.mass}")

        print("\nJoints:")
        if len(self.joints) == 0:
            print("  None")
        else:
            for i, joint in enumerate(self.joints):
                connections = [f"bar {bar_index} at s={s}" for bar_index, s in zip(joint.bar_index, joint.s)]
                print(f"  Joint {i}: " + ", ".join(connections))

        print("\nSupports:")
        if len(self.supports) == 0:
            print("  None")
        else:
            for i, support in enumerate(self.supports):
                print(
                    f"  Support {i}: "
                    f"bar {support.bar_index} at s={support.s}, "
                    f"position=({support.x}, {support.y}), "
                    f"dof_in_x={support.dof_in_x}, "
                    f"dof_in_y={support.dof_in_y}"
                )

        print("\nInitial state:")
        for i, bar in enumerate(self.bars):
            if bar.x0 is None or bar.y0 is None or bar.theta0 is None:
                print(f"  Bar {i}: Not set")
            else:
                print(
                    f"  Bar {i}: "
                    f"x={bar.x0}, y={bar.y0}, theta={bar.theta0}, "
                    f"xd={bar.xd0}, yd={bar.yd0}, thetad={bar.thetad0}"
                )
        
    def _build_state_variables(self):
        q_names = []
        qd_names = []

        for i in range(len(self.bars)):
            q_names.extend([f"x{i}", f"y{i}", f"theta{i}"])
            qd_names.extend([f"xd{i}", f"yd{i}", f"thetad{i}"])

        self.q = list(hy.make_vars(*q_names))
        self.qd = list(hy.make_vars(*qd_names))

    def _build_lagrangian(self):
        kinetic = 0.0
        potential = 0.0

        for i, bar in enumerate(self.bars):
            x, y, theta = self.q[3 * i : 3 * i + 3]
            xd, yd, thetad = self.qd[3 * i : 3 * i + 3]

            kinetic += 0.5 * bar.mass * (xd**2 + yd**2)
            kinetic += 0.5 * bar.inertia * thetad**2
            potential += bar.mass * self.g * y

        self.lagrangian = kinetic - potential

    def _expr_point_on_bar(self, bar_index, s):
        bar = self.bars[bar_index]
        x, y, theta = self.q[3 * bar_index : 3 * bar_index + 3]

        px = x + s * bar.length * hy.cos(theta)
        py = y + s * bar.length * hy.sin(theta)

        return px, py

    def _initial_point_on_bar(self, bar_index, s):
        bar = self.bars[bar_index]
        px = bar.x0 + s * bar.length * np.cos(bar.theta0)
        py = bar.y0 + s * bar.length * np.sin(bar.theta0)
        return px, py

    def _current_point_on_bar_from_state(self, state, bar_index, s):
        bar = self.bars[bar_index]
        x, y, theta = state[3 * bar_index : 3 * bar_index + 3]

        px = x + s * bar.length * np.cos(theta)
        py = y + s * bar.length * np.sin(theta)

        return px, py

    def _build_constraints(self):
        constraints = []

        # Joints
        for joint in self.joints:
            ref_bar = joint.bar_index[0]
            ref_s = joint.s[0]
            ref_x, ref_y = self._expr_point_on_bar(ref_bar, ref_s)

            # Every other point in the joint must coincide with the first point.
            for bar_index, s in zip(joint.bar_index[1:], joint.s[1:]):
                px, py = self._expr_point_on_bar(bar_index, s)
                constraints.append(px - ref_x)
                constraints.append(py - ref_y)

        # Supports
        for support in self.supports:
            px, py = self._expr_point_on_bar(support.bar_index, support.s)

            if not support.dof_in_x:
                constraints.append(px - support.x)

            if not support.dof_in_y:
                constraints.append(py - support.y)

        self.F = np.asarray(constraints, dtype=object)

    def _build_external_forces(self):
        Q_nc = [0.0] * len(self.q)

        for force in self.external_forces:
            i = force.bar_index
            s = force.s
            bar = self.bars[i]

            x, y, theta = self.q[3 * i : 3 * i + 3]

            fx = force.fx
            fy = force.fy

            Q_nc[3 * i] += fx
            Q_nc[3 * i + 1] += fy

            Q_nc[3 * i + 2] += (
                -s * bar.length * hy.sin(theta) * fx
                + s * bar.length * hy.cos(theta) * fy
                + force.torque
            )

        self.Q_nc = np.asarray(Q_nc, dtype=object)
