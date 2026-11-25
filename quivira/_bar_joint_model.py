"""Bar-and-joint model for constrained rigid-link systems.

This module stores a compact set of bars, joints, supports, and point loads for
forming symbolic Lagrange equations of motion in a quivira model.
"""

import numpy as np
import heyoka as hy
import quivira as qv
from ._fix_state import (
            _compile_constraints,
            parameter_values,
            fix_configuration,
            fix_velocity,
        )
from ._lagrangian_math import build_mass_matrix

class bar:
    """Rigid bar segment with mass, geometric length, and initial pose.

    The bar stores the center position, orientation, and inertial properties that
    are used to assemble the generalized coordinates of the model.
    """

    def __init__(self, length, mass):
        """Initialize the bar geometry and inertial properties.

        Args:
            length (:class:`float` or :class:`heyoka.expression`): Bar length.

            mass (:class:`float` or :class:`heyoka.expression`): Bar mass.

        Notes:
            Heyoka expressions are preserved in the bar inertia and the symbolic
            equations of motion.
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
        new_bar.set_bar_pose(x0, y0, theta0)

        return new_bar

    def set_bar_pose(self, x0, y0, theta0):
        """Set the center position and angle of the bar.

        Args:
            x0 (:class:`float`): Center x coordinate.

            y0 (:class:`float`): Center y coordinate.

            theta0 (:class:`float`): Orientation angle.
        """
        self.x0 = x0
        self.y0 = y0
        self.theta0 = theta0

    def set_bar_velocities(self, xd0, yd0, thetad0):
        """Set the translational and angular velocities of the bar.

        Args:
            xd0 (:class:`float`): x velocity of the center.

            yd0 (:class:`float`): y velocity of the center.

            thetad0 (:class:`float`): Angular velocity.
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

            torque (:class:`float`, optional): Applied moment about the bar
                center, positive counterclockwise. Default is ``0``.
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
        self.g = 0.0
        self.bars = []
        self.joints = []
        self.supports = []
        self.external_forces = []

        self.q = []
        self.qd = []

        self.lagrangian = None
        self.mass_matrix = None
        self.F = np.asarray([], dtype=object)
        self.Q_nc = np.asarray([], dtype=object)

    def add_bar(self, length, mass_per_length):
        """Append a new bar to the assembly.

        Args:
            length (:class:`float` or :class:`heyoka.expression`): Bar length.

            mass_per_length (:class:`float` or :class:`heyoka.expression`):
                Mass per unit length.
        """
        mass = mass_per_length * length
        self.bars.append(bar(length, mass))

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
        """Infer joints from intersections and endpoint contacts in initial poses.

        Detects finite-segment intersections and collinear endpoint contacts,
        groups bars meeting at the same point, and merges any existing joints
        to avoid duplicates. Collinear overlap over a nonzero length is
        ambiguous and raises an error.

        Raises:
            ValueError: If a bar lacks an initial attitude, has a non-finite or
                near-zero length, has a conflicting existing joint position, or
                overlaps another bar along a segment.

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
                initial point position is used. If either coordinate is omitted,
                both coordinates are taken from the initial bar pose. Default is
                ``None``.

            y (:class:`float`, optional): Support y position. If omitted, the
                initial point position is used. If either coordinate is omitted,
                both coordinates are taken from the initial bar pose. Default is
                ``None``.

        Raises:
            ValueError: If either support coordinate is omitted and the bar does
                not have an initial state from which to compute the point.
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

            torque (:class:`float`, optional): Applied moment about the bar
                center, positive counterclockwise. Default is ``0.0``.
        """
        self.external_forces.append(external_force(bar_index, s, fx, fy, torque))

    def set_bar_pose(self, bar_index, x0, y0, theta0):
        """Set the position and orientation of a bar.

        Args:
            bar_index (:class:`int`): Index of the target bar.

            x0 (:class:`float`): x coordinate of the bar center.

            y0 (:class:`float`): y coordinate of the bar center.

            theta0 (:class:`float`): Angle of the bar.
        """
        self.bars[bar_index].set_bar_pose(x0, y0, theta0)

    def set_bar_velocities(self, bar_index, xd0, yd0, thetad0):
        """Set the translational and angular velocities of a bar.

        Args:
            bar_index (:class:`int`): Index of the target bar.

            xd0 (:class:`float`): x velocity of the center.

            yd0 (:class:`float`): y velocity of the center.

            thetad0 (:class:`float`): Angular velocity.
        """
        self.bars[bar_index].set_bar_velocities(xd0, yd0, thetad0)

    def set_bar_state(self, bar_index, state):
        """Set one bar's position, orientation, and velocities.

        Args:
            bar_index (:class:`int`): Index of the target bar.

            state (:class:`list` or :class:`numpy.ndarray`): Six values in the
                order ``[x, y, theta, xd, yd, thetad]``.

        Raises:
            ValueError: If the state does not contain six values.
        """
        bar_state = np.asarray(state, dtype=float).ravel()
        if bar_state.size != 6:
            raise ValueError(
                f"State for bar {bar_index} must contain [x, y, theta, xd, yd, thetad]"
            )

        bar = self.bars[bar_index]
        bar.x0, bar.y0, bar.theta0 = bar_state[:3]
        bar.xd0, bar.yd0, bar.thetad0 = bar_state[3:]

    def set_state(self, state, pars=None):
        """Set bar poses and velocities from a flattened or per-bar state.

        Args:
            state (:class:`list` or :class:`numpy.ndarray`): Either a flattened
                state ordered as in :meth:`get_state`, or a
                sequence with one six-value state
                ``[x0, y0, theta0, xd0, yd0, thetad0]`` per bar.

            pars (:class:`list` or :class:`numpy.ndarray`, optional): Values for
                Heyoka parameters appearing in the constraints. Default is
                ``None``.

        Raises:
            ValueError: If the input shape is invalid or the assigned state
                violates a position or velocity constraint.
        """
        state_array = np.asarray(state, dtype=float)
        n_bars = len(self.bars)

        if state_array.shape == (n_bars, 6):
            for bar_index, bar_state in enumerate(state_array):
                self.set_bar_state(bar_index, bar_state)
        elif state_array.ndim == 1 and state_array.size == 6 * n_bars:
            q_size = 3 * n_bars
            for bar_index in range(n_bars):
                q_offset = 3 * bar_index
                qd_offset = q_size + 3 * bar_index
                bar_state = np.concatenate((
                    state_array[q_offset : q_offset + 3],
                    state_array[qd_offset : qd_offset + 3],
                ))
                self.set_bar_state(bar_index, bar_state)
        else:
            raise ValueError(
                f"Expected state shape ({n_bars}, 6) or length {6 * n_bars}, "
                f"got {state_array.shape}"
            )

        if not self.check_state(self.get_state(), pars=pars):
            raise ValueError("State violates position or velocity constraints")

    def get_state(self):
        """Return the stacked generalized coordinates and velocities.

        Returns:
            list: State vector containing ``[x0, y0, theta0, xd0, yd0, thetad0]``
            for each bar.

        Raises:
            ValueError: If any bar does not have a pose.
        """
        missing_bars = [
            i for i, bar in enumerate(self.bars)
            if any(value is None for value in (bar.x0, bar.y0, bar.theta0))
        ]
        if missing_bars:
            raise ValueError(
                f"Bar poses must be set before retrieving the state; missing for bars {missing_bars}"
            )

        q0 = []
        for bar in self.bars:
            q0.extend([bar.x0, bar.y0, bar.theta0])

        qd0 = []
        for bar in self.bars:
            qd0.extend([bar.xd0, bar.yd0, bar.thetad0])
        return q0 + qd0

    def check_state(self, state, pars=None, verbose=False):
        """Check whether a state satisfies its position and velocity constraints.

        The state is split into generalized coordinates followed by generalized
        velocities. Both the position residuals and the velocity residuals
        :math:`J(q)\\dot q` must be finite and no larger than ``self.tolerance``
        in absolute value. Heyoka parameters used by the constraints can be
        supplied through ``pars``.

        Args:
            state (:class:`list` or :class:`numpy.ndarray`): Coordinates followed
                by velocities, in the ordering returned by
                :meth:`get_state`.

            pars (:class:`list` or :class:`numpy.ndarray`, optional): Values for
                Heyoka parameters appearing in the constraints. Default is
                ``None``.

            verbose (:class:`bool`, optional): If ``True``, print whether each
                position and velocity constraint is fulfilled. Default is
                ``False``.

        Returns:
            :class:`bool`: ``True`` if all position and velocity constraints
            are fulfilled, otherwise ``False``.
        """

        nq = 3 * len(self.bars)

        position = state[:nq]
        velocity = state[nq:]

        # Build the symbolic coordinates and constraints
        # This also works before build_lagrange_eom() is called
        if len(self.q) != nq:
            self._build_state_variables()

        self._build_constraints()

        F = list(self.F)

        if not F:
            return True

        # Compile the constraints and their Jacobian
        # The required heyoka parameters are selected internally
        constraints, jacobian = _compile_constraints(self.q, F, pars)

        # Evaluate the position and velocity constraints
        position_residuals = constraints(position)
        velocity_residuals = jacobian(position) @ velocity

        fulfilled_all = True

        for i, residual in enumerate(position_residuals):
            fulfilled = np.isfinite(residual) and abs(residual) <= self.tolerance
            fulfilled_all = fulfilled_all and fulfilled
            if verbose:
                status = "fulfilled" if fulfilled else "not fulfilled"
                print(f"Position constraint {i}: {status} (residual = {residual})")

        for i, residual in enumerate(velocity_residuals):
            fulfilled = np.isfinite(residual) and abs(residual) <= self.tolerance
            fulfilled_all = fulfilled_all and fulfilled
            if verbose:
                status = "fulfilled" if fulfilled else "not fulfilled"
                print(f"Velocity constraint {i}: {status} (residual = {residual})")

        return bool(fulfilled_all)

    def fix_state(
        self,
        state,
        pars=None,
        set_velocities_zero=False,
        use_exact_endpoints=False,
        tolerance=None,
        maxiter=1000,
    ):
        """Project a bar-joint state onto its position and velocity constraints.

        Positions are corrected by minimizing a displacement objective, using
        either a quadratic small-angle approximation or exact endpoint
        displacement. Unless velocities are requested to be zero, they are
        corrected using the generalized mass matrix.

        Args:
            state (:class:`numpy.ndarray`): Coordinates followed by velocities,
                in the ordering returned by :meth:`get_state`.

            pars (:class:`list` or :class:`numpy.ndarray`, optional): Values for
                Heyoka parameters appearing in the constraints or bar lengths.
                Default is ``None``.

            set_velocities_zero (:class:`bool`, optional): If ``True``, return
                zero velocities instead of projecting the supplied velocities.
                Default is ``False``.

            use_exact_endpoints (:class:`bool`, optional): If ``True``, use the
                exact endpoint-displacement objective for position correction;
                otherwise use its quadratic small-angle approximation. Default
                is ``False``.

            tolerance (:class:`float`, optional): Constraint tolerance for
                checking and correction. If omitted, ``self.tolerance`` is used.
                Default is ``None``.

            maxiter (:class:`int`, optional): Maximum iterations for the
                position and velocity correction routines. Default is ``1000``.

        Raises:
            ValueError: If the tolerance is not positive and finite, the model
                has not been built, or the bar lengths are not positive and
                finite.

        Returns:
            :class:`numpy.ndarray`: Corrected coordinates followed by corrected
            (or zero) velocities.
        """

        if tolerance is None:
            tolerance = self.tolerance

        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")

        nq = 3 * len(self.bars)

        if not self.bars or len(self.q) != nq or self.lagrangian is None:
            raise ValueError("Build the model before calling fix_state")
        
        q_initial = state[:nq]
        qdot_initial = state[nq:]

        F = list(self.F)

        # Check position and velocity constraints once
        position_valid = True
        velocity_valid = True

        if F:
            constraints, jacobian = _compile_constraints(self.q, F, pars)

            position_error = np.max(np.abs(constraints(q_initial)))
            velocity_error = np.max(
                np.abs(jacobian(q_initial) @ qdot_initial)
            )

            position_valid = (
                np.isfinite(position_error) and position_error <= tolerance
            )
            velocity_valid = (
                np.isfinite(velocity_error) and velocity_error <= tolerance
            )

        # If the state is valid, no correction is necessary
        if position_valid and velocity_valid:
            if set_velocities_zero:
                return np.concatenate((q_initial, np.zeros(nq)))
            return state.copy()

        # Correct the configuration if necessary
        if position_valid:
            q_corrected = q_initial.copy()

        else:
            # Evaluate the bar lengths for the endpoint objective
            length_expressions = []

            for bar in self.bars:
                length = bar.length

                if not isinstance(length, hy.expression):
                    length = hy.expression(float(length))

                length_expressions.append(length)

            length_func = hy.cfunc(length_expressions, vars=[], compact_mode=True)
            length_pars = parameter_values(length_expressions, pars)
            lengths = np.asarray(length_func([], pars=length_pars), dtype=float)

            if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
                raise ValueError("Bar lengths must be positive and finite")

            if use_exact_endpoints:
                # Exact endpoint-displacement objective
                def objective(candidate, reference):
                    dq = candidate - reference
                    translation = np.sum(dq[0::3]**2 + dq[1::3]**2)
                    rotation = 0.5 * np.sum(lengths**2 * (1.0 - np.cos(dq[2::3])))
                    return float(translation + rotation)

                def gradient(candidate, reference):
                    dq = candidate - reference
                    result = np.zeros_like(dq)

                    result[0::3] = 2.0 * dq[0::3]
                    result[1::3] = 2.0 * dq[1::3]
                    result[2::3] = (0.5 * lengths**2 * np.sin(dq[2::3]))

                    return result

            else:
                # Quadratic small-angle approximation
                R_diag = np.full(nq, 2.0)
                R_diag[2::3] = lengths**2 / 2.0

                def objective(candidate, reference):
                    dq = candidate - reference
                    return 0.5 * np.dot(R_diag * dq, dq)

                def gradient(candidate, reference):
                    return R_diag * (candidate - reference)

            q_corrected = fix_configuration(
                q_initial,
                self.q,
                F,
                pars=pars,
                objective=objective,
                gradient=gradient,
                tolerance=tolerance,
                maxiter=maxiter,
            )

        # Zero the velocities if requested
        if set_velocities_zero:
            return np.concatenate((q_corrected, np.zeros(nq)))

        # Use the mass matrix already constructed during ODE generation
        if self.mass_matrix is None:
            # Fallback if no mass matrix has been stored
            self.mass_matrix = build_mass_matrix(self.lagrangian, self.qd)

        # To make it a cfunc, we expect a list
        M_expressions = self.mass_matrix.ravel().tolist()
        # Compile and evaluate the mass matrix
        mass_func = hy.cfunc(M_expressions, vars=self.q, compact_mode=True)
        mass_pars = parameter_values(M_expressions, pars)
        # Now we can make it a matrix again
        M = np.asarray(mass_func(q_corrected, pars=mass_pars),dtype=float).reshape(nq, nq)

        # Correct the velocities using the generalized mass matrix
        qdot_corrected = fix_velocity(
            q_corrected,
            qdot_initial,
            self.q,
            F,
            mass_matrix=M,
            pars=pars,
            tolerance=tolerance,
            maxiter=maxiter,
        )

        return np.concatenate((q_corrected, qdot_corrected))

    def build_lagrange_eom(self, return_multipliers=False, return_W=False, verbose=False):
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

                verbose (:class:`bool`, optional): If ``True``, print the elapsed
                    time for each equation-assembly step. Default is ``False``.

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
        result = qv.lagrange_eom(self.lagrangian, self.F, self.q, self.qd, self.Q_nc, return_multipliers=return_multipliers, return_W=return_W, return_M=True, verbose=verbose)
        # Save the symbolic mass matrix
        self.mass_matrix = result[-1]
        # Now return only the return values of build_lagrange_eom(), not the mass matrix
        original_results = result[:-1]
        if len(original_results) == 1:
            return original_results[0]

        return original_results

# Human-readable model summary
    def __repr__(self):
        """Return a human-readable summary of the assembled model."""
        lines = ["Bars:"]
        for i, bar in enumerate(self.bars):
            lines.append(f"  Bar {i}: length={bar.length}, mass={bar.mass}")

        lines.append("\nJoints:")
        if len(self.joints) == 0:
            lines.append("  None")
        else:
            for i, joint in enumerate(self.joints):
                connections = [f"bar {bar_index} at s={s}" for bar_index, s in zip(joint.bar_index, joint.s)]
                lines.append(f"  Joint {i}: " + ", ".join(connections))

        lines.append("\nSupports:")
        if len(self.supports) == 0:
            lines.append("  None")
        else:
            for i, support in enumerate(self.supports):
                lines.append(
                    f"  Support {i}: "
                    f"bar {support.bar_index} at s={support.s}, "
                    f"position=({support.x}, {support.y}), "
                    f"dof_in_x={support.dof_in_x}, "
                    f"dof_in_y={support.dof_in_y}"
                )

        lines.append("\nInitial state:")
        for i, bar in enumerate(self.bars):
            if bar.x0 is None or bar.y0 is None or bar.theta0 is None:
                lines.append(f"  Bar {i}: Not set")
            else:
                lines.append(
                    f"  Bar {i}: "
                    f"x={bar.x0}, y={bar.y0}, theta={bar.theta0}, "
                    f"xd={bar.xd0}, yd={bar.yd0}, thetad={bar.thetad0}"
                )

        return "\n".join(lines)

# Private methods for internal use within the model
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
