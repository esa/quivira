"""Generic bar-hinge model data container."""

import heyoka as hy

class bar_hinge_model:
    """Store bar lengths, masses, and initial conditions.

    The lists are paired by index: each bar length and mass share the same
    position in the corresponding arrays. The initial conditions remain empty by
    default and can be filled later as the model is extended.
    """

    def __init__(self):
        """Initialize the model with empty bar data."""
        self.bar_lengths = []
        self.bar_masses = []
        self.q = []
        self.qd = []

    def make_state_variables(self, no_bars: int):
        """Create the generalized coordinates and velocities for all bars.

        Args:
            no_bars (:class:`int`): Number of bars in the chain.

        Returns:
            tuple: The coordinate list ``q`` and velocity list ``qd``.
        """
        # The state is ordered by bar and component:
        # q = [x0, y0, theta0, x1, y1, theta1, ...]
        # qd = [xd0, yd0, thetad0, xd1, yd1, thetad1, ...]
        q_names = []
        qd_names = []

        for i in range(no_bars):
            q_names += [f"x{i}", f"y{i}", f"theta{i}"]
            qd_names += [f"xd{i}", f"yd{i}", f"thetad{i}"]

        self.q = list(hy.make_vars(*q_names))
        self.qd = list(hy.make_vars(*qd_names))

        return self.q, self.qd

    def build_bar_hinge_model(self, no_bars, g=9.81):

        q, qd = self.make_state_variables(no_bars)

        # Assemble the free Lagrangian before differentiating the equations of
        # motion for the unconstrained system.
        T = 0.0
        V = 0.0

        for i in range(no_bars):
            # Each bar contributes translational and rotational kinetic energy
            # as well as the gravitational potential associated with its center.
            x, y, theta = q[3 * i : 3 * i + 3]
            xd, yd, thetad = qd[3 * i : 3 * i + 3]

            T += 0.5 * self.bar_masses[i] * (xd**2 + yd**2)
            T += 0.5 * self.bar_masses[i] * (self.bar_lengths[i] ** 2) * (thetad**2) / 12
            V += self.bar_masses[i] * g * y

        Lagrangian = T - V

        # Heyoka returns the Euler-Lagrange equations in the same order as the
        # generalized velocity variables in qd.
        free_system = hy.lagrangian(Lagrangian, q, qd)

        # Keep only the acceleration expressions associated with the velocity
        # variables, matching the ordering in qd.
        qdd_free = []
        for variable, rhs in free_system:
            if variable in qd:
                qdd_free.append(rhs)

    def build_lattice_initial_state_from_points(self, point_pairs, mass_per_length, g=9.81):
        """Build bar geometry and the hinge-constraint Jacobian from point pairs.

        Args:
            point_pairs (sequence): Each entry contains a start and end point for
                one bar, represented as ``(start, end)`` with each point being a
                2D coordinate pair ``(x, y)``.
            mass_per_length (:class:`float`): Mass assigned per unit bar length.
            g (:class:`float`, optional): Gravity constant used in the Lagrangian
                when the model is assembled. Default is 9.81.

        Returns:
            :class:`numpy.ndarray`: The constraint Jacobian ``J`` from the
                holonomic hinge constraints.
        """
        self.bar_lengths = []
        self.bar_masses = []

        for start, end in point_pairs:
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = (dx * dx + dy * dy) ** 0.5
            if length == 0.0:
                raise ValueError("bar length must be nonzero")

            self.bar_lengths.append(length)
            self.bar_masses.append(mass_per_length * length)

        self.make_state_variables(len(point_pairs))
        J = self.build_constraint_jacobian(point_pairs)

        return J

    def build_constraint_jacobian(self, point_pairs):
        return 0
        
        

