import numpy as np
import heyoka as hy
import quivira as qv

class bar:
    x0 = float
    y0 = float
    theta0 = float
    length = float
    mass = float
    inertia = float

    def __init__(self, length, mass):
        self.x0 = None
        self.y0 = None
        self.theta0 = None
        self.length = length
        self.mass = mass
        self.inertia = self.mass * self.length**2 / 12.0

    @classmethod
    def from_point_pair(cls, start, end, mass_per_length=1.0):
        start_x, start_y = start
        end_x, end_y = end

        length = np.sqrt((end_x - start_x)**2 +(end_y - start_y)**2)
        theta0 = np.arctan2(end_y - start_y, end_x - start_x)

        mass = mass_per_length * length
        x0 = (start_x + end_x) / 2.0
        y0 = (start_y + end_y) / 2.0

        new_bar = cls(length, mass)
        new_bar.position_bar(x0, y0, theta0)

        return new_bar

    def position_bar(self, x0, y0, theta0):
        self.x0 = x0
        self.y0 = y0
        self.theta0 = theta0

class joint:
    # u, v, are symbolic expressions for the point on the bar in local coordinates
    bar_index = []
    s = []

    def __init__(self, bar_index, s):
        if len(bar_index) != len(s):
            raise ValueError("bar_index and s must have the same length")

        if len(bar_index) < 2:
            raise ValueError("A joint must connect at least two bars")
    
        self.bar_index = bar_index
        self.s = s

class support:
    # u, v, are symbolic expressions for the point on the bar in local coordinates
    bar_index = int
    s = float
    x = float
    y = float
    dof_in_x = bool
    dof_in_y = bool

    def __init__(self, bar_index, s, x, y, dof_in_x=False, dof_in_y=False):
        self.bar_index = bar_index
        self.s = s
        self.x = x
        self.y = y
        self.dof_in_x = dof_in_x
        self.dof_in_y = dof_in_y

class external_force:
    bar_index = int
    s = float
    fx = float
    fy = float

    def __init__(self, bar_index, s, fx, fy):
        self.bar_index = bar_index
        self.s = s
        self.fx = fx
        self.fy = fy

class bar_joint_model:
    def __init__(self, tolerance = 1e-10):
        self.tolerance = tolerance 
        self.clear()

    def clear(self):
        self.g = 0.0
        self.bars = []
        self.joints = []
        self.supports = []
        self.external_forces = []

        self.q = []
        self.qd = []

        self.lagrangian = None
        self.F = np.asarray([], dtype=object)
        self.state0 = []

        self.ode_system = None
        self.lambda_expr = None
        self.ta = None # Taylor adaptive integrator

        self.solution = None

    def set_g(self, g):
        self.g = float(g)

    # def build_from_point_pairs(self, point_pairs):
    #     self.clear()

    #     for start, end in point_pairs:
    #         self.add_bar_from_point_pair(start, end)

    def add_bar(self, length, mass):
        self.bars.append(bar(length, mass))

    def position_bar(self, bar_index, x0, y0, theta0):
        self.bars[bar_index].position_bar(x0, y0, theta0)

    def add_bar_at_attitude(self, x0, y0, theta0, length, mass):
        self.add_bar(length, mass)
        self.position_bar(-1, x0, y0, theta0)

    def add_bar_from_point_pair(self, start, end, mass_per_length=1.0):
        self.bars.append(bar.from_point_pair(start, end, mass_per_length))

    def add_joint(self, bar_index, s):
        self.joints.append(joint(bar_index, s))

    def add_support(self, bar_index, s, dof_in_x, dof_in_y, x=None, y=None):
        if x == None or y == None:
            if self.bars[bar_index].x0 == None or self.bars[bar_index].y0 == None:
                raise ValueError("Bar " + str(bar_index) + " does not have an initial state yet, can't set support")
        
            x, y = self._initial_point_on_bar(bar_index, s)

        self.supports.append(support(bar_index, s, x, y, dof_in_x, dof_in_y))

    def add_external_force(self, bar_index, s, fx, fy):
        self.external_forces.append(external_force(bar_index, s, fx, fy))

    def build_eom_integrator(self):
        self._build_state_variables()
        self._build_lagrangian()
        self._build_constraints()
        self._build_external_forces()
        self.ode_system, self.lambda_expr = qv.lagrange_eom(self.lagrangian, self.F, self.q, self.qd, return_multipliers=True)
        self.ta = hy.taylor_adaptive(self.ode_system)

    def set_initial_state(self, initial_states):
        if len(initial_states) != len(self.bars):
            raise ValueError(f"Expected initial states for {len(self.bars)} bars, got {len(initial_states)}")

        for i, state in enumerate(initial_states):
            if len(state) != 3:raise ValueError(f"Initial state for bar {i} must contain [x0, y0, theta0]")

            x0, y0, theta0 = state

            self.bars[i].x0 = float(x0)
            self.bars[i].y0 = float(y0)
            self.bars[i].theta0 = float(theta0)

        self._check_initial_states_against_constraints()
        self._apply_initial_state()

    def integrate(self, no_steps, end_time, start_time = 0.0):
        self.ta.state[:] = self.state0
        self.ta.time = start_time
        tgrid = np.linspace(start_time, end_time, no_steps)
        self.solution = self.ta.propagate_grid(tgrid)[-1]
        return self.solution
        
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
        Q = [0.0] * len(self.q)

        for force in self.external_forces:
            i = force.bar_index
            s = force.s
            bar = self.bars[i]

            x, y, theta = self.q[3 * i : 3 * i + 3]

            fx = force.fx
            fy = force.fy

            Q[3 * i] += fx
            Q[3 * i + 1] += fy

            Q[3 * i + 2] += (
                -s * bar.length * hy.sin(theta) * fx
                + s * bar.length * hy.cos(theta) * fy
            )

        self.external_forces_expr = np.asarray(Q, dtype=object)

    def _check_initial_states_against_constraints(self):
        # Check joints
        for joint_index, joint in enumerate(self.joints):
            ref_bar = joint.bar_index[0]
            ref_s = joint.s[0]

            ref_x, ref_y = self._initial_point_on_bar(ref_bar,ref_s)

            for bar_index, s in zip(joint.bar_index[1:],joint.s[1:]):
                px, py = self._initial_point_on_bar(bar_index,s)

                if (abs(px - ref_x) > self.tolerance or abs(py - ref_y) > self.tolerance):
                    raise ValueError(f"Initial state violates joint {joint_index}: bar {bar_index} does not coincide with bar {ref_bar}")

        # Check supports
        for support_index, support in enumerate(self.supports):
            px, py = self._initial_point_on_bar(support.bar_index, support.s)

            if (not support.dof_in_x and abs(px - support.x) > self.tolerance):
                raise ValueError(f"Initial state violates support {support_index} in x: got {px}, expected {support.x}")

            if (not support.dof_in_y and abs(py - support.y) > self.tolerance):
                raise ValueError(f"Initial state violates support {support_index} in y: got {py}, expected {support.y}")

    def _apply_initial_state(self):
        q0 = []
        for bar in self.bars:
            q0.extend([bar.x0, bar.y0, bar.theta0])

        qd0 = [0.0] * len(self.qd)
        self.state0 = q0 + qd0