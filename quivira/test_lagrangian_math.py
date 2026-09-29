"""Tests for Lagrangian mechanics helper functions."""

import unittest as _ut

import numpy as np
import sympy as sp
import heyoka as hy

import quivira as qv


class LagrangianMathTests(_ut.TestCase):
    """Tests for the Lagrangian-math utilities."""

    def test_build_mass_matrix(self):
        """Return the Hessian of the Lagrangian with respect to the generalized velocities.

        The expected entries are computed by taking the second partial derivative
        M_ij = d^2 L / (d qd_i d qd_j). This is useful because the mass matrix is
        the coefficient matrix that appears in the kinetic-energy part of the
        equations of motion.
        """
        q0 = hy.make_vars("q0")
        q1 = hy.make_vars("q1")
        v0 = hy.make_vars("v0")
        v1 = hy.make_vars("v1")
        qd = [v0, v1]

        lagrangian = 0.5 * (v0**2 + 2.0 * v0 * v1 + 3.0 * v1**2)
        computed = qv.build_mass_matrix(lagrangian, qd)
        expected = np.array([[1.0, 1.0], [1.0, 3.0]], dtype=object)

        self.assertIsInstance(computed, np.ndarray)
        self.assertEqual(computed.shape, expected.shape)
        for row in range(2):
            for col in range(2):
                simplified = sp.simplify(hy.to_sympy(computed[row, col] - expected[row, col]))
                self.assertEqual(simplified, 0)

    def test_build_Jdqd(self):
        """Return the product Jdot qd by summing the coordinate-wise chain-rule terms.

        The expected value is computed as
        Jdot qd = sum_i qd_i * (dJ/dq_i) @ qd, with each row differentiated
        separately. This is useful because the acceleration constraint uses this
        term in J qdd + Jdot qd = 0.
        """
        q0 = hy.make_vars("q0")
        q1 = hy.make_vars("q1")
        v0 = hy.make_vars("v0")
        v1 = hy.make_vars("v1")
        q = [q0, q1]
        qd = [v0, v1]
        J = np.array([[q0, q1], [q1, q0]], dtype=object)

        expected = np.array([v0**2 + v1**2, 2.0 * v0 * v1], dtype=object)
        computed = qv.build_Jdqd(J, q, qd)

        self.assertEqual(computed.shape, expected.shape)
        for idx in range(2):
            simplified = sp.simplify(hy.to_sympy(computed[idx] - expected[idx]))
            self.assertEqual(simplified, 0)

    def test_find_constrained_accelerations(self):
        """Project the free acceleration into the constraint-consistent subspace.

        The expected value is obtained from the closed form
        qdd = qdd_free + M^{-1} J^T lambda with
        lambda = -W^{-1}(J qdd_free + Jdot qd), where W = J M^{-1} J^T.
        This is useful because it verifies the sign convention and the
        acceleration projection used by the constrained system.
        """
        qdd_free = np.array([3.0, 4.0], dtype=object)
        J = np.array([[1.0, 1.0]], dtype=object)
        M = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=object)
        Jdqd = np.array([1.0], dtype=object)

        constrained = qv.find_constrained_accelerations(qdd_free, J, M, Jdqd)
        expected = np.array([-1.8, 0.8], dtype=object)

        self.assertEqual(constrained.shape, expected.shape)
        for idx in range(2):
            residual = float(sp.N(hy.to_sympy(constrained[idx] - expected[idx])))
            self.assertAlmostEqual(residual, 0.0, places=12)

    def test_build_constraint_jacobian(self):
        """Return the Jacobian of the algebraic constraints with respect to the coordinates.

        The expected values are computed entry by entry as
        J_ij = dF_i / d q_j, which matches the definition of the constraint
        Jacobian. This is useful because the bond and hinge constraints are
        enforced through J qdd + Jdot qd = 0.
        """
        q0 = hy.make_vars("q0")
        q1 = hy.make_vars("q1")
        q = [q0, q1]
        F = [q0 + q1, q0**2 - q1]

        computed = qv.build_constraint_jacobian(F, q)
        expected = np.array([[1.0, 1.0], [2.0 * q0, -1.0]], dtype=object)

        self.assertEqual(computed.shape, expected.shape)
        for row in range(2):
            for col in range(2):
                simplified = sp.simplify(hy.to_sympy(computed[row, col] - expected[row, col]))
                self.assertEqual(simplified, 0)

    def test_build_ode_equations_of_motion(self):
        """Assemble the first-order ODE system from the unconstrained Lagrangian and constraints.

        The expected value is built by evaluating the Lagrangian with
        hy.lagrangian(...), selecting the acceleration equations for qd, and then
        pairing each coordinate with its velocity and each velocity with its
        acceleration. This is useful because it validates the conversion from the
        symbolic constrained dynamics into the form required by a time integrator.
        """
        q0 = hy.make_vars("q0")
        q1 = hy.make_vars("q1")
        v0 = hy.make_vars("v0")
        v1 = hy.make_vars("v1")

        q = [q0, q1]
        qd = [v0, v1]
        lagrangian = 0.5 * (v0**2 + v1**2)
        F = [q0 + q1]

        computed = qv.build_ode_equations_of_motion(lagrangian, F, q, qd)
        expected = [(q0, v0), (q1, v1), (v0, 0.0), (v1, 0.0)]

        self.assertEqual(len(computed), len(expected))
        for (lhs_actual, rhs_actual), (lhs_expected, rhs_expected) in zip(computed, expected):
            self.assertEqual(sp.simplify(hy.to_sympy(lhs_actual - lhs_expected)), 0)
            self.assertEqual(sp.simplify(hy.to_sympy(rhs_actual - rhs_expected)), 0)
