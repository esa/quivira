"""Regression tests for the constrained-state projection helpers."""

import unittest as _ut

import heyoka as hy
import numpy as np

from ._fix_state import fix_configuration, fix_velocity


class FixStateTests(_ut.TestCase):
    """Protect the validity of the configuration and velocity projections."""

    def test_fix_configuration_projects_to_constraint_surface(self):
        """A point off the unit circle is projected back onto the constraint manifold.

        This is the basic holonomic-constraint case: the state is invalid at first, and the
        function must return a nearby valid point. It is representative because it checks the
        core projection behavior of ``fix_configuration``.
        """
        x, y = hy.make_vars("x", "y")
        q = [x, y]
        F = [x**2 + y**2 - 1.0]
        q_invalid = np.array([0.2, 0.2], dtype=float)

        corrected = fix_configuration(q_invalid, q, F)

        self.assertTrue(np.all(np.isfinite(corrected)))
        np.testing.assert_allclose(corrected[0] ** 2 + corrected[1] ** 2, 1.0, atol=1e-8)
        self.assertLessEqual(np.linalg.norm(corrected - q_invalid), np.linalg.norm(np.array([1.0, 0.0]) - q_invalid) + 1e-8)

    def test_fix_velocity_maintains_tangency_and_energy(self):
        """A velocity is corrected to stay tangent to the constraint and keep the same energy.

        This checks the main constrained-velocity use case: the input violates the tangent-space
        condition, and the corrected velocity must satisfy it without changing the kinetic energy.
        It is representative because it exercises both the geometric and energetic constraints.
        """
        x, y = hy.make_vars("x", "y")
        q = [x, y]
        F = [x**2 + y**2 - 1.0]
        q_valid = np.array([1.0, 0.0], dtype=float)
        qdot_invalid = np.array([0.5, 1.0], dtype=float)
        mass_matrix = np.diag([2.0, 3.0])

        corrected = fix_velocity(q_valid, qdot_invalid, q, F, mass_matrix)

        jacobian = np.array([[2.0 * q_valid[0], 2.0 * q_valid[1]]], dtype=float)
        np.testing.assert_allclose(jacobian @ corrected, 0.0, atol=1e-8)
        original_energy = 0.5 * qdot_invalid @ mass_matrix @ qdot_invalid
        corrected_energy = 0.5 * corrected @ mass_matrix @ corrected
        self.assertAlmostEqual(corrected_energy, original_energy, places=8)
        self.assertAlmostEqual(corrected[0], 0.0, places=8)
        self.assertGreater(corrected[1], 0.0)

    def test_fix_velocity_leaves_unconstrained_velocity_unchanged(self):
        """An unconstrained velocity is left unchanged.

        This is a boundary case: if there are no geometric constraints, the step should do
        nothing. It is representative because it protects the no-constraint branch of
        ``fix_velocity`` from accidental changes.
        """
        x, y = hy.make_vars("x", "y")
        q = [x, y]
        q_valid = np.array([2.0, -1.0], dtype=float)
        qdot_invalid = np.array([0.7, -0.4], dtype=float)
        mass_matrix = np.diag([1.0, 2.0])

        corrected = fix_velocity(q_valid, qdot_invalid, q, [], mass_matrix)

        np.testing.assert_allclose(corrected, qdot_invalid, atol=1e-10)
