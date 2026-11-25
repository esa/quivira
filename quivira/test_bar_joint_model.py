"""Tests for bar-joint model state management."""

import unittest as _ut

import heyoka as hy
import quivira as qv


class BarJointModelStateTests(_ut.TestCase):
    """Protect state completeness and constrained-state workflows."""

    def test_build_lagrange_eom_without_bar_poses(self):
        model = qv.bar_joint_model()
        model.add_bar(length=1.0, mass_per_length=1.0)

        ode_system = model.build_lagrange_eom()

        self.assertEqual(len(ode_system), 6)

    def test_get_state_requires_every_bar_pose(self):
        model = qv.bar_joint_model()
        model.add_bar(length=1.0, mass_per_length=1.0)
        model.add_bar(length=1.0, mass_per_length=1.0)
        model.set_bar_pose(0, 1.0, 2.0, 0.5)

        with self.assertRaisesRegex(ValueError, r"missing for bars \[1\]"):
            model.get_state()

    def test_pose_only_state_has_zero_velocities(self):
        model = qv.bar_joint_model()
        model.add_bar(length=1.0, mass_per_length=1.0)
        model.set_bar_pose(0, 1.0, 2.0, 0.5)

        self.assertEqual(model.get_state(), [1.0, 2.0, 0.5, 0, 0, 0])

    def test_set_bar_state_and_set_state_layouts(self):
        model = qv.bar_joint_model()
        model.add_bar(length=1.0, mass_per_length=1.0)
        model.add_bar(length=1.0, mass_per_length=1.0)

        model.set_bar_state(0, [1.0, 2.0, 0.5, 0.1, 0.2, 0.3])
        model.set_bar_state(1, [4.0, 5.0, 0.6, 0.4, 0.5, 0.6])

        expected = [1.0, 2.0, 0.5, 4.0, 5.0, 0.6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.assertEqual(model.get_state(), expected)

        flattened = [2.0, 3.0, 0.7, 5.0, 6.0, 0.8, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        model.set_state(flattened)
        self.assertEqual(model.get_state(), flattened)

        per_bar = [
            [3.0, 4.0, 0.9, 0.3, 0.4, 0.5],
            [6.0, 7.0, 1.0, 0.6, 0.7, 0.8],
        ]
        model.set_state(per_bar)
        expected = [3.0, 4.0, 0.9, 6.0, 7.0, 1.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.assertEqual(model.get_state(), expected)

    def test_set_state_validates_parameterized_constraints(self):
        model = qv.bar_joint_model()
        model.add_bar(length=1.0, mass_per_length=1.0)
        model.add_support(0, 0.0, False, False, x=hy.par[0], y=0.0)

        model.set_state([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], pars=[1.0])

        with self.assertRaisesRegex(ValueError, "State violates"):
            model.set_state([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], pars=[1.0])

    def test_fix_state_still_projects_constraint_violations(self):
        model = qv.bar_joint_model()
        model.add_bar(length=1.0, mass_per_length=1.0)
        model.add_support(0, 0.0, False, False, x=0.0, y=0.0)
        model.build_lagrange_eom()
        broken_state = [0.1, -0.2, 0.0, 0.4, 0.5, 0.6]

        fixed_state = model.fix_state(broken_state, set_velocities_zero=True)

        self.assertTrue(model.check_state(fixed_state))