"""Tests for symbolic linear-algebra helpers."""

import unittest as _ut

import heyoka as hy
import numpy as np
import sympy as sp

import quivira as qv


class linear_algebra_tests(_ut.TestCase):
    """Tests for the public linear-algebra helpers."""

    def test_identity_passthrough(self):
        """We test that identity returns the exact same object."""
        sample_string = "Quivira"
        sample_integer = 42

        self.assertIs(qv.identity(sample_string), sample_string)
        self.assertIs(qv.identity(sample_integer), sample_integer)

    def test_invert_matrix(self):
        """We test that the symbolic inverse satisfies :math:`A A^{-1}=I`."""
        a = hy.par[0]
        b = hy.par[1]
        c = hy.par[2]

        # Build a nontrivial symbolic 2x2 matrix with mixed entries.
        matrix = np.array(
            [[a, b],
             [c, a + c]],
            dtype=object,
        )

        # Compute its inverse using the Gaussian-elimination helper.
        inverse = qv.invert_matrix(matrix)

        # Check that multiplying the matrix by its inverse gives the identity.
        for row in range(2):
            for col in range(2):
                product_entry = 0
                for idx in range(2):
                    product_entry = product_entry + matrix[row, idx] * inverse[idx, col]

                simplified = sp.simplify(hy.to_sympy(product_entry))
                expected = sp.Integer(1) if row == col else sp.Integer(0)
                self.assertEqual(simplified, expected)

    def test_block_diagonal_system(self):
        """We test that symbolic elimination matches NumPy on repeated blocks."""
        coefficient_symbols = np.array(
            [hy.make_vars("S00"), hy.make_vars("S01"),
             hy.make_vars("S10"), hy.make_vars("S11")],
            dtype=object,
        ).reshape(2, 2)
        rhs_symbols = np.array(
            [hy.make_vars("b0"), hy.make_vars("b1")], dtype=object
        ).reshape(1, 2)

        # Repeat one symbolic block to exercise known-zero tracking.
        coefficient_matrix = qv.create_block_diagonal(
            [coefficient_symbols] * 10
        )
        symbolic_solution = qv.gaussian_elimination(
            coefficient_matrix, np.tile(rhs_symbols, 10)
        )

        # Evaluate the symbolic solution at one shared matrix and right-hand side.
        values = [1.4, 1.0, 1.0, 1.1, 0.1, 0.2]
        evaluator = hy.cfunc(
            symbolic_solution.flatten(),
            vars=coefficient_symbols.flatten().tolist() + rhs_symbols.flatten().tolist(),
        )
        symbolic_result = np.asarray(evaluator(values)).reshape(10, 2)

        # Compare with a direct numerical solve, repeated for every block.
        numeric_block = np.array([[1.4, 1.0], [1.0, 1.1]])
        numeric_result = np.tile(
            np.linalg.solve(numeric_block, [0.1, 0.2]), (10, 1)
        )

        np.testing.assert_allclose(symbolic_result, numeric_result)