"""Tests for symbolic Gaussian elimination."""

import unittest as _ut

import heyoka as hy
import numpy as np

import quivira as qv


class gaussian_elimination_tests(_ut.TestCase):
    """Test symbolic Gaussian elimination against numerical linear algebra."""

    def test_block_diagonal_system(self):
        """Compare the evaluated symbolic solution with NumPy's solution."""
        coefficient_symbols = np.array(
            [hy.make_vars("S00"), hy.make_vars("S01"),
             hy.make_vars("S10"), hy.make_vars("S11")],
            dtype=object,
        ).reshape(2, 2)
        rhs_symbols = np.array(
            [hy.make_vars("b0"), hy.make_vars("b1")], dtype=object
        ).reshape(1, 2)

        # Repeat one symbolic block to create a sparse block-diagonal system.
        # This exercises the elimination algorithm's known-zero tracking.
        coefficient_matrix = qv.create_block_diagonal(
            [coefficient_symbols] * 10
        )
        symbolic_solution = qv.gaussian_elimination(
            coefficient_matrix, np.tile(rhs_symbols, 10)
        )

        # Compile the symbolic solution and evaluate it at one numerical
        # matrix and right-hand side shared by every diagonal block.
        values = [1.4, 1.0, 1.0, 1.1, 0.1, 0.2]
        evaluator = hy.cfunc(
            symbolic_solution.flatten(),
            vars=coefficient_symbols.flatten().tolist() + rhs_symbols.flatten().tolist(),
        )
        symbolic_result = np.asarray(evaluator(values)).reshape(10, 2)

        # Use a direct numerical solve as the reference for the symbolic
        # construction, then repeat it for the ten identical blocks.
        numeric_block = np.array([[1.4, 1.0], [1.0, 1.1]])
        numeric_result = np.tile(
            np.linalg.solve(numeric_block, [0.1, 0.2]), (10, 1)
        )

        np.testing.assert_allclose(symbolic_result, numeric_result)
