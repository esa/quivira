"""Tests for miscellaneous utility helpers."""

import unittest as _ut

import numpy as np
import sympy as sp
import heyoka as hy

import quivira as qv


class misc_tests(_ut.TestCase):
    """Tests for small miscellaneous helpers."""

    def test_identity_passthrough(self):
        """The identity utility should return the exact same object."""
        sample_string = "Quivira"
        sample_integer = 42

        self.assertIs(qv.identity(sample_string), sample_string)
        self.assertIs(qv.identity(sample_integer), sample_integer)

    def test_invert_matrix(self):
        """The inverse should satisfy the symbolic identity A A^{-1} = I."""
        a = hy.par[0]
        b = hy.par[1]
        c = hy.par[2]

        # A nontrivial symbolic 2x2 matrix with mixed entries.
        matrix = np.array(
            [[a, b],
             [c, a + c]],
            dtype=object,
        )

        # Compute the symbolic inverse using the Gaussian-elimination helper.
        inverse = qv.invert_matrix(matrix)

        # Check each entry of A @ A^{-1}; the symbolic result should simplify to I.
        for row in range(2):
            for col in range(2):
                product_entry = 0
                for idx in range(2):
                    product_entry = product_entry + matrix[row, idx] * inverse[idx, col]

                simplified = sp.simplify(hy.to_sympy(product_entry))
                expected = sp.Integer(1) if row == col else sp.Integer(0)
                self.assertEqual(simplified, expected)

