"""Sparse-aware Gaussian elimination for symbolic matrices."""

import heyoka as hy
import numpy as np


def gaussian_elimination(S, b):
    """Solve a symbolic linear system with sparse-aware Gaussian elimination.

    Args:
        S (:class:`numpy.ndarray` or sequence): A square coefficient matrix
            containing :class:`heyoka.expression` objects. The input may be a
            NumPy array, preferably with object dtype, or a nested sequence
            such as a list of lists. It is converted to a two-dimensional
            object array before the elimination is performed.

        b (:class:`numpy.ndarray` or sequence): The right-hand side vector
            containing symbolic or numeric expressions. It may have shape
            ``(dim,)``, ``(1, dim)``, or ``(dim, 1)`` and must contain exactly
            ``dim`` elements. It is converted to an object array before use.

    Raises:
        ValueError: If ``S`` is not a square two-dimensional array or ``b``
            does not contain exactly ``dim`` elements.

        ZeroDivisionError: If a diagonal pivot is equal to
            ``hy.expression(0)``.

    Notes:
        Entries equal to :class:`heyoka.expression` zero are tracked as known
        zeros and skipped during elimination. The coefficient matrix and
        right-hand side are copied before row operations are applied, so the
        input arrays are not modified. The algorithm does not perform row
        pivoting.

    Returns:
        :class:`numpy.ndarray`: The solution vector as an object array with
            shape ``(dim, 1)``.
    """
    # Reuse one symbolic zero while tracking which matrix entries are known
    # to vanish, allowing sparse rows to skip unnecessary expression work.
    zero = hy.expression(0)

    def is_zero(value):
        return value == zero

    # Normalize the coefficient matrix and right-hand side before modifying
    # private working copies during elimination.
    S = np.asarray(S, dtype=object)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square 2D array")

    dim = S.shape[0]
    rhs = np.asarray(b, dtype=object)
    if rhs.size != dim:
        raise ValueError("b must contain exactly dim elements")

    A = S.copy()
    rhs = rhs.reshape(dim).copy()

    # Build the zero mask once so both elimination phases can avoid known
    # zero entries without repeatedly comparing symbolic expressions.
    zero_pattern = np.empty(A.shape, dtype=bool)
    for row in range(dim):
        for col in range(dim):
            zero_pattern[row, col] = is_zero(A[row, col])

    # Reduce the matrix to upper-triangular form while updating the right-hand
    # side with the same row operations.
    for pivot_row in range(dim - 1):
        pivot = A[pivot_row, pivot_row]
        if zero_pattern[pivot_row, pivot_row]:
            raise ZeroDivisionError("zero pivot encountered")

        for row in range(pivot_row + 1, dim):
            if zero_pattern[row, pivot_row]:
                continue

            factor = A[row, pivot_row] / pivot
            A[row, pivot_row] = zero
            zero_pattern[row, pivot_row] = True

            for col in range(pivot_row + 1, dim):
                if zero_pattern[pivot_row, col]:
                    continue
                A[row, col] = A[row, col] - factor * A[pivot_row, col]
                zero_pattern[row, col] = is_zero(A[row, col])

            rhs[row] = rhs[row] - factor * rhs[pivot_row]

    if zero_pattern[dim - 1, dim - 1]:
        raise ZeroDivisionError("zero pivot encountered")

    # Recover the solution by substituting already-solved variables backwards
    # through the upper-triangular system.
    x = np.empty(dim, dtype=object)
    for row in range(dim - 1, -1, -1):
        known_terms = zero
        for col in range(row + 1, dim):
            if zero_pattern[row, col]:
                continue
            known_terms = known_terms + A[row, col] * x[col]
        x[row] = (rhs[row] - known_terms) / A[row, row]

    return x.reshape(dim, 1)
