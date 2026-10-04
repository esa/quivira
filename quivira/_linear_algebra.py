"""Symbolic linear-algebra helpers for Quivira."""

import heyoka as hy
import numpy as np


def identity(x):
    """Return the input unchanged.

    This utility is useful as a no-op placeholder when a function is needed
    but the caller wants to preserve the original value.

    Args:
        x: Any Python object to return unchanged.

    Returns:
        The original object passed in.
    """
    return x


def create_block_diagonal(blocks):
    """Create a block-diagonal matrix from square symbolic arrays.

    Args:
        blocks (:class:`list`): A sequence of two-dimensional square arrays
            to place along the main diagonal.

    Raises:
        ValueError: If a block is not a two-dimensional square array.

    Notes:
        Entries outside the diagonal blocks are filled with
        :class:`heyoka.expression` objects representing zero, rather than
        Python integer zeros. An empty sequence returns an empty ``(0, 0)``
        object array.

    Returns:
        :class:`numpy.ndarray`: An object array containing the block-diagonal
        matrix.
    """
    zero = hy.expression(0)
    # Handle the degenerate case before computing the total matrix dimension.
    if len(blocks) == 0:
        return np.empty((zero, zero), dtype=object)

    # Normalize the inputs and validate their shapes before constructing the
    # combined matrix.
    block_arrays = []
    for block in blocks:
        block = np.asarray(block, dtype=object)
        if block.ndim != 2 or block.shape[0] != block.shape[1]:
            raise ValueError("each block must be a square 2D array")
        block_arrays.append(block)

    # Allocate the full matrix with symbolic zeros so off-diagonal entries
    # remain compatible with heyoka expressions.
    total_dim = sum(block.shape[0] for block in block_arrays)
    result = np.full((total_dim, total_dim), zero, dtype=object)

    # Copy each block into the next diagonal slice and advance the placement
    # offset by that block's dimension.
    start = 0
    for block in block_arrays:
        end = start + block.shape[0]
        result[start:end, start:end] = block
        start = end

    return result


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


def invert_matrix(matrix):
    r"""Return the inverse of a square matrix using Gaussian elimination.

    The inverse is defined by

    .. math::

        A A^{-1} = I,

    where :math:`I` is the identity matrix. The implementation solves the
    linear systems

    .. math::

        A x_j = e_j

    for each basis vector :math:`e_j` and stores the resulting columns in the
    inverse matrix.

    Args:
        matrix (:class:`numpy.ndarray` or sequence): Square matrix to invert.

    Raises:
        ValueError: If ``matrix`` is not a square two-dimensional array.

    Notes:
        This routine is intended for the symbolic matrices used throughout
        Quivira. It relies on the existing Gaussian elimination routine and is
        therefore compatible with :class:`heyoka.expression` entries.

    Returns:
        :class:`numpy.ndarray`: Inverse matrix with the same object dtype as the
            input matrix.
    """
    matrix = np.asarray(matrix, dtype=object)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square 2D array")

    dim = matrix.shape[0]
    identity_matrix = np.eye(dim, dtype=object)
    inverse = np.empty((dim, dim), dtype=object)

    for column in range(dim):
        rhs = identity_matrix[:, column]
        solution = gaussian_elimination(matrix, rhs)
        inverse[:, column] = solution.reshape(dim)

    return inverse