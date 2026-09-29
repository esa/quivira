"""Miscellaneous small utility functions for Quivira."""

import numpy as np

from ._gaussian_elimination import gaussian_elimination


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
    identity = np.eye(dim, dtype=object)
    inverse = np.empty((dim, dim), dtype=object)

    for column in range(dim):
        rhs = identity[:, column]
        solution = gaussian_elimination(matrix, rhs)
        inverse[:, column] = solution.reshape(dim)

    return inverse

