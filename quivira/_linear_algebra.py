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


def _as_expression(value):
    """Convert a numeric scalar to a Heyoka expression."""
    if isinstance(value, hy.expression):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    return hy.expression(value)


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


def old_gaussian_elimination(S, b):
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

# New gaussian elimination splits factorization and solve


def gaussian_factorize(S):
    """Compute a sparsity-aware symbolic LU factorization.

    Stores L (without its unit diagonal) and U in one matrix.
    Tracks structural zeros and caches nonzero column indices
    for efficient repeated solves. Does not perform row pivoting.

    Args:
        S (:class:`numpy.ndarray` or sequence): Square symbolic
            coefficient matrix with shape ``(n, n)``.

    Raises:
        ValueError: If ``S`` is not square.
        ZeroDivisionError: If a structurally zero pivot is encountered.

    Returns:
        tuple: (LU, zero_pattern, lower_cols, upper_cols).
            The column lists contain the nonzero positions of
            the lower and upper triangular factors.
    """
    zero = hy.expression(0)

    # Validate and copy the coefficient matrix.
    A = np.asarray(S, dtype=object)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("S must be a square 2D array")

    A = A.copy()
    dim = A.shape[0]

    for index in np.ndindex(A.shape):
        A[index] = _as_expression(A[index])

    # Identify structural zeros and nonzero entries above the diagonal.
    zero_pattern = np.empty(A.shape, dtype=bool)
    upper_active = [set() for _ in range(dim)]

    for row in range(dim):
        for col in range(dim):
            is_zero = A[row, col] == zero
            zero_pattern[row, col] = is_zero

            if col > row and not is_zero:
                upper_active[row].add(col)

    # Perform Gaussian elimination and store the factors of L.
    for pivot_row in range(dim):
        if zero_pattern[pivot_row, pivot_row]:
            raise ZeroDivisionError("zero pivot encountered")

        pivot = A[pivot_row, pivot_row]

        # Only these columns can contribute to elimination.
        pivot_cols = sorted(upper_active[pivot_row])

        for row in range(pivot_row + 1, dim):
            if zero_pattern[row, pivot_row]:
                continue

            factor = A[row, pivot_row] / pivot

            # Store the elimination factor in the lower triangle.
            A[row, pivot_row] = factor
            zero_pattern[row, pivot_row] = factor == zero

            if zero_pattern[row, pivot_row]:
                continue

            # Update only columns where the pivot row is nonzero.
            for col in pivot_cols:
                if zero_pattern[row, col]:
                    updated = -(factor * A[pivot_row, col])
                else:
                    updated = A[row, col] - factor * A[pivot_row, col]

                A[row, col] = updated
                is_zero = updated == zero
                zero_pattern[row, col] = is_zero

                # Update the sparsity pattern as fill-in occurs.
                if col > row:
                    if is_zero:
                        upper_active[row].discard(col)
                    else:
                        upper_active[row].add(col)

    # Cache nonzero column indices for repeated substitutions.
    lower_cols = [
        [col for col in range(row) if not zero_pattern[row, col]]
        for row in range(dim)
    ]

    upper_cols = [
        sorted(cols) for cols in upper_active
    ]

    return A, zero_pattern, lower_cols, upper_cols



def gaussian_solve_factored(factorization, b):
    """Solve a symbolic system using a cached sparsity-aware LU factorization.

    Solves S @ x = b through forward and backward substitution,
    skipping structurally zero matrix entries and intermediate values.

    Args:
        factorization (tuple): Output from ``gaussian_factorize()``,
            containing the LU matrix and cached sparsity information.
        b (:class:`numpy.ndarray` or sequence): Right-hand side vector
            containing exactly ``n`` elements.

    Raises:
        ValueError: If ``b`` has an incompatible size.

    Returns:
        :class:`numpy.ndarray`: Symbolic solution vector with
        shape ``(n, 1)``.
    """
    LU, zero_pattern, lower_cols, upper_cols = factorization
    dim = LU.shape[0]
    zero = hy.expression(0)

    # Validate the right-hand side.
    rhs = np.asarray(b, dtype=object)

    if rhs.size != dim:
        raise ValueError("b must contain exactly dim elements")

    rhs = rhs.reshape(dim).copy()
    for row in range(dim):
        rhs[row] = _as_expression(rhs[row])

    # 1. Forward substitution: L @ y = b.
    y = np.empty(dim, dtype=object)

    for row in range(dim):
        terms = []

        if rhs[row] != zero:
            terms.append(rhs[row])

        # Use only nonzero entries of L and y.
        for col in lower_cols[row]:
            if y[col] == zero:
                continue

            terms.append(-LU[row, col] * y[col])

        # Combine terms into a flat symbolic sum.
        if not terms:
            y[row] = zero
        elif len(terms) == 1:
            y[row] = terms[0]
        else:
            y[row] = hy.sum(terms)

    # 2. Backward substitution: U @ x = y.
    x = np.empty(dim, dtype=object)

    for row in range(dim - 1, -1, -1):
        terms = []

        if y[row] != zero:
            terms.append(y[row])

        # Use only nonzero entries of U and x.
        for col in upper_cols[row]:
            if x[col] == zero:
                continue

            terms.append(-LU[row, col] * x[col])

        if not terms:
            x[row] = zero
        else:
            numerator = (
                terms[0] if len(terms) == 1 else hy.sum(terms)
            )

            # Avoid unnecessary symbolic division by the pivot.
            x[row] = (
                zero if numerator == zero
                else numerator / LU[row, row]
            )

    return x.reshape(dim, 1)

def gaussian_elimination(S, b):
    r"""Solve a symbolic linear system using sparsity-aware Gaussian elimination.

    Solves the linear system

    .. math::

        S x = b

    by first computing an LU factorization of the coefficient matrix
    and then performing forward and backward substitution.

    The factorization is computed for each call. To solve multiple systems
    with the same coefficient matrix more efficiently, use
    ``gaussian_factorize()`` once and reuse the result with
    ``gaussian_solve_factored()``.

    Args:
        S (:class:`numpy.ndarray` or sequence): Square coefficient matrix
            containing symbolic or numeric entries, with shape ``(n, n)``.

        b (:class:`numpy.ndarray` or sequence): Right-hand side vector
            containing symbolic or numeric entries. Must contain exactly
            ``n`` elements.

    Raises:
        ValueError: If ``S`` is not square or ``b`` has an incompatible size.

        ZeroDivisionError: If a structurally zero pivot is encountered
            during factorization.

    Notes:
        The underlying factorization skips known zero entries to reduce
        unnecessary symbolic operations. No row pivoting is performed,
        so the solver may fail for invertible matrices with zero pivots.

        The input matrix and right-hand side are not modified.

    Returns:
        :class:`numpy.ndarray`: Symbolic solution vector with shape
        ``(n, 1)``.
    """
    factorization = gaussian_factorize(S)
    return gaussian_solve_factored(factorization, b)


def sparse_matmul(A, B):
    """Multiply two symbolic matrices while skipping known zero entries.

    Computes C = A @ B using only products where both entries are
    structurally nonzero. Nonzero entries of B are identified once
    and reused throughout the multiplication.

    Args:
        A (:class:`numpy.ndarray` or sequence): Left matrix (m, n).
        B (:class:`numpy.ndarray` or sequence): Right matrix (n, p).

    Returns:
        :class:`numpy.ndarray`: Symbolic product matrix (m, p).
    """
    A = np.asarray(A, dtype=object)
    B = np.asarray(B, dtype=object)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Both inputs must be 2D matrices")

    m, n = A.shape
    n2, p = B.shape
    print(f"my_matmul: A{A.shape} @ B{B.shape} -> C({m}, {p})")

    if n != n2:
        raise ValueError("Incompatible matrix shapes")

    zero = hy.expression(0)
    C = np.empty((m, p), dtype=object)

    # Store only nonzero entries of each row of B.
    B_rows = [
        [(j, B[k, j]) for j in range(p) if B[k, j] != zero]
        for k in range(n)
    ]

    # Calculate one output row at a time.
    for i in range(m):

        # Collect the symbolic terms for each output column.
        terms = [[] for _ in range(p)]

        for k in range(n):
            aik = A[i, k]

            # A zero entry contributes nothing to the product.
            if aik == zero:
                continue

            # Only multiply by nonzero entries of B.
            for j, bkj in B_rows[k]:
                terms[j].append(aik * bkj)

        # Combine the contributions with heyoka.sum().
        for j in range(p):
            if not terms[j]:
                C[i, j] = zero
            elif len(terms[j]) == 1:
                C[i, j] = terms[j][0]
            else:
                C[i, j] = hy.sum(terms[j])

    return C
