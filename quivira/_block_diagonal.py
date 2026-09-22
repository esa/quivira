"""Block-diagonal matrices for symbolic arrays."""

import heyoka as hy
import numpy as np


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