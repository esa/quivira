"""Symbolic and Taylor-based dynamics for pin-jointed lattices."""

from ._version import __version__
from ._block_diagonal import create_block_diagonal
from ._gaussian_elimination import gaussian_elimination

__all__ = ["__version__", "create_block_diagonal", "gaussian_elimination"]