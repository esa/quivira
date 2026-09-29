"""Symbolic and Taylor-based dynamics for pin-jointed lattices."""

from ._version import __version__
from ._block_diagonal import create_block_diagonal
from ._gaussian_elimination import gaussian_elimination
from ._lagrangian_math import (
    build_Jdqd,
    build_constraint_jacobian,
    build_mass_matrix,
    build_ode_equations_of_motion,
    find_constrained_accelerations,
)
from ._misc import identity, invert_matrix
from . import test

__all__ = [
    "__version__",
    "build_constraint_jacobian",
    "build_Jdqd",
    "build_mass_matrix",
    "build_ode_equations_of_motion",
    "create_block_diagonal",
    "find_constrained_accelerations",
    "gaussian_elimination",
    "identity",
    "invert_matrix",
    "test",
]