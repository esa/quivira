"""Symbolic and Taylor-based dynamics for pin-jointed lattices."""

from ._version import __version__
from ._linear_algebra import (
    create_block_diagonal,
    gaussian_elimination,
    identity,
    invert_matrix,
)
from ._lagrangian_math import (
    build_Jdqd,
    build_constraint_jacobian,
    build_mass_matrix,
    lagrange_eom,
    find_constrained_accelerations,
)
from . import test

__all__ = [
    "__version__",
    "build_constraint_jacobian",
    "build_Jdqd",
    "build_mass_matrix",
    "lagrange_eom",
    "create_block_diagonal",
    "find_constrained_accelerations",
    "gaussian_elimination",
    "identity",
    "invert_matrix",
    "test",
]