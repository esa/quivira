API
===

The public Quivira interface provides symbolic matrix utilities and helpers
for assembling constrained Lagrangian equations of motion. These functions
work directly with :class:`heyoka.expression` objects and NumPy arrays.

.. currentmodule:: quivira

.. autofunction:: create_block_diagonal

.. autofunction:: gaussian_elimination

.. autofunction:: identity

.. autofunction:: invert_matrix

Constrained Lagrangian mechanics
--------------------------------

.. autofunction:: build_mass_matrix

.. autofunction:: build_Jdqd

.. autofunction:: build_constraint_jacobian

.. autofunction:: find_constrained_accelerations

.. autofunction:: lagrange_eom