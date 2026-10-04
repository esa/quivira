"""Important matrices for constrained Lagrangian systems."""

import numpy as np
import heyoka as hy
from scipy.differentiate import jacobian

from ._linear_algebra import gaussian_elimination, invert_matrix


def build_mass_matrix(lagrangian, qd):
    r"""Return the symbolic mass matrix for a Lagrangian.

    The mass matrix is defined by

    .. math::

        M_{ij} = \frac{\partial^2 \mathcal L}{\partial \dot q_i \partial \dot q_j}.

    Args:
        lagrangian (:class:`heyoka.expression`): Scalar Lagrangian
            :math:`\mathcal L(q, \dot q)`.
        qd (sequence): Generalized velocities in the same ordering as the
            model coordinates.

    Returns:
        :class:`numpy.ndarray`: Symbolic mass matrix with shape ``[n, n]``.
    """
    n = len(qd)
    matrix = []

    for i in range(n):
        row = []
        for j in range(n):
            entry = hy.diff(hy.diff(lagrangian, qd[j]), qd[i])
            row.append(entry)
        matrix.append(row)

    return np.asarray(matrix, dtype=object)


def build_Jdqd(J, q, qd):
    r"""Return the vector :math:`\dot J \dot q` from the Jacobian and state rate.

    The derivative of the constraint Jacobian along the trajectory is

    .. math::

        \dot J = \sum_i \frac{\partial J}{\partial q_i} \dot q_i,

    so the product entering the constrained-acceleration equation is

    .. math::

        \dot J \dot q = \sum_i \frac{\partial J}{\partial q_i} \dot q_i \dot q.

    Args:
        J (:class:`numpy.ndarray` or sequence): Constraint Jacobian :math:`J(q)`.
        q (sequence): Generalized coordinates.
        qd (sequence): Generalized velocities :math:`\dot q`.

    Returns:
        :class:`numpy.ndarray`: The vector :math:`\dot J \dot q`.
    """
    J = np.asarray(J, dtype=object)
    q = list(q)
    qd = list(qd)

    if J.ndim != 2:
        raise ValueError("J must be a 2D array")
    if len(q) != J.shape[1]:
        raise ValueError("J.shape[1] must match len(q)")
    if len(qd) != len(q):
        raise ValueError("len(qd) must match len(q)")

    # The product is the chain-rule form of \dot J \dot q.
    Jdqd = np.zeros(J.shape[0], dtype=object)

    for row in range(J.shape[0]):
        for i, qi in enumerate(q):
            # Differentiate each row of J with respect to the current coordinate.
            jacobian_row = []
            for col in range(J.shape[1]):
                entry = J[row, col]
                if entry == 0:
                    derivative = 0
                else:
                    derivative = hy.diff(entry, qi)
                jacobian_row.append(derivative)

            # Accumulate the term (dJ/dq_i) qdot_i dot qdot, row by row.
            Jdqd[row] = Jdqd[row] + qd[i] * np.dot(jacobian_row, qd)

    return Jdqd

def find_constrained_accelerations(qdd_free, J, M, Jdqd):
    r"""Return constrained accelerations and the constraint multipliers.

    The constrained acceleration is

    .. math::

        \ddot{q} = \ddot{q}_{\mathrm{free}} - M^{-1} J^T W^{-1}
        \left(J \ddot{q}_{\mathrm{free}} + \dot{J} \dot{q}\right),

    where

    .. math::

        W = J M^{-1} J^T.

    Args:
        qdd_free (sequence): Free acceleration vector.
        J (:class:`numpy.ndarray` or sequence): Constraint Jacobian :math:`J`.
        M (:class:`numpy.ndarray` or sequence): Mass matrix :math:`M`.
        Jdqd (sequence): Product :math:`\dot J \dot q`.

    Returns:
        :class:`tuple`: A pair containing the constrained acceleration vector
        and the Lagrange multiplier vector, in the same order as the constraint
        residuals in :math:`F`.
    """
    J = np.asarray(J, dtype=object)
    M = np.asarray(M, dtype=object)
    qdd_free = np.asarray(qdd_free, dtype=object)
    Jdqd = np.asarray(Jdqd, dtype=object)

    # The constraint force acts through the term J^T W^{-1}(J qdd_free + Jdot qd),
    # with the sign convention from the derivation in the TeX notes.
    M_inv = invert_matrix(M)
    JT = J.T
    W = J @ M_inv @ JT

    # Solve W * lambda = -(J qdd_free + Jdqd), which is equivalent to
    # lambda = -W^{-1}(J qdd_free + Jdqd) in the paper's sign convention.
    rhs = -(J @ qdd_free + Jdqd)
    lambda_mult = gaussian_elimination(W, rhs).reshape(-1)
    correction = M_inv @ JT @ lambda_mult
    qdd = qdd_free + correction

    return qdd, lambda_mult

def build_constraint_jacobian(F, q):
    r"""Return the constraint Jacobian :math:`J = \partial F / \partial q`.

    The algebraic constraints are expressed as a vector :math:`F(q) = 0`. The
    Jacobian maps coordinate perturbations into constraint-space variations and
    is used to enforce the acceleration-level constraint
    :math:`J \ddot{q} + \dot{J} \dot{q} = 0`.

    Args:
        F (sequence): Constraint residuals :math:`F(q)`.
        q (sequence): Generalized coordinates.

    Returns:
        :class:`numpy.ndarray`: Jacobian matrix with shape ``[m, n]``.
    """
    # Differentiate each scalar constraint with respect to each generalized
    # coordinate, producing the symbolic Jacobian matrix.
    jacobian = hy.diff_tensors(F, diff_args=q, diff_order=1)
    J = np.asarray(jacobian.jacobian, dtype=object)
    return J


def lagrange_eom(lagrangian, F, q, qd, *, return_multipliers=False):
    r"""Assemble the first-order ODE system for the constrained dynamics.

    This helper forms the free-acceleration terms from the Lagrangian,
    evaluates the constraint Jacobian and its time derivative, and then returns
    the equivalent first-order system that can be passed to a numerical
    integrator.

    Args:
        lagrangian (:class:`heyoka.expression`): Scalar Lagrangian.
        F (sequence): Constraint residuals :math:`F(q)`.
        q (sequence): Generalized coordinates.
        qd (sequence): Generalized velocities.
        return_multipliers (:class:`bool`, optional): Whether to also return
            the symbolic Lagrange multipliers. Default is False.

    Returns:
        :class:`list` or :class:`tuple`: The ODE system as pairs
        ``(state_var, rhs_expr)``. If ``return_multipliers`` is True, returns a
        pair containing the ODE system and the multiplier vector, ordered as
        the constraint residuals in :math:`F`.
    """
    # Heyoka returns the Euler-Lagrange equations as ordered pairs of the form
    # (state_variable, rhs_expression). Keep only the entries for the velocity
    # variables so the free acceleration matches the ordering in qd.
    free_system = hy.lagrangian(lagrangian, q, qd)
    qdd_free = []
    for variable, rhs in free_system:
        if variable in qd:
            qdd_free.append(rhs)
    qdd_free = np.asarray(qdd_free, dtype=object)

    # Build the mass matrix, the constraint Jacobian, and the chain-rule term
    # Jdot qd that appears in the acceleration constraint.
    M = build_mass_matrix(lagrangian, qd)
    J = build_constraint_jacobian(F, q)
    Jdqd = build_Jdqd(J, q, qd)

    # Enforce the algebraic constraints by projecting the free acceleration onto the constraint-consistent subspace.
    qdd, lambda_mult = find_constrained_accelerations(qdd_free, J, M, Jdqd)

    # Turn the second-order system into the first-order form expected by the integrator: qdot = v and vdot = qdd.
    ode_system = []
    for coord, vel in zip(q, qd):
        ode_system.append((coord, vel))
    for vel, acc in zip(qd, qdd):
        ode_system.append((vel, acc))

    if return_multipliers:
        return ode_system, lambda_mult

    return ode_system



