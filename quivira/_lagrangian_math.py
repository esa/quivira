"""Important matrices for constrained Lagrangian systems."""

import numpy as np
import heyoka as hy
from scipy.differentiate import jacobian

from ._linear_algebra import gaussian_elimination, gaussian_solve_factored, gaussian_factorize, sparse_matmul

# optional
from contextlib import contextmanager
import time

@contextmanager
def timed(name):
    start = time.perf_counter()
    yield
    print(f"{name} took {time.perf_counter() - start:.6f} seconds")

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

    derivatives = hy.diff_tensors(
        [lagrangian],
        diff_args=list(qd),
        diff_order=2,
    )

    return derivatives.hessian(0)


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
    q = list(q)
    qd = np.asarray(qd, dtype=object)

    # First construct J * qd
    Jqd = J @ qd

    # Differentiate this vector with respect to q
    derivative = hy.diff_tensors(
        list(Jqd),
        diff_args=q,
        diff_order=1,
    ).jacobian

    # Multiply by qd again
    return derivative @ qd

def find_constrained_accelerations(qdd_free, J, M, Jdqd,  Q_nc = None, *, return_W=False):
    r"""Return constrained accelerations and the constraint multipliers.

    The constrained acceleration is

    .. math::

        \ddot{q} = \ddot{q}_{\mathrm{free}} - M^{-1} J^T W^{-1}
        \left(J \ddot{q}_{\mathrm{free}} + \dot{J} \dot{q}\right),

    where

    .. math::

        W = J M^{-1} J^T.

    this is simplified to:
    The constrained acceleration is

    .. math::

        \ddot{q} = \ddot{q}_{\mathrm{applied}} + G \lambda,

    where

    .. math::

        G = M^{-1} J^T, \qquad W = JG,

    and

    .. math::

        W \lambda = -(J \ddot{q}_{\mathrm{applied}} + \dot{J}\dot{q}).

    The matrix G is calculated by solving M G = J^T, avoiding the
    explicit inversion of M. The mass matrix is factorized only once,
    and the factorization is reused for all right-hand sides.

    Args:
        qdd_free (sequence): Free acceleration vector.
        J (:class:`numpy.ndarray` or sequence): Constraint Jacobian :math:`J`.
        M (:class:`numpy.ndarray` or sequence): Mass matrix :math:`M`.
        Jdqd (sequence): Product :math:`\dot J \dot q`.
        Q_nc (:class:`numpy.ndarray` or sequence, optional): Generalized applied forces,
            in the same ordering as q. Entries may be symbolic
            functions of q, qd, and time. Defaults to None.
        return_W (:class:`bool`, optional): Whether to also return the Delassus
            matrix :math:`W`. Default is False.

    Returns:
        :class:`tuple`: By default, a pair ``(qdd, lambda_mult)`` containing the
        constrained acceleration vector and Lagrange multiplier vector. If
        ``return_W`` is True, a triple ``(qdd, lambda_mult, W)`` is returned;
        :math:`W = J M^{-1} J^T` is the Delassus matrix. The multiplier entries
        follow the constraint-row ordering in :math:`J`.
    """
    J = np.asarray(J, dtype=object)
    M = np.asarray(M, dtype=object)
    qdd_free = np.asarray(qdd_free, dtype=object).reshape(-1)
    Jdqd = np.asarray(Jdqd, dtype=object).reshape(-1)

    # Matrix checks
    if J.ndim != 2:
        raise ValueError("J must be a 2D array")
    m, n = J.shape
    if M.shape != (n, n):
        raise ValueError("M must have shape (n, n)")
    if qdd_free.size != n or Jdqd.size != m:
        raise ValueError("Incompatible vector dimensions")
    if Q_nc is not None:
        Q_nc = np.asarray(Q_nc, dtype=object)
        if Q_nc.shape != qdd_free.shape:
            raise ValueError("Q_nc must have the same shape as qdd_free")

    # Factorize M once and reuse the factorization for all subsequent solves.
    with timed("factorize M"):
        M_factors = gaussian_factorize(M)

    # Add generalized applied forces by solving M @ force_acc = Q_nc.
    with timed("force contribution"):
        if Q_nc is not None:
            force_acc = gaussian_solve_factored(M_factors, Q_nc).reshape(-1)
            qdd_free = qdd_free + force_acc

    # Handle systems without constraints.
    if m == 0:
        lambda_mult = np.empty(0, dtype=object)
        if return_W:
            return qdd_free, lambda_mult, np.empty((0, 0), dtype=object)
        return qdd_free, lambda_mult

    # Calculate G = M^-1 @ J.T without explicitly inverting M.
    # Solve M @ G = J.T column by column, reusing the factorization.
    with timed("solve M @ G = J.T"):
        G = np.empty((n, m), dtype=object)
        for col in range(m):
            rhs = J[col, :]
            G[:, col] = gaussian_solve_factored(M_factors, rhs).reshape(-1)

    # Construct the Delassus matrix W = J @ M^-1 @ J.T = J @ G.
    with timed("build W"):
        W = sparse_matmul(J, G)

    # Construct the right-hand side of W @ lambda = rhs.
    with timed("build rhs"):
        rhs = -(J @ qdd_free + Jdqd)

    # Solve for the Lagrange multipliers.
    with timed("gaussian_elimination"):
        lambda_mult = gaussian_elimination(W, rhs).reshape(-1)

    # Calculate constrained accelerations, reusing G = M^-1 @ J.T.
    with timed("acceleration correction"):
        correction = sparse_matmul(G, lambda_mult.reshape(-1, 1)).reshape(-1)
        qdd = qdd_free + correction
    
    if return_W:
        return qdd, lambda_mult, W

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


def lagrange_eom(lagrangian, F, q, qd, Q_nc=None, *, return_multipliers=False, return_W=False):
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
        Q_nc (sequence, optional): Generalized applied forces,
            in the same ordering as q. Entries may be symbolic
            functions of q, qd, and time. Defaults to None.
        return_multipliers (:class:`bool`, optional): Whether to also return
            the Lagrange multiplier vector. Default is False.
        return_W (:class:`bool`, optional): Whether to also return the Delassus
            matrix :math:`W = J M^{-1} J^T`. Default is False.

    Returns:
        :class:`list` or :class:`tuple`: The ODE system as pairs
        ``(state_var, rhs_expr)``. If both options are False, this list is
        returned directly. If only ``return_multipliers`` is True, returns
        ``(ode_system, lambda_mult)``; if only ``return_W`` is True, returns
        ``(ode_system, W)``; if both are True, returns
        ``(ode_system, lambda_mult, W)``. The multiplier entries follow the
        constraint-row ordering in :math:`J`.
    """
    # Heyoka returns the Euler-Lagrange equations as ordered pairs of the form
    # (state_variable, rhs_expression). Keep only the entries for the velocity
    # variables so the free acceleration matches the ordering in qd.
    start = time.perf_counter()
    free_system = hy.lagrangian(lagrangian, q, qd)
    elapsed = time.perf_counter() - start
    print(f"hy.lagrangian took {elapsed:.6f} seconds")
    qdd_free = []
    for variable, rhs in free_system:
        if variable in qd:
            qdd_free.append(rhs)
    qdd_free = np.asarray(qdd_free, dtype=object)

    # Build the mass matrix, the constraint Jacobian, and the chain-rule term
    # Jdot qd that appears in the acceleration constraint.
    start = time.perf_counter()
    M = build_mass_matrix(lagrangian, qd)
    elapsed = time.perf_counter() - start
    print(f"build_mass_matrix took {elapsed:.6f} seconds")
    start = time.perf_counter()
    J = build_constraint_jacobian(F, q)
    elapsed = time.perf_counter() - start
    print(f"build_constraint_jacobian took {elapsed:.6f} seconds")
    start = time.perf_counter()
    Jdqd = build_Jdqd(J, q, qd)
    elapsed = time.perf_counter() - start
    print(f"build_Jdqd took {elapsed:.6f} seconds")

    # Enforce the algebraic constraints by projecting the free acceleration onto the constraint-consistent subspace.
    start = time.perf_counter()
    if return_W:
        qdd, lambda_mult, W = find_constrained_accelerations(qdd_free, J, M, Jdqd, Q_nc, return_W=True)
    else:
        qdd, lambda_mult = find_constrained_accelerations(qdd_free, J, M, Jdqd, Q_nc)
    elapsed = time.perf_counter() - start
    print(f"find_constrained_accelerations took {elapsed:.6f} seconds")

    # Turn the second-order system into the first-order form expected by the integrator: qdot = v and vdot = qdd.
    ode_system = []
    for coord, vel in zip(q, qd):
        ode_system.append((coord, vel))
    for vel, acc in zip(qd, qdd):
        ode_system.append((vel, acc))

    if return_multipliers:
        if return_W:
            return ode_system, lambda_mult, W
        return ode_system, lambda_mult
    if return_W:
        return ode_system, W

    return ode_system


