
"""Correct configurations and velocities of holonomically constrained systems.

Both functions work with generalized coordinates and symbolic constraints.
They do not depend on a particular Quivira model class.
"""

import heyoka as hy
import numpy as np
from scipy.optimize import minimize

from ._lagrangian_math import build_constraint_jacobian

def parameter_values(expressions, pars=None):
    """Select numerical parameter values required by Heyoka expressions.

    The returned array includes all supplied values through the highest
    parameter index used by ``expressions``; unused intermediate values are
    retained because Heyoka parameters are indexed.

    Args:
        expressions (sequence): Symbolic expressions whose Heyoka parameters
            determine the required values.

        pars (array_like, optional): Numerical parameter values indexed from
            ``hy.par[0]``. Default is ``None``.

    Raises:
        ValueError: If expressions require parameters but no values are supplied,
            or if the supplied values do not include every required parameter.

    Returns:
        :class:`numpy.ndarray`: One-dimensional parameter values through the
        highest required parameter index, or an empty array if none are used.
    """

    used_parameters = hy.get_params(expressions)

    # No parameters are required
    if not used_parameters:
        return np.empty(0, dtype=float)

    if pars is None:
        raise ValueError("Numerical parameter values are required")

    values = np.asarray(pars, dtype=float).ravel()

    # Find the indices of the parameters used in the expressions
    used_indices = []

    for index in range(len(values)):
        if hy.par[index] in used_parameters:
            used_indices.append(index)

    # Make sure the supplied array contains every required parameter
    if len(used_indices) != len(used_parameters):
        raise ValueError("Not enough parameter values supplied")

    # Keep all numerical values up to the highest required index
    highest_index = max(used_indices)
    return values[:highest_index + 1]


def _compile_constraints(q, F, pars=None):
    """Compile symbolic constraints and their Jacobian for numerical evaluation.

    Args:
        q (sequence): Symbolic generalized coordinates.

        F (sequence): Holonomic constraint residuals.

        pars (array_like, optional): Numerical Heyoka parameter values. Default
            is ``None``.

    Returns:
        :class:`tuple`: Constraint and Jacobian callables. For a coordinate
        vector of length ``n`` and ``m`` constraints, they return arrays with
        shapes ``(m,)`` and ``(m, n)``, respectively.
    """

    nq = len(q)
    m = len(F)

    # Build the symbolic constraint Jacobian
    J = build_constraint_jacobian(F, q)
    J = np.asarray(J, dtype=object).ravel().tolist()

    # Compile the symbolic expressions
    f_compiled = hy.cfunc(F, vars=q, compact_mode=True)
    j_compiled = hy.cfunc(J, vars=q, compact_mode=True)

    # Select the numerical parameters required by each compiled function
    f_pars = parameter_values(F, pars)
    j_pars = parameter_values(J, pars)

    # Numerical constraint function
    def constraints(q_value):
        result = f_compiled(q_value, pars=f_pars)
        return np.asarray(result, dtype=float).reshape(m)

    # Numerical constraint Jacobian
    def jacobian(q_value):
        result = j_compiled(q_value, pars=j_pars)
        return np.asarray(result, dtype=float).reshape(m, nq)

    return constraints, jacobian

def _check_vector(values, size, name):
    """Return a finite one-dimensional float copy of the expected length."""

    # Convert the input into a NumPy array of floating-point numbers
    array = np.asarray(values, dtype=float)

    # Check that the array is one-dimensional and has the expected length
    if array.ndim != 1 or len(array) != size:
        raise ValueError(
            f"{name} must be a one-dimensional array of length {size}"
        )

    # Check that no entries are NaN or infinity
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    # Return an independent copy so the original input cannot be modified
    return array.copy()

def fix_configuration(
    q_invalid,
    q,
    F,
    pars=None,
    objective=None,
    gradient=None,
    tolerance=1e-8,
    maxiter=1000,
    bounds=None,
):
    """Project a configuration onto the holonomic constraint manifold.

    Args:
        q_invalid (array_like): Initial generalized coordinates, which may not satisfy the
            geometric constraints.

        q (sequence of heyoka expressions): Symbolic generalized coordinates, in the same
            order as ``q_invalid``.

        F (sequence of heyoka expressions): Time-independent holonomic constraint equations.

        pars (array_like, optional): Numerical values of the heyoka parameters appearing in
            ``F``. Default is ``None``.

        objective (callable, optional): Custom distance objective evaluated as
            ``objective(candidate, q_invalid)``. If omitted, the solver minimizes
            :math:`\tfrac{1}{2} \|candidate - q_{invalid}\|^2`.

        gradient (callable, optional): Gradient of ``objective`` evaluated as
            ``gradient(q_candidate, q_invalid)``. If ``objective`` is supplied without a
            gradient, SciPy estimates the gradient numerically. Default is ``None``.

        tolerance (float, optional): Maximum accepted absolute constraint residual.
            Default is ``1e-8``.

        maxiter (int, optional): Maximum number of SLSQP iterations. Default is ``1000``.

        bounds (sequence, optional): Optional lower and upper bounds for each generalized
            coordinate. Default is ``None``.

    Raises:
        ValueError: If ``q_invalid`` is not a finite one-dimensional vector of
            the expected length, or if a gradient is supplied without an
            objective. Missing Heyoka parameter values also raise this error.

        RuntimeError: If the optimizer fails to find a configuration satisfying the
            constraints within the requested tolerance.

    Notes:
        The corrected state is obtained by solving a local equality-constrained minimization
        problem around the initial guess. If the constraint list is empty or the
        input already satisfies the constraints, the function returns an
        unchanged-value copy of the input vector.

    Returns:
        :class:`numpy.ndarray`: Corrected one-dimensional generalized coordinates
        with the same length as ``q_invalid``.
    """
    q = list(q)
    F = list(F)
    reference = _check_vector(q_invalid, len(q), "q_invalid")

    if objective is None and gradient is not None:
        raise ValueError("A gradient requires a corresponding objective")

    # If there are no geometric constraints, every configuration is already valid
    if not F:
        return reference

    constraints, jacobian = _compile_constraints(q, F, pars)

    # Check whether the reference configuration already satisfies the constraints
    initial_residuals = constraints(reference)
    max_initial_residual = np.max(np.abs(initial_residuals))

    if max_initial_residual <= tolerance:
        return reference

    # If there is no problem specific objective given, we just use the difference between the lagrangian variables
    if objective is None:
        def cost(candidate):
            difference = candidate - reference
            return 0.5 * np.dot(difference, difference)

        def cost_gradient(candidate):
            return candidate - reference
    # Otherwise we use the supplied ones
    else:
        def cost(candidate):
            return float(objective(candidate, reference))

        if gradient is None:
            cost_gradient = None
        else:
            def cost_gradient(candidate):
                return np.asarray(gradient(candidate, reference), dtype=float)

    result = minimize(
        cost,
        reference,
        method="SLSQP",
        jac=cost_gradient,
        bounds=bounds,
        constraints={"type": "eq", "fun": constraints, "jac": jacobian},
        options={"ftol": min(tolerance, 1e-10), "maxiter": maxiter},
    )

    corrected = np.asarray(result.x, dtype=float)
    residuals = constraints(corrected)
    max_residual = float(np.max(np.abs(residuals)))

    if not result.success or not np.isfinite(max_residual) or max_residual > tolerance:
        raise RuntimeError(f"Configuration projection failed: {result.message}; " f"maximum constraint residual = {max_residual:.3e}")

    return corrected.copy()

def fix_velocity(
    q_valid,
    qdot_invalid,
    q,
    F,
    mass_matrix,
    pars=None,
    objective=None,
    gradient=None,
    tolerance=1e-8,
    maxiter=1000,
    bounds=None,
    target_energy=None 
):
    """Project a velocity onto the tangent space of the constraints while preserving kinetic energy.

    Args:
        q_valid (:class:`numpy.ndarray`): Finite generalized coordinates at which
            the mass matrix and constraint Jacobian are evaluated. If constraints
            are present, this configuration must already satisfy them.

        qdot_invalid (:class:`numpy.ndarray`): Finite generalized velocity to be
            corrected.

        q (sequence of heyoka expressions): Symbolic generalized coordinates, in the same
            order as ``q_valid``.

        F (sequence of heyoka expressions): Time-independent holonomic constraint equations.

        mass_matrix (:class:`numpy.ndarray`): Mass matrix evaluated at
            ``q_valid``. It must be square, symmetric within NumPy's default
            ``allclose`` tolerance, and positive definite after symmetrization.

        pars (array_like, optional): Numerical values of the Heyoka parameters
            appearing in ``F``. Default is ``None``.

        objective (callable, optional): Custom distance objective evaluated as
            ``objective(qdot_candidate, qdot_invalid)``. If omitted, the solver minimizes
            :math:`\tfrac{1}{2} \|qdot_{candidate} - qdot_{invalid}\|^2`.

        gradient (callable, optional): Gradient of ``objective`` evaluated as
            ``gradient(qdot_candidate, qdot_invalid)``. If ``objective`` is supplied without
            a gradient, SciPy estimates the gradient numerically. Default is ``None``.

        tolerance (float, optional): Maximum admissible absolute residual for the
            velocity constraints and scaled kinetic-energy constraint. Default
            is ``1e-8``.

        maxiter (int, optional): Maximum number of SLSQP iterations. Default is ``1000``.

        bounds (sequence, optional): Optional lower and upper bounds for each generalized
            velocity component. Default is ``None``.

        target_energy (float, optional): Required kinetic energy
            :math:`T = \\tfrac{1}{2} \\dot q^{\\mathsf T} M \\dot q`. If omitted,
            the energy of ``qdot_invalid`` is preserved. Default is ``None``.

    Raises:
        ValueError: If an input vector has an invalid shape or non-finite values,
            the mass matrix has an invalid shape, is not symmetric within the
            stated tolerance, or is not positive definite after symmetrization;
            if a gradient is supplied without an objective; if the target energy
            is negative or non-finite; or if ``q_valid`` violates the position
            constraints. Missing Heyoka parameter values also raise this error.

        RuntimeError: If the constrained optimization does not find a feasible velocity.

    Notes:
        The configuration ``q_valid`` is held fixed. By default, the solver
        minimizes Euclidean distance to the original velocity; a supplied
        objective replaces this criterion. The tangent-space constraint
        :math:`J(q_{valid}) \dot q = 0` and the kinetic-energy constraint
        :math:`T(\dot q) = E_{\\mathrm{target}}` are enforced using the mass
        matrix held fixed at ``q_valid``. If there are no geometric constraints,
        only the energy constraint is enforced.

    Returns:
        :class:`numpy.ndarray`: Corrected one-dimensional generalized velocity
        with the same length as ``qdot_invalid``.
    """

    q = list(q)
    F = list(F)

    # Check the input configuration and velocities
    position = _check_vector(q_valid, len(q), "q_valid")
    reference = _check_vector(qdot_invalid, len(q), "qdot_invalid")

    # Check supplied objective and gradient
    if objective is None and gradient is not None:
        raise ValueError("A gradient requires a corresponding objective")

    # Check the mass matrix
    M = np.asarray(mass_matrix, dtype=float)
    if M.shape != (len(q), len(q)):
        raise ValueError("Mass matrix has incorrect dimensions")
    if not np.allclose(M, M.T):
        raise ValueError("Mass matrix must be symmetric")
    M = 0.5 * (M + M.T) # Erases numerical errors for symmetry
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        raise ValueError("Mass matrix must be positive definite")

    # Calculate kinetic energy using the fixed mass matrix
    def kinetic_energy(candidate):
        return 0.5 * candidate @ M @ candidate

    # By default, preserve the kinetic energy of the original velocities
    if target_energy is None:
        target_energy = kinetic_energy(reference)

    target_energy = float(target_energy)

    if not np.isfinite(target_energy) or target_energy < 0:
        raise ValueError("Target kinetic energy must be finite and nonnegative")

    # Compile the geometric constraints and evaluate their Jacobian
    if F:
        constraints, jacobian = _compile_constraints(q, F, pars)

        position_error = np.max(np.abs(constraints(position)))
        if not np.isfinite(position_error) or position_error > tolerance:
            raise ValueError(
                "q_valid must satisfy the position constraints before "
                "correcting velocities"
            )

        J = jacobian(position)
    else:
        J = np.empty((0, len(q)))

    # Normalize the energy constraint for better numerical scaling
    energy_scale = max(1.0, abs(target_energy))

    # Return immediately if the reference is already valid
    velocity_error = np.max(np.abs(J @ reference)) if F else 0.0
    energy_error = abs(kinetic_energy(reference) - target_energy)

    if (
        objective is None
        and bounds is None
        and velocity_error <= tolerance
        and energy_error <= tolerance * energy_scale
    ):
        return reference

    # Define the objective and its gradient
    if objective is None:
        def cost(candidate):
            difference = candidate - reference
            return 0.5 * np.dot(difference, difference)

        def cost_gradient(candidate):
            return candidate - reference
    else:
        def cost(candidate):
            return float(objective(candidate, reference))

        if gradient is None:
            cost_gradient = None
        else:
            def cost_gradient(candidate):
                return np.asarray(
                    gradient(candidate, reference), dtype=float
                )

    # Equality constraint: J @ qdot = 0
    def velocity_constraint(candidate):
        return J @ candidate

    def velocity_constraint_jac(candidate):
        return J

    # Equality constraint: T(candidate) = target_energy
    def energy_constraint(candidate):
        return (kinetic_energy(candidate) - target_energy) / energy_scale

    def energy_constraint_jac(candidate):
        return (M @ candidate) / energy_scale

    # Assemble the constraints for SLSQP
    optimization_constraints = []

    if F:
        optimization_constraints.append({
            "type": "eq",
            "fun": velocity_constraint,
            "jac": velocity_constraint_jac,
        })

    optimization_constraints.append({
        "type": "eq",
        "fun": energy_constraint,
        "jac": energy_constraint_jac,
    })

    # Find the closest velocities satisfying both constraints
    result = minimize(
        cost,
        reference,
        method="SLSQP",
        jac=cost_gradient,
        bounds=bounds,
        constraints=optimization_constraints,
        options={
            "ftol": min(tolerance, 1e-10),
            "maxiter": maxiter,
        },
    )

    corrected = np.asarray(result.x, dtype=float)

    # Verify the velocity constraints and conservation of kinetic energy
    velocity_error = np.max(np.abs(J @ corrected)) if F else 0.0
    energy_error = abs(kinetic_energy(corrected) - target_energy)

    if (
        not result.success
        or not np.isfinite(velocity_error)
        or not np.isfinite(energy_error)
        or velocity_error > tolerance
        or energy_error > tolerance * energy_scale
    ):
        raise RuntimeError(
            f"Velocity correction failed: {result.message}; "
            f"velocity residual = {velocity_error:.3e}, "
            f"energy residual = {energy_error:.3e}"
        )

    return corrected.copy()
