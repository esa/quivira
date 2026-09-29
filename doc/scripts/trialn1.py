import quivira as qv
import heyoka as hy

def mass_matrix(lagrangian, qdot):
    r"""Return the symbolic mass matrix for a Lagrangian.

    The mass matrix is defined by

    .. math::

        M_{ij} = \frac{\partial^2 \mathcal L}{\partial \dot q_i \partial \dot q_j}.

    Args:
        lagrangian (:class:`heyoka.expression`): Scalar Lagrangian
            :math:`\mathcal L(q, \dot q)`.
        qdot (sequence): Generalized velocities in the same ordering as the
            model coordinates.

    Returns:
        :class:`numpy.ndarray`: Symbolic mass matrix with shape ``[n, n]``.
    """
    import numpy as np

    n = len(qdot)
    matrix = []

    for i in range(n):
        row = []
        for j in range(n):
            entry = hy.diff(hy.diff(lagrangian, qdot[j]), qdot[i])
            row.append(entry)
        matrix.append(row)

    return np.asarray(matrix, dtype=object)


if __name__ == "__main__":
    q = hy.make_vars("q0", "q1")
    qdot = hy.make_vars("v0", "v1")

    m = hy.par[0]
    g = hy.par[1]

    lagrangian = 0.5 * m * (qdot[0] ** 2 + qdot[1] ** 2) - m * g * q[1]
    mass = mass_matrix(lagrangian, qdot)

    for row in mass:
        print(row)


# if __name__ == "__main__":
#     print(qv.identity("Hello, Quivira!"))