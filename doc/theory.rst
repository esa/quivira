Theory
======

This page derives the equations used for constrained mechanical systems. The
notation follows the generalized coordinates :math:`\bm q`, the Lagrangian
:math:`\mathcal L`, and the constraint functions :math:`\bm F`.

Euler--Lagrange equations
-------------------------

Let :math:`\bm q=(q_1,\ldots,q_N)` describe the configuration of a system.
Its motion is described by the Lagrangian

.. math::

   \mathcal L(\bm q, \dot{\bm q})

The action over a time interval :math:`[0,T]` is

.. math::

   S = \int_0^T \mathcal L(\bm q, \dot{\bm q})\,dt.

We seek a stationary trajectory :math:`\overline{\bm q}(t)`. A nearby
trajectory is written as

.. math::

   \bm q(t) = \overline{\bm q}(t) + \delta\bm q(t),

where the endpoints stay fixed, so :math:`\delta\bm q(0)=\delta\bm q(T)=\bm 0`.
To first order, the change in the Lagrangian is

.. math::

   \delta\mathcal L =
   \left.\frac{d}{d\varepsilon}
   \mathcal L(\overline{\bm q}+\varepsilon\delta\bm q,
   \dot{\overline{\bm q}}+\varepsilon\delta\dot{\bm q})
   \right|_{\varepsilon=0}
   =
   \frac{\partial\mathcal L}{\partial\bm q}\,\delta\bm q
   +
   \frac{\partial\mathcal L}{\partial\dot{\bm q}}\,\delta\dot{\bm q}.

Requiring the first change in the action to vanish gives

.. math::

   \begin{aligned}
   \delta S
   &= \int_0^T \left(
   \frac{\partial\mathcal L}{\partial\bm q}\delta\bm q
   + \frac{\partial\mathcal L}{\partial\dot{\bm q}}
   \frac{d}{dt}(\delta\bm q)\right)dt \\
   &= \int_0^T \frac{\partial\mathcal L}{\partial\bm q}
   \delta\bm q\,dt
   + \left.\frac{\partial\mathcal L}{\partial\dot{\bm q}}
   \delta\bm q\right|_0^T
   - \int_0^T \frac{d}{dt}\left(
   \frac{\partial\mathcal L}{\partial\dot{\bm q}}\right)
   \delta\bm q\,dt \\
   &= \int_0^T \left[
   \frac{\partial\mathcal L}{\partial\bm q}
   - \frac{d}{dt}\left(
   \frac{\partial\mathcal L}{\partial\dot{\bm q}}\right)
   \right]\delta\bm q\,dt = 0.
   \end{aligned}

The endpoint term vanishes because the endpoints are fixed. Since the
variation is otherwise arbitrary, the Euler--Lagrange equations are

.. math::

   \frac{d}{dt}\left(\frac{\partial\mathcal L}{\partial\dot{\bm q}}\right)
   - \frac{\partial\mathcal L}{\partial\bm q} = \bm 0.

Holonomic constraints
---------------------

Suppose the configuration must also satisfy :math:`M` time-independent
constraints, with :math:`M<N`:

.. math::

   \bm F(\bm q)=\bm 0,
   \qquad \bm F(\bm q)\in\mathbb R^M.

Introduce time-dependent Lagrange multipliers
:math:`\bm\lambda(t)\in\mathbb R^M` and add the constraints to the action:

.. math::

   \widetilde S = \int_0^T
   \left[\mathcal L(\bm q,\dot{\bm q})
   + \bm\lambda(t)\cdot\bm F(\bm q)\right]dt.

The constraints change by

.. math::

   \delta\bm F(\bm q)
   = \frac{\partial\bm F}{\partial\bm q}\delta\bm q.

Thus the first variation of the constrained action is

.. math::

   \delta\widetilde S = \int_0^T \left[
   \frac{\partial\mathcal L}{\partial\bm q}\delta\bm q
   + \frac{\partial\mathcal L}{\partial\dot{\bm q}}\delta\dot{\bm q}
   + \bm\lambda\cdot\frac{\partial\bm F}{\partial\bm q}
   \delta\bm q
   + \delta\bm\lambda\cdot\bm F(\bm q)\right]dt.

The constraint Jacobian uses the numerator-layout convention:

.. math::

   \bm J(\bm q) \triangleq \frac{\partial\bm F}{\partial\bm q},
   \qquad \bm J\in\mathbb R^{M\times N}.

Varying the action with respect to :math:`\bm\lambda` gives back
:math:`\bm F(\bm q)=\bm 0`. Varying with respect to :math:`\bm q` gives the
constrained Euler--Lagrange equations:

.. math::

   \delta\widetilde S =
   \int_0^T \left[
   \frac{\partial\mathcal L}{\partial\bm q}
   - \frac{d}{dt}\left(
   \frac{\partial\mathcal L}{\partial\dot{\bm q}}\right)
   + \bm J^T\bm\lambda\right]\delta\bm q\,dt
   + \int_0^T \delta\bm\lambda\cdot\bm F(\bm q)\,dt = 0.

.. math::

   \frac{d}{dt}\left(\frac{\partial\mathcal L}{\partial\dot{\bm q}}\right)
   - \frac{\partial\mathcal L}{\partial\bm q}
   = \bm J(\bm q)^T\bm\lambda,
   \qquad \bm F(\bm q)=\bm 0.

This is a differential-algebraic system: it contains both equations of motion
and algebraic constraints.

Solving for the constrained acceleration
----------------------------------------

Define the mass matrix, the velocity-configuration derivative, and a vector
of Lagrangian derivatives by

.. math::

   M_{ij} = \frac{\partial^2\mathcal L}
   {\partial\dot q_i\,\partial\dot q_j},
   \qquad
   C_{ij} = \frac{\partial^2\mathcal L}
   {\partial\dot q_i\,\partial q_j},
   \qquad
   B_i = \frac{\partial\mathcal L}{\partial q_i}.

Here we assume the mass matrix depends on configuration, but not velocity.
The equations of motion become

.. math::

   \bm M\ddot{\bm q} + \bm C\dot{\bm q} - \bm B
   = \bm J^T\bm\lambda,
   \qquad \bm F(\bm q)=\bm 0.

Differentiating the constraint once and twice gives

.. math::

   \bm J\dot{\bm q}=\bm 0,
   \qquad
   \dot{\bm J}\dot{\bm q}+\bm J\ddot{\bm q}=\bm 0,

where

.. math::

   \dot J_{ij} =
   \frac{\partial^2 F_i}{\partial q_j\,\partial q_k}\dot q_k.

Together, the equations form a saddle-point linear system for acceleration
:math:`\ddot{\bm q}` and the constraint forces :math:`\bm\lambda`:

.. math::

   \begin{bmatrix}
   \bm M & -\bm J^T \\
   \bm J & \bm 0
   \end{bmatrix}
   \begin{bmatrix}
   \ddot{\bm q} \\
   \bm\lambda
   \end{bmatrix}
   =
   \begin{bmatrix}
   -\bm C\dot{\bm q}+\bm B \\
   -\dot{\bm J}\dot{\bm q}
   \end{bmatrix}.

This system can be solved at each step of a numerical integration method.

An expression for the right-hand side
-------------------------------------

Define the unconstrained force and acceleration, and the Delassus matrix:

.. math::

   \bm f \triangleq \bm B-\bm C\dot{\bm q},
   \qquad
   \ddot{\bm q}_{\mathrm{free}} \triangleq \bm M^{-1}\bm f,
   \qquad
   \bm W \triangleq \bm J\bm M^{-1}\bm J^T.

The Delassus matrix maps constraint forces to acceleration along the
constraint directions. The first row of the saddle-point system gives

.. math::

   \ddot{\bm q} = \ddot{\bm q}_{\mathrm{free}}
   + \bm M^{-1}\bm J^T\bm\lambda.

Using the twice-differentiated constraint,
:math:`\bm J\ddot{\bm q}=-\dot{\bm J}\dot{\bm q}`, we obtain

.. math::

   \bm\lambda = -\bm W^{-1}
   \left(\bm J\ddot{\bm q}_{\mathrm{free}}
   + \dot{\bm J}\dot{\bm q}\right).

Substituting this result gives the constrained acceleration:

.. math::

   \ddot{\bm q} = \ddot{\bm q}_{\mathrm{free}}
   - \bm M^{-1}\bm J^T\bm W^{-1}
   \left(\bm J\ddot{\bm q}_{\mathrm{free}}
   + \dot{\bm J}\dot{\bm q}\right).

Adding non-conservative forces
------------------------------

So far, applied forces have been assumed to be conservative and represented
by the Lagrangian :math:`\mathcal L=T-V`, where :math:`T` is kinetic energy
and :math:`V` is potential energy. Forces such as friction, damping, actuator
forces, and aerodynamic loads cannot in general be represented by a scalar
potential.

Let :math:`\bm Q_{\mathrm{nc}}(\bm q,\dot{\bm q},t)\in\mathbb R^N` be the
vector of generalized non-conservative forces. Its virtual work is

.. math::

   \delta W_{\mathrm{nc}}
   = \bm Q_{\mathrm{nc}}(\bm q,\dot{\bm q},t)^T\delta\bm q
   = \sum_{i=1}^N Q_{\mathrm{nc},i}\,\delta q_i.

Each generalized force is paired with one coordinate: a translational
coordinate has a force as its unit, while an angular coordinate has a torque.
The Lagrange--d'Alembert principle adds this virtual work to the variation
of the constrained action:

.. math::

   \delta\widetilde S
   + \int_0^T \delta W_{\mathrm{nc}}\,dt = 0,
   \qquad
   \widetilde S = \int_0^T
   \left[\mathcal L(\bm q,\dot{\bm q})
   + \bm\lambda^T\bm F(\bm q)\right]dt.

The resulting equations are

.. math::

   \frac{d}{dt}\left(\frac{\partial\mathcal L}
   {\partial\dot{\bm q}}\right)
   - \frac{\partial\mathcal L}{\partial\bm q}
   = \bm Q_{\mathrm{nc}} + \bm J(\bm q)^T\bm\lambda,
   \qquad \bm F(\bm q)=\bm 0.

The applied generalized force and the constraint reaction have different
roles: :math:`\bm Q_{\mathrm{nc}}` describes applied loads, while
:math:`\bm J^T\bm\lambda` enforces the constraints.

Using the mass-matrix definitions above gives

.. math::

   \bm M\ddot{\bm q} + \bm C\dot{\bm q} - \bm B
   = \bm Q_{\mathrm{nc}} + \bm J^T\bm\lambda,
   \qquad
   \bm M\ddot{\bm q}
   = \bm f_{\mathrm{nc}} + \bm J^T\bm\lambda,

where

.. math::

   \bm f_{\mathrm{nc}}
   \\triangleq \bm B - \bm C\dot{\bm q} + \bm Q_{\mathrm{nc}}.

The saddle-point system becomes

.. math::

   \begin{bmatrix}
   \bm M & -\bm J^T \\
   \bm J & \bm 0
   \end{bmatrix}
   \begin{bmatrix}
   \ddot{\bm q} \\
   \bm\lambda
   \end{bmatrix}
   =
   \begin{bmatrix}
   \bm B - \bm C\dot{\bm q} + \bm Q_{\mathrm{nc}} \\
   -\dot{\bm J}\dot{\bm q}
   \end{bmatrix}.

The free acceleration includes the applied loads but not the constraint
reaction:

.. math::

   \ddot{\bm q}_{\mathrm{free}}
   = \bm M^{-1}\bm f_{\mathrm{nc}}
   = \bm M^{-1}
   \left(\bm B - \bm C\dot{\bm q} + \bm Q_{\mathrm{nc}}\right).

With this definition, the expressions for the multipliers and constrained
acceleration keep the same form:

.. math::

   \bm\lambda = -\bm W^{-1}
   \left(\bm J\ddot{\bm q}_{\mathrm{free}}
   + \dot{\bm J}\dot{\bm q}\right),
   \qquad
   \ddot{\bm q} = \ddot{\bm q}_{\mathrm{free}}
   - \bm M^{-1}\bm J^T\bm W^{-1}
   \left(\bm J\ddot{\bm q}_{\mathrm{free}}
   + \dot{\bm J}\dot{\bm q}\right).

Generalized forces from physical loads
--------------------------------------

The generalized force may be specified directly if it satisfies the virtual-
work relation. Often, however, forces and torques are given in physical
coordinates. Let a force :math:`\bm f_P` act at a point :math:`P` with
position :math:`\bm r_P(\bm q)`. Its virtual displacement is

.. math::

   \delta\bm r_P
   = \frac{\partial\bm r_P}{\partial\bm q}\delta\bm q.

Define the point-position Jacobian

.. math::

   \bm J_{r,P}(\bm q)
   \\triangleq \frac{\partial\bm r_P}{\partial\bm q}
   \in\mathbb R^{3\times N}.

The force's virtual work can then be written in generalized coordinates:

.. math::

   \delta W_{\bm f_P}
   = \bm f_P^T\delta\bm r_P
   = \left(\bm J_{r,P}^T\bm f_P\right)^T\delta\bm q.

Therefore, the corresponding generalized force is

.. math::

   \bm Q_{\bm f_P} = \bm J_{r,P}^T\bm f_P.

For a pure torque, or free couple, :math:`\bm\tau_B` on a rigid body :math:`B`,
let :math:`\bm\omega_B` be the body's angular velocity and define its
angular-velocity Jacobian by

.. math::

   \bm\omega_B = \bm J_{\omega,B}(\bm q)\dot{\bm q},
   \qquad \bm J_{\omega,B}(\bm q)\in\mathbb R^{3\times N}.

The virtual rotation is :math:`\delta\bm\theta_B
=\bm J_{\omega,B}\delta\bm q`, so the torque's virtual work gives

.. math::

   \delta W_{\bm\tau_B}
   = \bm\tau_B^T\delta\bm\theta_B
   = \left(\bm J_{\omega,B}^T\bm\tau_B\right)^T\delta\bm q,
   \qquad
   \bm Q_{\bm\tau_B} = \bm J_{\omega,B}^T\bm\tau_B.

These equations give the general mechanics formulation. The current
``lagrange_eom`` helper derives forces from the Lagrangian
and constraints; it does not yet accept a separate
:math:`\bm Q_{\mathrm{nc}}` input.