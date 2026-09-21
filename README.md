# Quivira

Symbolic Lagrangian equations of motion, Taylor propagation, and arbitrary-order variational dynamics for pin-jointed lattices.

## Scope

Quivira develops symbolic mechanical models of pin-jointed lattice structures from a Lagrangian formulation. The resulting equations of motion support Taylor-based state propagation together with variational dynamics at arbitrary order, enabling the analysis of sensitivities and higher-order local behavior.

## Current Work

The current exploratory implementation lives in [`prova.ipynb`](prova.ipynb). It uses Python together with `heyoka` and `numpy` to construct symbolic quantities and test the supporting linear-algebra operations.