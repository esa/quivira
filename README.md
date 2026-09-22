# Quivira

Symbolic Lagrangian equations of motion, Taylor propagation, and arbitrary-order variational dynamics for pin-jointed lattices.

A legendary land of wealth sought by us, the developers, explorers in modern coding "ways".

Quivira is the place of the unknown, something beyond the familiar map. Heyoka is the voice that travels toward it backwards. Together, they frame exploration not as a straight path to an answer, but as the willingness to seek differently.

## Scope

Quivira develops symbolic mechanical models of pin-jointed lattice structures from a Lagrangian formulation. The resulting equations of motion support Taylor-based state propagation together with variational dynamics at arbitrary order, enabling the analysis of sensitivities and higher-order local behavior.

## Current Work

The current exploratory implementation lives in [`prova.ipynb`](prova.ipynb). It uses Python together with `heyoka` and `numpy` to construct symbolic quantities and test the supporting linear-algebra operations.

## Tests

The test suite uses Python's standard-library `unittest` discovery. Run it from
the `quivira` Conda environment with:

```console
conda run -n quivira python -c "import quivira; quivira.test.run_test_suite()"
```

New tests should be added to modules named `test_*.py` inside the `quivira`
package.