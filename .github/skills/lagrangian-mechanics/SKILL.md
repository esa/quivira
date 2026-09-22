---
name: lagrangian-mechanics
description: "Use when writing, editing, reviewing, or commenting on Lagrangian-mechanics mathematics in Python files or Jupyter notebook code or Markdown cells, including Euler-Lagrange equations, actions, generalized coordinates, velocities, variations, holonomic constraints, multipliers, and related numerical implementations."
---

# Lagrangian Mechanics in Quivira

Before coding or documenting Lagrangian-mechanics mathematics, read `doc/latex/quivira.tex` and use it as the authoritative source for mathematical notation, variable names, definitions, assumptions, and derivation logic.

## Implementation and commentary

- Preserve the notation established in `doc/latex/quivira.tex` in Python identifiers, docstrings, comments, notebook Markdown, and notebook code whenever the language permits.
- Keep the code and explanatory mathematics consistent with the document's definitions of generalized coordinates, velocities, variations, constraints, the Lagrangian, and the action.
- When translating a mathematical expression into code, state any necessary representation convention, such as vector shape, coordinate ordering, or discretization, close to the implementation when it is not self-evident.
- Do not silently normalize notation, alter sign conventions, or change a derivation's assumptions.

## Deviations and inconsistencies

When a deviation from `doc/latex/quivira.tex` is necessary, or when the document conflicts with existing code or contains an inconsistency, explicitly tag the user in the response or code-review finding as:

`@user Lagrangian-mechanics deviation: <concise description, reason, and affected notation or logic>.`

Do not proceed silently past a material inconsistency. State the source-document convention, the conflicting convention, and the chosen resolution.
