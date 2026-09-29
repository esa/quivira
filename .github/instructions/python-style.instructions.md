---
applyTo: "**/*.py"
description: "Use when editing Python files in this workspace."
---

# Python style for quivira code

Use the existing codebase of quivira as the primary style reference. Preserve its compact scientific-programming character: explicit mathematical code, minimal abstraction, and documentation focused on physical and numerical meaning.

## Docstrings

- Use triple-quoted docstrings immediately below modules, classes, functions, and methods.
- Start with a concise description of the object or operation.
- Use ordinary prose to state the mathematical formulation, physical meaning, and relevant assumptions.
- Document public function arguments, exceptions, notes, and return values using the following section order when applicable: `Args:`, `Raises:`, `Notes:`, and `Returns:`.
- Format each argument as `name (:class:`type`, optional): Description. State the default in the description. Do not add emphasis markers around the argument name; Sphinx renders it in bold automatically.
- Format exceptions as `ExceptionType: Condition that raises the exception.`
- Format return values as `:class:`type`: Description.`
- Use Sphinx roles such as `:class:` and `~module.name` for documented types and project objects.
- Use single backticks for short code identifiers and double backticks for longer inline code, vector layouts, and expressions, for example ``[x, y, z, vx, vy, vz]``.
- For displayed mathematical equations in docstrings, use the Sphinx `.. math::` directive with a blank line before the directive, a blank line after it, and the equation indented by four spaces. Leave a blank line after the equation before any following prose or section. Use `:math:` for inline expressions.
- When documenting a mathematical model, introduce the equations with a short sentence, then define symbols and conventions in bullet points, including state ordering, parameter ordering, units, and reference frames when relevant.
- Preserve LaTeX structure inside `.. math::` blocks, including aligned systems, cases, matrices, and line breaks. Escape backslashes correctly in Python docstrings.
- Use mathematical notation where it makes a scientific definition clearer.
- State units, reference frames, epochs, coordinate conventions, component ordering, sign conventions, and normalization when they are relevant.
- Describe numerical assumptions, validity limits, and special cases when they affect interpretation or use.
- Keep docstrings concise. Explain the model and interface; do not narrate obvious implementation details.

### Docstring pattern

Use this structure for public functions, omitting sections that do not apply:

```python
def function_name(argument, count: int = 1):
	"""Short description of the operation.

	Args:
		argument (:class:`type`): Description of the argument.

		count (:class:`int`, optional): Description. Default is 1.

	Raises:
		ValueError: Description of when the error is raised.

	Notes:
		Additional mathematical, physical, or numerical context.

	Returns:
		:class:`type`: Description of the returned value.
	"""
```

For private helpers or very simple functions, use only the concise summary when the full structure would add no useful information.

## Code formatting

- Use four spaces for indentation and no tabs.
- Keep imports at the top of the module and preserve the import grouping and ordering used nearby.
- Use `snake_case` for functions, methods, arguments, and local variables.
- Do not use `PascalCase` for anything, including classes and constants.
- Use compact but readable expressions for scientific and numerical code.
- Prefer NumPy arrays and vectorized operations when they make the implementation clearer or more efficient.
- Make array dimensions, component ordering, and matrix conventions explicit in names, docstrings, or short comments when they are not self-evident.
- Use parentheses to format multiline expressions, calls, and mathematical formulas clearly.
- Match the nearby style for line breaks, alignment, whitespace, and blank lines.
- Avoid unnecessary wrappers, properties, helper layers, type annotations, or defensive abstractions when they are not established in the surrounding code.
- Do not rename existing public objects or alter established API conventions without a specific reason.
- Do not apply broad formatting changes to unrelated legacy code.

## Comments and numerical code

- Use comments to describe the overall logic, construction, and flow of a non-trivial algorithm, especially how its stages or transformations fit together.
- In non-trivial tests, use short comments to record the test logic at its meaningful stages: what mathematical or numerical object is constructed, how the implementation under test is evaluated, and what reference result or invariant is used for comparison.
- Place a short orienting comment before a complex block when it helps explain the purpose of the block and how it connects to the surrounding computation.
- Use additional comments for mathematical intent, numerical details, algorithmic choices, or non-obvious implementation constraints.
- Do not comment on syntax or restate what a clear individual line of code already says.
- Keep equations and transformations close to the code that implements them.
- State numerical tolerances, convergence criteria, singular cases, and domain restrictions where they affect results.
- Preserve established terminology for orbital elements, epochs, time of flight, gravitational parameters, thrust, mass, and reference frames.

## Editing rule

Before editing a file, inspect adjacent classes and functions in the same module. Reuse their docstring layout, naming, import style, spacing, notation, and degree of explanation.

Local consistency within quivira takes precedence over generic Python, documentation, linting, or formatting conventions.

## Generic coding philosophy

The first rule is: Write like a human and for human readability, only disobey this rule when there is a clear computational advantage, but then state this in a comment above the line.
Do not add too many blank lines.
Before writing a line ask yourself:
- Is this line necessary?
- Does this line follow the established style and conventions within quivira?
- Is there a clearer or comutationally more efficient way to achieve the same result?
- Does this line maintain readability and understandability for future maintainers?
- Is this not defensive coding?
- Is this not over-engineering?
- Am I writing like a human and for human readability, not like a machine? Example: Keep for loops simple, with the code contained in it after it, never put a for loop on the end of the line

## Mandatory readability rules

- Prefer explicit loops over compact one-line comprehensions when the loop body is not trivially obvious.
- Do not hide algorithmic structure inside a one-line return expression.
- Keep control flow readable and scan-friendly; do not pack loops, conditionals, or matrix assembly into a single expression.
- Put docstrings immediately under public functions, classes, and modules.
- Preserve the mathematical structure of the implementation; do not compress symbolic derivations into dense expressions when a short block is clearer.
- If a short expression is less readable than a loop, prefer the readable version.

## Anti-patterns to avoid

- Avoid returning a list comprehension that assembles a matrix in one dense expression when a short loop is clearer.
- Avoid `return [ ... for ... ]` for symbolic derivations that are easier to inspect by row.
- Avoid terse comments in place of docstrings.