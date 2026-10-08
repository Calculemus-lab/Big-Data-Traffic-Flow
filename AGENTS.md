# Development guidance

## Code clarity

- Use precise names that make a value's meaning and role clear without relying only on nearby type hints.
- Prefer explicit, straightforward code. Keep implementations concise without hiding their steps.
- Use the narrowest suitable scope for functions, types, and constants. Define shared integrity values once and reuse them in their smallest possible scope. Keep obvious defaults, formatting details, and trivial values inline.
- Prefer immutable, named data structures (e.g. NamedTuples)when their fields describe a stable record. Use mutable structures when mutation is part of the design.
- Prefer lookup tables and declarative transformations when they make the behavior clearer. Keep conditional control flow when it expresses meaningful decisions.

## Types and dependencies

- Use precise annotations for public interfaces and nontrivial functions. Prefer current language typing syntax and concrete domain types over generic strings, untyped containers, or broad types.
- Use types to make invalid combinations difficult to express. Keep runtime validation at external input and trust boundaries; do not duplicate guarantees already provided by types or libraries.
- Prefer established library features over custom parsing, conversion, validation, caching, and data-access machinery. Add abstractions only when they make a repeated or complex operation simpler to use and maintain.
- Avoid unnecessary casts, compatibility layers, wrappers, and suppression comments. Resolve the underlying typing or design issue when that can be done simply.

## Documentation

- Write complete, neutral docstrings for nontrivial modules, classes, and functions. Explain purpose, arguments, return values, and behavior that is not clear from names and types.
- Use comments to explain intent, invariants, or non-obvious steps. Place a brief comment immediately before a substantial code block when it helps a reader understand why the block exists. Do not narrate obvious syntax.
- Keep names and explanations concrete and consistent. Define specialized terms before relying on them, and avoid shorthand formulation that leaves the reader to infer what a value represents.

## Changes and verification

- Keep changes focused on the requested behavior and update affected callers, examples, and documentation together when an interface changes instead of keeping compatability code around, unless for a good reason.
- Prefer tests of behavior and project-owned logic. Keep tests focused on important guarantees rather than duplicating library coverage.
- Use the repository's configured formatters, linters, type checkers, and test commands. Do not introduce alternate tooling or suppressions without clear need or instruction.
