"""Names vulture cannot see being used but that are intentional (public API, hooks,
framework callbacks). One bare name or ``_.attribute`` per line, as
``vulture --make-whitelist`` writes them, each with a comment saying why.

At the gate's minimum confidence (80) vulture reports only names it is sure are unused
(unused imports, arguments, variables, unreachable code), so this file starts empty. Real
dead code that cannot be removed yet goes in ``tools/vulture_baseline.json`` instead.
"""
