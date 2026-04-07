"""
Grader functions for CodeCompleteEnv.

Each grader accepts the completed source code (and optionally extra
context) and returns a score in [0.0, 1.0].

Graders are **deterministic** — no randomness, no network calls.
"""

from __future__ import annotations

import ast
import re
import textwrap
from typing import Any, Dict, List


# ======================================================================
# 1. AST Validity
# ======================================================================

def grade_ast_validity(code: str) -> float:
    """Return 1.0 if *code* is valid Python, else 0.0.

    A small partial-credit bump (0.3) is given when the code is *almost*
    parseable — i.e. only a single ``SyntaxError`` at the very end,
    suggesting the completion is truncated rather than garbage.
    """
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError as exc:
        # Partial credit if error is on the last line (truncated output)
        lines = code.strip().splitlines()
        if exc.lineno and exc.lineno >= len(lines):
            return 0.3
        return 0.0


# ======================================================================
# 2. Type Correctness (heuristic)
# ======================================================================

def grade_type_correct(code: str) -> float:
    """Heuristic type-correctness score.

    * Full credit  (1.0) — function with a ``return`` statement.
    * Partial      (0.6) — valid AST but no function or no return.
    * Zero         (0.0) — parse failure.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0

    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not functions:
        return 0.6  # Code is valid but has no functions — partial

    for func in functions:
        if any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(func)):
            return 1.0

    return 0.6


# ======================================================================
# 3. Style Match
# ======================================================================

_SNAKE_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def grade_style_match(code: str) -> float:
    """PEP-8-ish style score.

    Checks:
      * Line length ≤ 120
      * No trailing whitespace
      * 4-space indent multiples
      * ``snake_case`` function names

    Returns a score in [0.0, 1.0].
    """
    penalties = 0.0
    checks = 0

    for line in code.splitlines():
        checks += 1
        if len(line) > 120:
            penalties += 0.05
        if line.rstrip() != line:
            penalties += 0.02
        stripped = line.lstrip()
        if stripped and line != stripped:
            indent = len(line) - len(stripped)
            if indent % 4 != 0:
                penalties += 0.03

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                checks += 1
                if not _SNAKE_RE.match(node.name):
                    penalties += 0.08
    except SyntaxError:
        pass

    return max(0.0, min(1.0, 1.0 - penalties))


# ======================================================================
# 4. Test Pass Rate
# ======================================================================

def grade_test_pass(code: str, test_cases: List[Dict[str, Any]]) -> float:
    """Execute *test_cases* against *code* and return fraction that pass.

    Each test case is a dict::

        {
            "function": "<func_name>",
            "args":     [<positional args>],
            "kwargs":   {<keyword args>},    # optional
            "expected": <value>,
        }

    Equality is checked with ``==``; for floats an ``abs`` tolerance of
    1e-6 is used.
    """
    if not test_cases:
        return 0.5  # no tests → neutral

    passed = 0

    for tc in test_cases:
        try:
            ns: Dict[str, Any] = {}
            exec(code, ns)  # noqa: S102  — controlled env data only

            func_name = tc["function"]
            if func_name not in ns:
                continue

            result = ns[func_name](*tc.get("args", []), **tc.get("kwargs", {}))
            expected = tc["expected"]

            if result == expected:
                passed += 1
            elif isinstance(expected, float) and isinstance(result, (int, float)):
                if abs(result - expected) < 1e-6:
                    passed += 1
        except Exception:
            pass

    return passed / len(test_cases)


# ======================================================================
# 5. Simulated User Signal
# ======================================================================

def grade_user_signal(code: str, original_code: str) -> float:
    """Simulate user acceptance based on several quality proxies.

    The heuristic rewards:
      * AST validity
      * Reasonable length (not too short, not too long)
      * Presence of docstrings
      * Not being a destructive completion
    """
    score = 0.5

    # AST parse bonus / penalty
    try:
        ast.parse(code)
        score += 0.15
    except SyntaxError:
        score -= 0.20

    # Conciseness bonus
    n_lines = len(code.splitlines())
    if n_lines <= 25:
        score += 0.10
    elif n_lines > 60:
        score -= 0.10

    # Docstring bonus
    if '"""' in code or "'''" in code:
        score += 0.05

    # Destructive-completion penalty
    if len(code.strip()) < len(original_code.strip()) * 0.25:
        score -= 0.30

    return max(0.0, min(1.0, score))


# ======================================================================
# 6. Refactor-specific grader (for the hard task)
# ======================================================================

def grade_refactor(
    code: str,
    old_name: str,
    new_name: str,
    expected_replacements: int,
) -> float:
    """Score a rename-refactor completion.

    * Checks that all occurrences of *old_name* (as a word) are gone.
    * Checks that *new_name* appears at least *expected_replacements* times.
    """
    old_pat = re.compile(r"\b" + re.escape(old_name) + r"\b")
    new_pat = re.compile(r"\b" + re.escape(new_name) + r"\b")

    remaining_old = len(old_pat.findall(code))
    new_count = len(new_pat.findall(code))

    if remaining_old == 0 and new_count >= expected_replacements:
        return 1.0

    if new_count == 0:
        return 0.0

    replacement_ratio = min(1.0, new_count / expected_replacements)
    removal_ratio = max(0.0, 1.0 - remaining_old / max(expected_replacements, 1))

    return max(0.0, min(1.0, replacement_ratio * 0.5 + removal_ratio * 0.5))
