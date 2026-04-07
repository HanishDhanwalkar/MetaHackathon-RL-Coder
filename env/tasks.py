"""
Task definitions for CodeCompleteEnv.

Three difficulty levels — **easy**, **medium**, **hard** — each with
concrete Python code, a cursor marker, expected completions, unit-test
cases, and knowledge-graph seed nodes.

Every task is a plain ``dict`` so it serialises trivially.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ======================================================================
# EASY — Complete a missing expression
# ======================================================================

_EASY_TASK: Dict[str, Any] = {
    "name": "easy_expression_complete",
    "difficulty": "easy",
    "description": "Complete a missing return expression in a simple function",
    "initial_code": (
        'def calculate_area(length: float, width: float) -> float:\n'
        '    """Calculate the area of a rectangle.\n'
        '\n'
        '    Args:\n'
        '        length: The length of the rectangle.\n'
        '        width:  The width of the rectangle.\n'
        '\n'
        '    Returns:\n'
        '        The area of the rectangle.\n'
        '    """\n'
        '    return __CURSOR__\n'
    ),
    "cursor_file": "geometry.py",
    "cursor_line": 11,
    "cursor_marker": "__CURSOR__",
    "expected_completion": "length * width",
    "test_cases": [
        {"function": "calculate_area", "args": [5.0, 3.0], "expected": 15.0},
        {"function": "calculate_area", "args": [0.0, 10.0], "expected": 0.0},
        {"function": "calculate_area", "args": [2.5, 4.0], "expected": 10.0},
        {"function": "calculate_area", "args": [1.0, 1.0], "expected": 1.0},
        {"function": "calculate_area", "args": [7.0, 7.0], "expected": 49.0},
    ],
    "open_files": ["geometry.py", "shapes.py"],
    "kg_nodes": [
        {"name": "calculate_area", "kind": "function",
         "context": "Computes rectangle area from length and width"},
        {"name": "length", "kind": "variable",
         "context": "Rectangle length parameter (float)"},
        {"name": "width", "kind": "variable",
         "context": "Rectangle width parameter (float)"},
        {"name": "calculate_perimeter", "kind": "function",
         "context": "Computes rectangle perimeter: 2*(length+width)"},
        {"name": "Rectangle", "kind": "class",
         "context": "Rectangle shape with length and width attributes"},
    ],
    "max_steps": 8,
}

# ======================================================================
# MEDIUM — Generate a full function body
# ======================================================================

_MEDIUM_TASK: Dict[str, Any] = {
    "name": "medium_function_body",
    "difficulty": "medium",
    "description": "Generate a complete function body from signature and docstring",
    "initial_code": (
        'def flatten_list(nested_list: list) -> list:\n'
        '    """Flatten a nested list into a single flat list.\n'
        '\n'
        '    Recursively flattens a list that may contain nested lists\n'
        '    of arbitrary depth into a single-level list.\n'
        '\n'
        '    Args:\n'
        '        nested_list: A list that may contain nested lists.\n'
        '\n'
        '    Returns:\n'
        '        A flat list containing all non-list elements.\n'
        '\n'
        '    Examples:\n'
        '        >>> flatten_list([1, [2, 3], [4, [5, 6]]])\n'
        '        [1, 2, 3, 4, 5, 6]\n'
        '        >>> flatten_list([[1, 2], [3, [4, [5]]]])\n'
        '        [1, 2, 3, 4, 5]\n'
        '        >>> flatten_list([])\n'
        '        []\n'
        '    """\n'
        '    __CURSOR__\n'
    ),
    "cursor_file": "list_utils.py",
    "cursor_line": 20,
    "cursor_marker": "__CURSOR__",
    "expected_completion": (
        "result = []\n"
        "    for item in nested_list:\n"
        "        if isinstance(item, list):\n"
        "            result.extend(flatten_list(item))\n"
        "        else:\n"
        "            result.append(item)\n"
        "    return result"
    ),
    "test_cases": [
        {"function": "flatten_list", "args": [[1, [2, 3], [4, [5, 6]]]],
         "expected": [1, 2, 3, 4, 5, 6]},
        {"function": "flatten_list", "args": [[[1, 2], [3, [4, [5]]]]],
         "expected": [1, 2, 3, 4, 5]},
        {"function": "flatten_list", "args": [[]], "expected": []},
        {"function": "flatten_list", "args": [[1, 2, 3]], "expected": [1, 2, 3]},
        {"function": "flatten_list", "args": [[[[1]]]], "expected": [1]},
        {"function": "flatten_list", "args": [["a", ["b", ["c"]]]],
         "expected": ["a", "b", "c"]},
    ],
    "open_files": ["list_utils.py", "tests/test_list_utils.py"],
    "kg_nodes": [
        {"name": "flatten_list", "kind": "function",
         "context": "Recursively flattens nested lists into a single list"},
        {"name": "nested_list", "kind": "variable",
         "context": "Input list that may contain arbitrarily nested sublists"},
        {"name": "isinstance", "kind": "function",
         "context": "Built-in function — checks if object is instance of a type"},
        {"name": "list.extend", "kind": "function",
         "context": "Appends all elements from an iterable to the list"},
        {"name": "list.append", "kind": "function",
         "context": "Appends a single element to the end of the list"},
        {"name": "merge_sorted_lists", "kind": "function",
         "context": "Merges two pre-sorted lists into one sorted list"},
    ],
    "max_steps": 8,
}

# ======================================================================
# HARD — Refactor (rename) across multiple usages
# ======================================================================

_HARD_TASK: Dict[str, Any] = {
    "name": "hard_refactor",
    "difficulty": "hard",
    "description": (
        "Refactor code by renaming a function and updating all "
        "references consistently across the file"
    ),
    "initial_code": (
        'def calc(items):\n'
        '    """Calculate total price of items."""\n'
        '    total = 0\n'
        '    for item in items:\n'
        '        total += item["price"] * item["quantity"]\n'
        '    return total\n'
        '\n'
        '\n'
        'def apply_discount(items, discount_rate):\n'
        '    """Apply discount if total exceeds threshold."""\n'
        '    total = calc(items)\n'
        '    if total > 100:\n'
        '        return total * (1 - discount_rate)\n'
        '    return total\n'
        '\n'
        '\n'
        'def generate_receipt(items, customer_name):\n'
        '    """Generate a receipt string for the customer."""\n'
        '    total = calc(items)\n'
        '    receipt_lines = [\n'
        '        f"Customer: {customer_name}",\n'
        '        f"Items: {len(items)}",\n'
        '        f"Subtotal: ${calc(items):.2f}",\n'
        '        f"Total: ${total:.2f}",\n'
        '    ]\n'
        '    return "\\n".join(receipt_lines)\n'
        '\n'
        '\n'
        'def check_budget(items, budget):\n'
        '    """Check if items are within budget."""\n'
        '    return calc(items) <= budget\n'
        '\n'
        '\n'
        '# --- main execution ---\n'
        'sample_items = [\n'
        '    {"name": "Widget", "price": 25.0, "quantity": 2},\n'
        '    {"name": "Gadget", "price": 15.0, "quantity": 3},\n'
        ']\n'
        '\n'
        'order_total = calc(sample_items)\n'
        'is_affordable = check_budget(sample_items, 100)\n'
        'receipt = generate_receipt(sample_items, "Alice")\n'
        'discounted = apply_discount(sample_items, 0.1)\n'
    ),
    "cursor_file": "shopping.py",
    "cursor_line": 1,
    "cursor_marker": None,  # full-file replacement
    "refactor_target": {
        "old_name": "calc",
        "new_name": "calculate_total_price",
        "expected_replacements": 6,
    },
    "expected_completion": None,  # graded via reference checking
    "test_cases": [
        {"function": "calculate_total_price",
         "args": [[{"price": 25.0, "quantity": 2},
                    {"price": 15.0, "quantity": 3}]],
         "expected": 95.0},
        {"function": "calculate_total_price",
         "args": [[{"price": 10.0, "quantity": 1}]],
         "expected": 10.0},
        {"function": "calculate_total_price",
         "args": [[]], "expected": 0},
        {"function": "apply_discount",
         "args": [[{"price": 60.0, "quantity": 2}], 0.1],
         "expected": 108.0},
        {"function": "check_budget",
         "args": [[{"price": 50.0, "quantity": 1}], 100],
         "expected": True},
    ],
    "open_files": ["shopping.py", "tests/test_shopping.py", "main.py"],
    "kg_nodes": [
        {"name": "calc", "kind": "function",
         "context": "Calculates total price of items (to be renamed)"},
        {"name": "apply_discount", "kind": "function",
         "context": "Applies percentage discount when total > 100"},
        {"name": "generate_receipt", "kind": "function",
         "context": "Builds a receipt string for a customer"},
        {"name": "check_budget", "kind": "function",
         "context": "Returns True when item total is within budget"},
        {"name": "items", "kind": "variable",
         "context": "List of dicts with 'price' and 'quantity' keys"},
        {"name": "calculate_total_price", "kind": "function",
         "context": "Preferred descriptive name for the calc function"},
        {"name": "ShoppingCart", "kind": "class",
         "context": "Shopping-cart container class"},
    ],
    "max_steps": 8,
}

# ======================================================================
# Public helpers
# ======================================================================

TASKS: Dict[str, Dict[str, Any]] = {
    _EASY_TASK["name"]: _EASY_TASK,
    _MEDIUM_TASK["name"]: _MEDIUM_TASK,
    _HARD_TASK["name"]: _HARD_TASK,
}


def get_task(task_name: str) -> Dict[str, Any]:
    """Return task config by name, or raise ``ValueError``."""
    if task_name not in TASKS:
        available = ", ".join(TASKS.keys())
        raise ValueError(
            f"Unknown task '{task_name}'. Available tasks: {available}"
        )
    return TASKS[task_name]


def list_tasks() -> List[str]:
    """Return all registered task names."""
    return list(TASKS.keys())
