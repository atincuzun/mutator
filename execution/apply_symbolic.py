"""Apply symbolic modifications to AST nodes."""

import ast
from typing import Dict

from mutator import config
from .constants import ARG_TO_POS_MAP


def apply_symbolic_modification(node: ast.Call, mod: Dict) -> None:
    """Apply symbolic expression modifications."""
    modified = False
    symbolic_expr = mod['symbolic_expression']

    # Parse the symbolic expression into an AST node
    try:
        expr_node = ast.parse(symbolic_expr, mode='eval').body
    except SyntaxError:
        if config.DEBUG_MODE:
            print(f"  > ERROR: Could not parse symbolic expression '{symbolic_expr}'")
        return

    # Replace keyword argument
    for kw in node.keywords:
        if kw.arg == mod['arg_name']:
            kw.value = expr_node
            modified = True
            if config.DEBUG_MODE:
                print(f"  > Modified keyword arg '{mod['arg_name']}' to symbolic expression: {symbolic_expr}")
            break

    # Or positional
    if not modified and mod['arg_name'] in ARG_TO_POS_MAP:
        pos_index = ARG_TO_POS_MAP[mod['arg_name']]
        if pos_index < len(node.args):
            old_val = getattr(node.args[pos_index], 'value', 'unknown')
            node.args[pos_index] = expr_node
            if config.DEBUG_MODE:
                print(f"  > Modified positional arg {pos_index} ('{mod['arg_name']}') from ~{old_val} to symbolic expression: {symbolic_expr}")
            modified = True

    if not modified and config.DEBUG_MODE:
        print(f"  > WARNING: Could not find argument '{mod['arg_name']}' to modify at this location.")
