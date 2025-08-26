"""Apply dimension modifications to AST nodes."""

import ast
from typing import Dict

from mutator import config
from .constants import ARG_TO_POS_MAP
from .validators import validate_conv_params


def apply_dimension_modification(node: ast.Call, mod: Dict) -> None:
    """Apply dimension-based modifications with convolution validation."""
    modified = False
    additional_changes = {}

    # Validate convolution parameters before modification
    if mod['arg_name'] in ['in_channels', 'out_channels']:
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'Conv2d':
            additional_changes = validate_conv_params(
                mod['arg_name'], mod['new_value'], node
            )

    # Apply primary modification
    for kw in node.keywords:
        if kw.arg == mod['arg_name']:
            old_val = getattr(kw.value, 'value', 'some_variable')
            kw.value = ast.Constant(value=mod['new_value'])
            if config.DEBUG_MODE:
                print(f"  > Modified keyword arg '{mod['arg_name']}' from ~{old_val} to {mod['new_value']}.")
            modified = True
            break

    if not modified and mod['arg_name'] in ARG_TO_POS_MAP:
        pos_index = ARG_TO_POS_MAP[mod['arg_name']]
        if pos_index < len(node.args):
            old_val = getattr(node.args[pos_index], 'value', 'some_variable')
            node.args[pos_index] = ast.Constant(value=mod['new_value'])
            if config.DEBUG_MODE:
                print(f"  > Modified positional arg {pos_index} ('{mod['arg_name']}') from ~{old_val} to {mod['new_value']}.")
            modified = True

    # Apply additional changes (e.g., groups parameter)
    for param, value in additional_changes.items():
        param_found = False
        for kw in node.keywords:
            if kw.arg == param:
                kw.value = ast.Constant(value=value)
                param_found = True
                if config.DEBUG_MODE:
                    print(f"  > Adjusted {param} to {value} for depthwise conv constraint")
                break

        if not param_found:
            node.keywords.append(ast.keyword(arg=param, value=ast.Constant(value=value)))
            if config.DEBUG_MODE:
                print(f"  > Added {param}={value} for depthwise conv constraint")

    if not modified and config.DEBUG_MODE:
        print(f"  > WARNING: Could not find argument '{mod['arg_name']}' to modify at this location.")
