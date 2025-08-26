"""Validation utilities for AST-based code mutations."""

import ast

from typing import Dict, Any


def validate_conv_params(arg_name: str, new_value: int, node: ast.Call) -> Dict[str, Any]:
    """Validate convolution parameters to prevent invalid depthwise config.
    Returns a dict of additional parameter changes (e.g., {'groups': 1}).
    """
    current_params: Dict[str, Any] = {}

    # Collect current parameters
    for kw in node.keywords:
        current_params[kw.arg] = getattr(kw.value, 'value', None)
    for i, arg in enumerate(node.args):
        if i == 0:
            current_params['in_channels'] = getattr(arg, 'value', None)
        if i == 1:
            current_params['out_channels'] = getattr(arg, 'value', None)
        if i == 5:
            current_params['groups'] = getattr(arg, 'value', 1)

    # Handle depthwise convolution constraint
    if (
        arg_name == 'out_channels'
        and current_params.get('groups', 1) == current_params.get('in_channels')
        and new_value != current_params.get('in_channels')
    ):
        return {'groups': 1}  # Convert to standard convolution
    return {}
