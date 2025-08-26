"""Apply architectural modifications (e.g., ConvNeXT block configs)."""

import ast
from typing import Dict
from mutator import config


def apply_architectural_modification(node, mod: Dict) -> None:
    """Apply architectural modifications like changing block configurations."""
    architectural_type = mod['architectural_type']
    params = mod['params']

    if architectural_type == 'block_setting':
        if isinstance(node, ast.List):
            new_configs = params.get('new_configs', [])
            new_elements = []
            for config_tuple in new_configs:
                call_node = ast.Call(
                    func=ast.Name(id='CNBlockConfig', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=config_tuple[0]),
                        ast.Constant(value=config_tuple[1]) if config_tuple[1] is not None else ast.Constant(value=None),
                        ast.Constant(value=config_tuple[2])
                    ],
                    keywords=[]
                )
                new_elements.append(call_node)
            node.elts = new_elements
            if config.DEBUG_MODE:
                print(f"  > Modified block_setting configuration: {new_configs}")
        elif isinstance(node, ast.Call) and hasattr(node.func, 'id') and node.func.id == 'CNBlockConfig':
            if 'input_channels' in params and len(node.args) > 0:
                node.args[0] = ast.Constant(value=params['input_channels'])
            if 'out_channels' in params and len(node.args) > 1:
                node.args[1] = ast.Constant(value=params['out_channels']) if params['out_channels'] is not None else ast.Constant(value=None)
            if 'num_layers' in params and len(node.args) > 2:
                node.args[2] = ast.Constant(value=params['num_layers'])
            if config.DEBUG_MODE:
                print(f"  > Modified CNBlockConfig: {params}")

    elif architectural_type == 'depth_multiplier':
        multiplier = params.get('multiplier', 1.0)
        if isinstance(node, ast.List):
            for element in node.elts:
                if (
                    isinstance(element, ast.Call)
                    and hasattr(element.func, 'id')
                    and element.func.id == 'CNBlockConfig'
                    and len(element.args) > 2
                ):
                    current_layers = element.args[2].value if hasattr(element.args[2], 'value') else 3
                    new_layers = max(1, int(current_layers * multiplier))
                    element.args[2] = ast.Constant(value=new_layers)
            if config.DEBUG_MODE:
                print(f"  > Applied depth multiplier {multiplier} to block configurations")

    elif architectural_type == 'width_multiplier':
        multiplier = params.get('multiplier', 1.0)
        if isinstance(node, ast.List):
            for element in node.elts:
                if (
                    isinstance(element, ast.Call)
                    and hasattr(element.func, 'id')
                    and element.func.id == 'CNBlockConfig'
                ):
                    if len(element.args) > 0 and hasattr(element.args[0], 'value'):
                        current_in = element.args[0].value
                        new_in = max(1, int(current_in * multiplier))
                        element.args[0] = ast.Constant(value=new_in)
                    if len(element.args) > 1 and getattr(element.args[1], 'value', None) is not None:
                        current_out = element.args[1].value
                        new_out = max(1, int(current_out * multiplier))
                        element.args[1] = ast.Constant(value=new_out)
            if config.DEBUG_MODE:
                print(f"  > Applied width multiplier {multiplier} to channel dimensions")
