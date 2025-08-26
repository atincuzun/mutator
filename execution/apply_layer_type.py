"""Apply layer type modifications to AST nodes."""

import ast
from mutator import config


def apply_layer_type_modification(node: ast.Call, mod: dict) -> None:
    """Apply layer type modifications."""
    if isinstance(node.func, ast.Attribute):
        if (isinstance(node.func.value, ast.Name) and node.func.value.id == 'nn') or \
           (isinstance(node.func.value, ast.Attribute) and 
            isinstance(node.func.value.value, ast.Name) and 
            node.func.value.value.id == 'torch' and node.func.value.attr == 'nn'):

            old_layer_type = node.func.attr
            node.func.attr = mod['new_layer_type']

            # Update parameters based on layer type
            if mod['new_layer_type'] == 'GroupNorm':
                node.args = []
                node.keywords = [
                    ast.keyword(arg='num_groups', value=ast.Constant(value=mod['params']['num_groups'])),
                    ast.keyword(arg='num_channels', value=ast.Constant(value=mod['params']['num_channels']))
                ]
            elif mod['new_layer_type'] == 'BatchNorm2d':
                node.args = []
                node.keywords = [
                    ast.keyword(arg='num_features', value=ast.Constant(value=mod['params']['num_features']))
                ]
            elif mod['new_layer_type'] in ['AdaptiveMaxPool2d', 'AdaptiveAvgPool2d']:
                output_size = mod['params'].get('output_size', (7, 7))
                node.args = []
                node.keywords = [
                    ast.keyword(arg='output_size', value=ast.Tuple(
                        elts=[ast.Constant(value=output_size[0]), ast.Constant(value=output_size[1])],
                        ctx=ast.Load()
                    ))
                ]

            if config.DEBUG_MODE:
                print(f"  > Modified layer type from {old_layer_type} to {mod['new_layer_type']}")
                if not config.ALLOW_HELPER_FUNCTION_MUTATIONS:
                    print("  > Direct instantiation mode: only nn.Module calls mutated")

        elif not config.ALLOW_HELPER_FUNCTION_MUTATIONS:
            if config.DEBUG_MODE:
                func_name = getattr(node.func, 'id', 'unknown')
                print(f"  > Skipping potential helper function call: {func_name}")
