"""Apply activation modifications to AST nodes."""

import ast
from mutator import config


def apply_activation_modification(node: ast.Call, mod: dict) -> None:
    """Apply activation function modifications."""
    # Check if this is a call to nn.SomeActivation()
    if isinstance(node.func, ast.Attribute):
        if (isinstance(node.func.value, ast.Name) and node.func.value.id == 'nn') or \
           (isinstance(node.func.value, ast.Attribute) and 
            isinstance(node.func.value.value, ast.Name) and 
            node.func.value.value.id == 'torch' and node.func.value.attr == 'nn'):

            # Replace the activation function name
            old_activation = node.func.attr
            new_activation_name = mod['new_activation']

            # Handle special name mappings
            if new_activation_name == 'Swish':
                new_activation_name = 'SiLU'  # PyTorch uses SiLU for Swish

            node.func.attr = new_activation_name

            # Handle special cases for activation parameters
            if mod['new_activation'] == 'GELU':
                # GELU doesn't have inplace parameter, remove it if present
                node.keywords = [kw for kw in node.keywords if kw.arg != 'inplace']
            elif mod['new_activation'] in ('Tanh', 'Sigmoid'):
                node.keywords = [kw for kw in node.keywords if kw.arg != 'inplace']
            elif mod['new_activation'] == 'LeakyReLU':
                # Ensure negative_slope parameter exists
                has_negative_slope = any(kw.arg == 'negative_slope' for kw in node.keywords)
                if not has_negative_slope:
                    node.keywords.append(ast.keyword(arg='negative_slope', value=ast.Constant(value=0.01)))

            if config.DEBUG_MODE:
                print(f"  > Modified activation from {old_activation} to {new_activation_name}")
                if not config.ALLOW_HELPER_FUNCTION_MUTATIONS:
                    print("  > Direct instantiation mode: only nn.Module calls mutated")

        # When helper mutations are disabled, don't mutate helper function calls
        elif not config.ALLOW_HELPER_FUNCTION_MUTATIONS:
            if config.DEBUG_MODE:
                func_name = getattr(node.func, 'id', 'unknown')
                print(f"  > Skipping potential helper function call: {func_name}")
