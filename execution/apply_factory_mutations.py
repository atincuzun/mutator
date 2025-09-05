"""
Apply factory-based mutations for block-based architectures.
Handles mutations targeting factory method parameters instead of individual layers.
"""

import ast
from typing import Dict, Any, Optional
from mutator import config


class FactoryMutationApplier:
    """
    Applies mutations to factory method calls for block-based architectures.
    """
    
    def apply_factory_mutation(self, node: ast.Call, mod: Dict) -> bool:
        """
        Apply a factory method mutation by modifying parameter expressions.
        
        Args:
            node: The AST call node representing the factory method
            mod: The mutation specification
            
        Returns:
            True if mutation was applied successfully, False otherwise
        """
        modified = False
        
        if config.DEBUG_MODE:
            print(f"  > Attempting factory mutation for {mod['target_param']} with {mod['new_symbolic']}")
        
        # Check if this is the correct factory call to mutate
        if not self._matches_factory_target(node, mod):
            return False
        
        # Apply mutation to the target parameter
        modified |= self._modify_factory_parameter(node, mod)
        
        return modified
    
    def _matches_factory_target(self, node: ast.Call, mod: Dict) -> bool:
        """
        Check if this AST node matches the factory target from the mutation plan.
        """
        # For method calls (e.g., self._make_layer)
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            target_method_name = mod.get('factory_method_name', '')
            return method_name == target_method_name
        
        # For function calls
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
            target_method_name = mod.get('factory_method_name', '')
            return func_name == target_method_name
        
        return False
    
    def _modify_factory_parameter(self, node: ast.Call, mod: Dict) -> bool:
        """
        Modify a specific parameter in a factory method call.
        """
        target_param = mod['target_param']
        new_expr_str = mod['new_symbolic']
        
        # Parse the new symbolic expression
        try:
            new_expr = ast.parse(new_expr_str, mode='eval').body
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"  > Failed to parse symbolic expression: {e}")
            return False
        
        # Try to modify keyword arguments first
        for kw in node.keywords:
            if kw.arg == target_param:
                old_value = self._expr_to_string(kw.value)
                kw.value = new_expr
                if config.DEBUG_MODE:
                    print(f"  > Modified factory parameter '{target_param}' from {old_value} to {new_expr_str}")
                return True
        
        # Try to modify positional arguments
        if target_param.startswith('arg_'):
            try:
                arg_index = int(target_param.split('_')[1])
                if arg_index < len(node.args):
                    old_value = self._expr_to_string(node.args[arg_index])
                    node.args[arg_index] = new_expr
                    if config.DEBUG_MODE:
                        print(f"  > Modified factory positional arg {arg_index} from {old_value} to {new_expr_str}")
                    return True
            except (ValueError, IndexError):
                pass
        
        if config.DEBUG_MODE:
            print(f"  > Could not find parameter '{target_param}' in factory call")
        
        return False
    
    def _expr_to_string(self, expr) -> str:
        """
        Convert an AST expression to a string representation.
        """
        if isinstance(expr, ast.Constant):
            return str(expr.value)
        elif isinstance(expr, ast.Name):
            return expr.id
        elif isinstance(expr, ast.BinOp):
            left = self._expr_to_string(expr.left)
            op = self._op_to_string(expr.op)
            right = self._expr_to_string(expr.right)
            return f"({left} {op} {right})"
        else:
            return str(expr)
    
    def _op_to_string(self, op) -> str:
        """
        Convert AST operator to string representation.
        """
        op_map = {
            ast.Add: '+',
            ast.Sub: '-', 
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '//',
            ast.Mod: '%'
        }
        return op_map.get(type(op), '?')


# Global instance for convenience
factory_applier = FactoryMutationApplier()