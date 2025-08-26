"""
Context analysis utilities for understanding code structure and parameter availability.
"""

import ast
import re
from typing import List, Optional, Any

from mutator import config


def is_top_level_net_context(frame_info, source_code: str) -> bool:
    """
    Determine if the current stack frame is within a top-level Net class.
    Net classes are identified by name pattern and inheritance from nn.Module.
    
    Args:
        frame_info: Stack frame information
        source_code: Source code to analyze
        
    Returns:
        True if frame is in a top-level Net class context
    """
    try:
        if frame_info.code_context:
            code_line = frame_info.code_context[0].strip()

            # Check if we're in a class definition that matches top-level patterns
            for pattern in config.TOP_LEVEL_CLASS_PATTERNS:
                if f'class {pattern}' in code_line and 'nn.Module' in code_line:
                    return True

            # Check if we're in the __init__ method of a top-level class
            if frame_info.function == '__init__':
                # Parse the source to find the class containing this __init__
                class_node = _find_class_containing_line(source_code, frame_info.lineno)
                if class_node and any(pattern in class_node.name for pattern in config.TOP_LEVEL_CLASS_PATTERNS):
                    return True

    except (AttributeError, IndexError, TypeError):
        pass
    return False


def is_block_definition_context(frame_info, source_code: str) -> bool:
    """
    Determine if the current stack frame is within a block definition or helper function.
    Block definitions include helper functions and custom block classes.
    
    Args:
        frame_info: Stack frame information
        source_code: Source code to analyze
        
    Returns:
        True if frame is in a block definition context
    """
    function_name = frame_info.function

    # Check if function name matches helper patterns from config
    for pattern in config.HELPER_FUNCTION_PATTERNS:
        if pattern.lower() in function_name.lower():
            return True

    # Check if we're in a class that contains block-related patterns
    try:
        if frame_info.code_context:
            code_line = frame_info.code_context[0].strip()

            # Check for class definitions that contain block patterns
            if 'class ' in code_line and any(pattern in code_line for pattern in config.HELPER_FUNCTION_PATTERNS):
                return True

            # Check if we're in a method of a block class
            if frame_info.function != '__init__' and frame_info.function != '<module>':
                class_node = _find_class_containing_line(source_code, frame_info.lineno)
                if class_node and any(pattern in class_node.name for pattern in config.HELPER_FUNCTION_PATTERNS):
                    return True

    except (AttributeError, IndexError, TypeError):
        pass

    return False


def _find_class_containing_line(source_code: str, lineno: int) -> Optional[ast.ClassDef]:
    """
    Find the AST class node that contains the given line number.
    
    Args:
        source_code: Source code to parse
        lineno: Line number to search for
        
    Returns:
        AST ClassDef node containing the line, or None if not found
    """
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if (isinstance(node, ast.ClassDef) and 
                node.lineno <= lineno <= getattr(node, 'end_lineno', float('inf'))):
                return node
    except (SyntaxError, AttributeError):
        pass
    return None


def get_available_parameters(call_node: ast.Call, source_code: str) -> List[str]:
    """
    Extract available parameter names from the current function/class context
    for use in symbolic mutations.
    
    Args:
        call_node: AST call node to analyze
        source_code: Source code containing the call
        
    Returns:
        List of available parameter names
    """
    parameters = []
    try:
        # Find the function or class containing this call
        tree = ast.parse(source_code)
        containing_node = None

        for node in ast.walk(tree):
            if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and
                node.lineno <= call_node.lineno <= getattr(node, 'end_lineno', float('inf'))):
                containing_node = node
                break

        if containing_node:
            # Extract argument names from function/class
            if isinstance(containing_node, ast.FunctionDef):
                for arg in containing_node.args.args:
                    parameters.append(arg.arg)
                if containing_node.args.vararg:
                    parameters.append(containing_node.args.vararg.arg)
                if containing_node.args.kwarg:
                    parameters.append(containing_node.args.kwarg.arg)

            # For classes, look at __init__ method
            elif isinstance(containing_node, ast.ClassDef):
                for item in containing_node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        for arg in item.args.args:
                            if arg.arg != 'self':
                                parameters.append(arg.arg)
                        if item.args.vararg:
                            parameters.append(item.args.vararg.arg)
                        if item.args.kwarg:
                            parameters.append(item.args.kwarg.arg)
                        break

    except (SyntaxError, AttributeError):
        pass

    return parameters


def find_call_node_at_line(source_code: str, lineno: int) -> Optional[ast.Call]:
    """
    Find the AST call node at the given line number.
    
    Args:
        source_code: Source code to parse
        lineno: Line number to search for
        
    Returns:
        AST Call node at the line, or None if not found
    """
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and 
                hasattr(node, 'lineno') and 
                node.lineno == lineno):
                return node
    except (SyntaxError, AttributeError):
        pass
    return None


def extract_variable_names(code_line: str) -> List[str]:
    """
    Extract variable names from a line of code.
    
    Args:
        code_line: Line of code to analyze
        
    Returns:
        List of variable names found in the code
    """
    # Simple regex to find identifier patterns
    pattern = r'\\b[a-zA-Z_][a-zA-Z0-9_]*\\b'
    variables = re.findall(pattern, code_line)
    
    # Filter out common keywords and built-ins
    keywords = {'and', 'or', 'not', 'if', 'else', 'elif', 'for', 'while', 'def', 'class',
                'import', 'from', 'as', 'try', 'except', 'finally', 'with', 'lambda',
                'True', 'False', 'None', 'return', 'yield', 'break', 'continue', 'pass'}
    
    return [var for var in variables if var not in keywords]
