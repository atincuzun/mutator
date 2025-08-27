"""
Source code tracing utilities for tracking neural network module instantiations.

This module provides the ModuleSourceTracer class which can trace the source
location of PyTorch module instantiations during model creation.
"""

import os
import inspect
import ast
import torch.nn as nn
from functools import wraps
import re
from typing import Dict, List, Optional, Any

from mutator import config


class ModuleSourceTracer:
    """
    Traces the source location of PyTorch module instantiations.
    
    This class patches the __init__ methods of target PyTorch modules to
    capture their source location information during model construction.
    """
    
    _instance = None
    TARGET_MODULES = [
        nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm,
        nn.ReLU, nn.GELU, nn.ELU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid, nn.SiLU,
        nn.GroupNorm, nn.InstanceNorm2d, nn.MaxPool2d, nn.AvgPool2d
    ]

    def __init__(self, source_code: str):
        """
        Initialize the source tracer with the model's source code.
        
        Args:
            source_code: The source code of the model being traced
        """
        self.source_code = source_code
        self.source_ast = ast.parse(self.source_code)
        self.source_map = {}
        self._original_inits = {}
        ModuleSourceTracer._instance = self

    def _find_call_node_at_line(self, lineno: int) -> Optional[ast.Call]:
        """
        Enhanced call node finding that prioritizes assignment targets.
        This helps distinguish between helper function calls and direct instantiations.
        """
        candidates = []
        for node in ast.walk(self.source_ast):
            if isinstance(node, ast.Call) and hasattr(node, 'lineno') and node.lineno == lineno:
                candidates.append(node)

        if not candidates:
            return None

        # If multiple candidates, prefer ones that are assignment targets (like self.conv1 = ...)
        assignment_candidates = []
        for node in ast.walk(self.source_ast):
            if isinstance(node, ast.Assign) and hasattr(node, 'lineno') and node.lineno == lineno:
                if isinstance(node.value, ast.Call):
                    assignment_candidates.append(node.value)

        if assignment_candidates:
            # Prefer assignment targets - these are usually the actual module instantiations
            candidate = max(assignment_candidates, key=lambda n: n.col_offset)
            if config.DEBUG_MODE:
                print(f"[SourceTracer] Found assignment-based call node at line {lineno}, col {candidate.col_offset}")
            return candidate

        # Fallback to rightmost candidate
        candidate = max(candidates, key=lambda n: n.col_offset)
        if config.DEBUG_MODE:
            print(f"[SourceTracer] Found general call node at line {lineno}, col {candidate.col_offset}")
        return candidate

    def _is_helper_function_frame(self, frame_info) -> bool:
        """
        Detect if a stack frame represents a helper function definition.
        Helper functions typically:
        1. Have names like conv3x3, conv1x1, make_layer, etc.
        2. Return nn.Module instances directly
        3. Are defined at module level (not inside classes)
        """
        function_name = frame_info.function

        # If helper function mutations are allowed, we don't need to filter them out
        if config.ALLOW_HELPER_FUNCTION_MUTATIONS:
            return False

        # Check if function name matches helper patterns from config
        for pattern in config.HELPER_FUNCTION_PATTERNS:
            if pattern in function_name.lower():
                if config.DEBUG_MODE:
                    print(f"[SourceTracer] Detected helper function by name pattern: {function_name}")
                return True

        # Check if the frame's code context suggests it's a helper function
        try:
            if frame_info.code_context:
                code_line = frame_info.code_context[0].strip()
                # Look for direct returns of nn.Module instantiations
                if 'return nn.' in code_line or 'return torch.nn.' in code_line:
                    if config.DEBUG_MODE:
                        print(f"[SourceTracer] Detected helper function by return pattern: {code_line}")
                    return True
        except (AttributeError, IndexError):
            pass

        # Check the function definition itself from the source code
        try:
            lines = self.source_code.split('\\n')
            # Find the function definition line
            for i, line in enumerate(lines):
                if f'def {function_name}(' in line and i + 1 <= len(lines):
                    # Look for return statements in the next few lines
                    for j in range(i + 1, min(i + 10, len(lines))):
                        if 'return nn.' in lines[j] or 'return torch.nn.' in lines[j]:
                            if config.DEBUG_MODE:
                                print(f"[SourceTracer] Detected helper function by source analysis: {function_name}")
                            return True
                    break
        except (IndexError, AttributeError):
            pass

        return False

    def _is_direct_instantiation_call(self, frame_info) -> bool:
        """
        Check if a frame represents a direct nn.Module instantiation (not through helper).
        """
        try:
            if frame_info.code_context:
                code_line = frame_info.code_context[0].strip()
                # Direct instantiation patterns
                direct_patterns = ['nn.', 'torch.nn.']
                return any(pattern in code_line for pattern in direct_patterns)
        except (AttributeError, IndexError):
            pass
        return False

    @staticmethod
    def _make_patched_init(original_init):
        """Create a patched version of a module's __init__ method that captures source location."""
        @wraps(original_init)
        def patched_init(module_instance, *args, **kwargs):
            original_init(module_instance, *args, **kwargs)
            tracer = ModuleSourceTracer._instance
            if tracer:
                try:
                    # Smart stack walking to find the actual call site
                    target_frame = None
                    stack_frames = inspect.stack()

                    if config.DEBUG_MODE:
                        print(f"[SourceTracer] Stack analysis for {type(module_instance).__name__}:")
                        for i, frame in enumerate(stack_frames[:8]):  # Show first 8 frames
                            print(f"  Frame {i}: {frame.function} at {frame.filename}:{frame.lineno}")

                    # Start from frame 1 (caller of this patched_init)
                    for i in range(1, min(len(stack_frames), 15)):  # Increased search depth to 15
                        frame_info = stack_frames[i]

                        # Skip only true internal Python frames, but allow model methods like __init__, _make_layer
                        if ('site-packages' in frame_info.filename or
                            'lib/python' in frame_info.filename or
                            frame_info.filename.endswith('utils.py')):  # Skip our own utils.py
                            if config.DEBUG_MODE:
                                print(f"[SourceTracer] Skipping internal frame: {frame_info.function}")
                            continue

                        # Skip if this frame is inside a helper function (when helper mutations disabled)
                        if not config.ALLOW_HELPER_FUNCTION_MUTATIONS:
                            if tracer._is_helper_function_frame(frame_info):
                                if config.DEBUG_MODE:
                                    print(f"[SourceTracer] Skipping helper function frame: {frame_info.function} at line {frame_info.lineno}")
                                continue

                        # For symbolic mutations, we want to capture ALL call sites, not just direct instantiations
                        # This allows context-aware mutation decisions later
                        if frame_info.code_context and any('nn.' in line or 'torch.nn.' in line for line in frame_info.code_context):
                            target_frame = frame_info
                            if config.DEBUG_MODE:
                                print(f"[SourceTracer] Found target frame: {frame_info.function} at line {frame_info.lineno}")
                            break

                        # Also capture frames that look like module assignments (self.conv1 = ...)
                        if frame_info.code_context and any('self.' in line and '=' in line for line in frame_info.code_context):
                            target_frame = frame_info
                            if config.DEBUG_MODE:
                                print(f"[SourceTracer] Found assignment frame: {frame_info.function} at line {frame_info.lineno}")
                            break

                    if target_frame:
                        lineno = target_frame.lineno
                        call_node = tracer._find_call_node_at_line(lineno)
                        if call_node:
                            # Convert absolute path to relative path from project root
                            absolute_path = target_frame.filename
                            try:
                                relative_path = os.path.relpath(absolute_path, "f:/mutator_env")
                            except ValueError:
                                # Fallback to absolute path if relative path conversion fails
                                relative_path = absolute_path
                            
                            module_instance._source_location = {
                                "lineno": call_node.lineno,
                                "end_lineno": getattr(call_node, 'end_lineno', call_node.lineno),
                                "col_offset": call_node.col_offset,
                                "end_col_offset": getattr(call_node, 'end_col_offset', -1),
                                "filename": relative_path  # Use relative path instead of absolute
                            }
                            if config.DEBUG_MODE:
                                print(f"[SourceTracer] Captured source location: line {call_node.lineno}, col {call_node.col_offset}, file: {relative_path}")
                        else:
                            if config.DEBUG_MODE:
                                print(f"[SourceTracer] No AST call node found at line {lineno}")
                    else:
                        if config.DEBUG_MODE:
                            print("[SourceTracer] No suitable target frame found in stack")

                except (IndexError, RuntimeError) as e:
                    if config.DEBUG_MODE:
                        print(f"[SourceTracer] Error during stack walking: {e}")
        return patched_init

    def __enter__(self):
        """Context manager entry - patch all target module __init__ methods."""
        for module_cls in self.TARGET_MODULES:
            if module_cls not in self._original_inits:
                self._original_inits[module_cls] = module_cls.__init__
                module_cls.__init__ = self._make_patched_init(module_cls.__init__)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore original __init__ methods."""
        for module_cls, original_init in self._original_inits.items():
            module_cls.__init__ = original_init
        self._original_inits.clear()
        ModuleSourceTracer._instance = None

    def create_source_map(self, model: nn.Module) -> Dict[str, Any]:
        """
        Create a mapping from module names to their source locations.
        
        Args:
            model: The PyTorch model to create source map for
            
        Returns:
            Dictionary mapping module names to source location information
        """
        if config.DEBUG_MODE: 
            print("[SourceTracer] Creating source map...")
        for name, module in model.named_modules():
            if hasattr(module, '_source_location'):
                self.source_map[name] = module._source_location
                if config.DEBUG_MODE: 
                    print(f"  - Found location for '{name}': {module._source_location}")
                del module._source_location
        return self.source_map
