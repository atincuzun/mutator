import os
import json
import time
import inspect
import ast
import torch.nn as nn
from functools import wraps

import config

class ModuleSourceTracer:
    _instance = None
    TARGET_MODULES = [nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm]

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.source_ast = ast.parse(self.source_code)
        self.source_map = {}
        self._original_inits = {}
        ModuleSourceTracer._instance = self

    def _find_call_node_at_line(self, lineno):
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

    def _is_helper_function_frame(self, frame_info):
        """
        Detect if a stack frame represents a helper function definition.
        Helper functions typically:
        1. Have names like conv3x3, conv1x1, make_layer, etc.
        2. Return nn.Module instances directly
        3. Are defined at module level (not inside classes)
        """
        function_name = frame_info.function
        
        # Common helper function name patterns
        helper_patterns = [
            'conv1x1', 'conv3x3', 'conv5x5', 'conv7x7',  # Convolution helpers
            'make_layer', 'make_block', 'make_stage',      # Layer builders
            'build_', 'create_', 'get_',                   # Factory functions
            'downsample', 'upsample',                      # Sampling helpers
        ]
        
        # Check if function name matches helper patterns
        for pattern in helper_patterns:
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
            lines = self.source_code.split('\n')
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

    @staticmethod
    def _make_patched_init(original_init):
        @wraps(original_init)
        def patched_init(module_instance, *args, **kwargs):
            original_init(module_instance, *args, **kwargs)
            tracer = ModuleSourceTracer._instance
            if tracer:
                try:
                    # Smart stack walking to find the actual call site, not helper functions
                    target_frame = None
                    stack_frames = inspect.stack()
                    
                    if config.DEBUG_MODE:
                        print(f"[SourceTracer] Stack analysis for {type(module_instance).__name__}:")
                        for i, frame in enumerate(stack_frames[:8]):  # Show first 8 frames
                            print(f"  Frame {i}: {frame.function} at {frame.filename}:{frame.lineno}")
                    
                    # Start from frame 1 (caller of this patched_init)
                    for i in range(1, min(len(stack_frames), 10)):  # Limit search depth
                        frame_info = stack_frames[i]
                        
                        # Skip if this frame is inside a helper function
                        if tracer._is_helper_function_frame(frame_info):
                            if config.DEBUG_MODE:
                                print(f"[SourceTracer] Skipping helper function frame: {frame_info.function} at line {frame_info.lineno}")
                            continue
                        
                        # Additional check: ensure this frame is in a class method (__init__)
                        if frame_info.function not in ['__init__', '<module>']:
                            if config.DEBUG_MODE:
                                print(f"[SourceTracer] Skipping non-init frame: {frame_info.function}")
                            continue
                        
                        # This should be the actual call site (like self.conv1 = conv3x3(...))
                        target_frame = frame_info
                        if config.DEBUG_MODE:
                            print(f"[SourceTracer] Found target frame: {frame_info.function} at line {frame_info.lineno}")
                        break
                    
                    if target_frame:
                        lineno = target_frame.lineno
                        call_node = tracer._find_call_node_at_line(lineno)
                        if call_node:
                            module_instance._source_location = {
                                "lineno": call_node.lineno,
                                "end_lineno": getattr(call_node, 'end_lineno', call_node.lineno),
                                "col_offset": call_node.col_offset,
                                "end_col_offset": getattr(call_node, 'end_col_offset', -1)
                            }
                            if config.DEBUG_MODE:
                                print(f"[SourceTracer] Captured source location: line {call_node.lineno}, col {call_node.col_offset}")
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
        for module_cls in self.TARGET_MODULES:
            if module_cls not in self._original_inits:
                self._original_inits[module_cls] = module_cls.__init__
                module_cls.__init__ = self._make_patched_init(module_cls.__init__)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module_cls, original_init in self._original_inits.items():
            module_cls.__init__ = original_init
        self._original_inits.clear()
        ModuleSourceTracer._instance = None

    def create_source_map(self, model: nn.Module):
        if config.DEBUG_MODE: print("[SourceTracer] Creating source map...")
        for name, module in model.named_modules():
            if hasattr(module, '_source_location'):
                self.source_map[name] = module._source_location
                if config.DEBUG_MODE: print(f"  - Found location for '{name}': {module._source_location}")
                del module._source_location
        return self.source_map

def save_plan_to_file(model_name: str, status: str, plan: dict, details: dict):
    output_dir = os.path.join(config.PLANS_OUTPUT_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.time_ns()
    filename = f"{status}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    report = { "model_name": model_name, "status": status, "timestamp_ns": timestamp, "plan": plan, "details": details }
    
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Exception):
                return str(obj)
            return json.JSONEncoder.default(self, obj)

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=4, cls=CustomEncoder)