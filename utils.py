import os
import json
import time
import inspect
import ast
import torch.nn as nn
from functools import wraps
import re

from mutator import config

class ModuleSourceTracer:
    _instance = None
    TARGET_MODULES = [nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm, 
                      nn.ReLU, nn.GELU, nn.ELU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid, nn.SiLU,
                      nn.GroupNorm, nn.InstanceNorm2d, nn.MaxPool2d, nn.AvgPool2d]

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

    def _is_direct_instantiation_call(self, frame_info):
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
                            module_instance._source_location = {
                                "lineno": call_node.lineno,
                                "end_lineno": getattr(call_node, 'end_lineno', call_node.lineno),
                                "col_offset": call_node.col_offset,
                                "end_col_offset": getattr(call_node, 'end_col_offset', -1),
                                "filename": target_frame.filename  # Add filename for source code reading
                            }
                            if config.DEBUG_MODE:
                                print(f"[SourceTracer] Captured source location: line {call_node.lineno}, col {call_node.col_offset}, file: {target_frame.filename}")
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
    
    # Add debug output to confirm plan saving
    if config.DEBUG_MODE:
        print(f"[Utils] Saving mutation plan to: {filepath}")
        print(f"[Utils] Plan content: {json.dumps(report, indent=2)}")
    
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Exception):
                return str(obj)
            return json.JSONEncoder.default(self, obj)

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=4, cls=CustomEncoder)
    if config.DEBUG_MODE:
        print(f"[Utils] Successfully saved mutation plan to: {filepath}")

    # Verify the file was actually created
    if config.DEBUG_MODE:
        if os.path.exists(filepath):
            print(f"[Utils] CONFIRMED: Mutation plan file exists at {filepath}")
            # Check file size to ensure it's not empty
            file_size = os.path.getsize(filepath)
            print(f"[Utils] Mutation plan file size: {file_size} bytes")
        else:
            print(f"[Utils] ERROR: Mutation plan file was not created at {filepath}")
            # List directory contents to debug
            if os.path.exists(output_dir):
                print(f"[Utils] Directory contents of {output_dir}: {os.listdir(output_dir)}")
            else:
                print(f"[Utils] ERROR: Output directory {output_dir} does not exist")


def is_top_level_net_context(frame_info, source_code: str) -> bool:
    """
    Determine if the current stack frame is within a top-level Net class.
    Net classes are identified by name pattern and inheritance from nn.Module.
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


def _find_class_containing_line(source_code: str, lineno: int):
    """
    Find the AST class node that contains the given line number.
    """
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.lineno <= lineno <= getattr(node, 'end_lineno', float('inf')):
                return node
    except (SyntaxError, AttributeError):
        pass
    return None


def get_available_parameters(call_node, source_code: str) -> list:
    """
    Extract available parameter names from the current function/class context
    for use in symbolic mutations.
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
import os
import json
import time
import inspect
import ast
import torch.nn as nn
from functools import wraps
import re

from mutator import config

class ModuleSourceTracer:
    _instance = None
    TARGET_MODULES = [nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm, 
                      nn.ReLU, nn.GELU, nn.ELU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid, nn.SiLU,
                      nn.GroupNorm, nn.InstanceNorm2d, nn.MaxPool2d, nn.AvgPool2d]

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

    def _is_direct_instantiation_call(self, frame_info):
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
                            module_instance._source_location = {
                                "lineno": call_node.lineno,
                                "end_lineno": getattr(call_node, 'end_lineno', call_node.lineno),
                                "col_offset": call_node.col_offset,
                                "end_col_offset": getattr(call_node, 'end_col_offset', -1),
                                "filename": target_frame.filename  # Add filename for source code reading
                            }
                            if config.DEBUG_MODE:
                                print(f"[SourceTracer] Captured source location: line {call_node.lineno}, col {call_node.col_offset}, file: {target_frame.filename}")
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
    
    # Add debug output to confirm plan saving
    if config.DEBUG_MODE:
        print(f"[Utils] Saving mutation plan to: {filepath}")
        print(f"[Utils] Plan content: {json.dumps(report, indent=2)}")
    
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Exception):
                return str(obj)
            return json.JSONEncoder.default(self, obj)

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=4, cls=CustomEncoder)
    if config.DEBUG_MODE:
        print(f"[Utils] Successfully saved mutation plan to: {filepath}")

    # Verify the file was actually created
    if config.DEBUG_MODE:
        if os.path.exists(filepath):
            print(f"[Utils] CONFIRMED: Mutation plan file exists at {filepath}")
            # Check file size to ensure it's not empty
            file_size = os.path.getsize(filepath)
            print(f"[Utils] Mutation plan file size: {file_size} bytes")
        else:
            print(f"[Utils] ERROR: Mutation plan file was not created at {filepath}")
            # List directory contents to debug
            if os.path.exists(output_dir):
                print(f"[Utils] Directory contents of {output_dir}: {os.listdir(output_dir)}")
            else:
                print(f"[Utils] ERROR: Output directory {output_dir} does not exist")


def is_top_level_net_context(frame_info, source_code: str) -> bool:
    """
    Determine if the current stack frame is within a top-level Net class.
    Net classes are identified by name pattern and inheritance from nn.Module.
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


def _find_class_containing_line(source_code: str, lineno: int):
    """
    Find the AST class node that contains the given line number.
    """
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.lineno <= lineno <= getattr(node, 'end_lineno', float('inf')):
                return node
    except (SyntaxError, AttributeError):
        pass
    return None


def get_available_parameters(call_node, source_code: str) -> list:
    """
    Extract available parameter names from the current function/class context
    for use in symbolic mutations.
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
