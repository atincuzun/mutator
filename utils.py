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
        candidate = None
        for node in ast.walk(self.source_ast):
            if isinstance(node, ast.Call) and hasattr(node, 'lineno') and node.lineno == lineno:
                if candidate is None or node.col_offset > candidate.col_offset:
                    candidate = node
        return candidate

    @staticmethod
    def _make_patched_init(original_init):
        @wraps(original_init)
        def patched_init(module_instance, *args, **kwargs):
            original_init(module_instance, *args, **kwargs)
            tracer = ModuleSourceTracer._instance
            if tracer:
                try:
                    frame = inspect.stack()[1]
                    lineno = frame.lineno
                    call_node = tracer._find_call_node_at_line(lineno)
                    if call_node:
                        module_instance._source_location = {
                            "lineno": call_node.lineno,
                            "end_lineno": getattr(call_node, 'end_lineno', call_node.lineno),
                            "col_offset": call_node.col_offset,
                            "end_col_offset": getattr(call_node, 'end_col_offset', -1)
                        }
                except (IndexError, RuntimeError):
                    pass
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