import ast
from . import config
from . import utils
from .execution.constants import ARG_TO_POS_MAP
from .execution.apply_dimension import apply_dimension_modification
from .execution.apply_symbolic import apply_symbolic_modification
from .execution.apply_activation import apply_activation_modification
from .execution.apply_layer_type import apply_layer_type_modification
from .execution.apply_architectural import apply_architectural_modification
from .execution.apply_spatial import apply_kernel_size_modification, apply_stride_modification

class CodeMutator(ast.NodeTransformer):
    
    ARG_TO_POS_MAP = ARG_TO_POS_MAP

    def __init__(self, code_string: str):
        self.tree = ast.parse(code_string)
        self.modifications = []
        if config.DEBUG_MODE:
            print("[CodeMutator] Initialized.")

    def _validate_conv_params(self, arg_name: str, new_value: int, node: ast.Call):
        """Validate convolution parameters to prevent invalid depthwise config"""
        current_params = {}
        
        # Collect current parameters
        for kw in node.keywords:
            current_params[kw.arg] = getattr(kw.value, 'value', None)
        for i, arg in enumerate(node.args):
            if i == 0: current_params['in_channels'] = getattr(arg, 'value', None)
            if i == 1: current_params['out_channels'] = getattr(arg, 'value', None)
            if i == 5: current_params['groups'] = getattr(arg, 'value', 1)
        
        # Handle depthwise convolution constraint
        if (arg_name == 'out_channels' 
                and current_params.get('groups', 1) == current_params.get('in_channels')
                and new_value != current_params.get('in_channels')):
            return {'groups': 1}  # Convert to standard convolution
        return {}

    def schedule_modification(self, location: dict, arg_name: str, new_value):
        """Schedule a dimension-based modification (backward compatibility)."""
        if location and new_value is not None:
            mod = {
                'type': 'dimension',
                'location': location,
                'arg_name': arg_name,
                'new_value': new_value
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled dimension modification: {mod}")

    def schedule_symbolic_modification(self, location: dict, arg_name: str, symbolic_expression: str):
        """Schedule a symbolic expression modification."""
        if location and symbolic_expression:
            mod = {
                'type': 'symbolic',
                'location': location,
                'arg_name': arg_name,
                'symbolic_expression': symbolic_expression
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled symbolic modification: {mod}")

    def schedule_activation_modification(self, location: dict, new_activation: str):
        """Schedule an activation function modification."""
        if location and new_activation:
            mod = {
                'type': 'activation',
                'location': location,
                'new_activation': new_activation
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled activation modification: {mod}")

    def schedule_layer_type_modification(self, location: dict, new_layer_type: str, params: dict):
        """Schedule a layer type modification."""
        if location and new_layer_type:
            mod = {
                'type': 'layer_type',
                'location': location,
                'new_layer_type': new_layer_type,
                'params': params
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled layer type modification: {mod}")

    def schedule_architectural_modification(self, location: dict, architectural_type: str, params: dict):
        """Schedule an architectural modification for high-level network structure."""
        if location and architectural_type:
            mod = {
                'type': 'architectural',
                'location': location,
                'architectural_type': architectural_type,
                'params': params
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled architectural modification: {mod}")

    def schedule_kernel_size_modification(self, location: dict, new_kernel_size: int):
        """Schedule a kernel size modification."""
        if location and new_kernel_size:
            mod = {
                'type': 'kernel_size',
                'location': location,
                'new_kernel_size': new_kernel_size
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled kernel size modification: {mod}")

    def schedule_stride_modification(self, location: dict, new_stride: int):
        """Schedule a stride modification."""
        if location and new_stride:
            mod = {
                'type': 'stride',
                'location': location,
                'new_stride': new_stride
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled stride modification: {mod}")

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        
        for mod in self.modifications:
            loc = mod['location']
            if (hasattr(node, 'lineno') and node.lineno == loc['lineno'] and
                    hasattr(node, 'col_offset') and node.col_offset == loc['col_offset']):
                
                if config.DEBUG_MODE:
                    print(f"[CodeMutator] Found AST Call node at Line {loc['lineno']}, Col {loc['col_offset']} for {mod['type']} modification.")

                if mod['type'] == 'dimension':
                    apply_dimension_modification(node, mod)
                elif mod['type'] == 'symbolic':
                    apply_symbolic_modification(node, mod)
                elif mod['type'] == 'activation':
                    apply_activation_modification(node, mod)
                elif mod['type'] == 'layer_type':
                    apply_layer_type_modification(node, mod)
                elif mod['type'] == 'architectural':
                    apply_architectural_modification(node, mod)
                elif mod['type'] == 'kernel_size':
                    apply_kernel_size_modification(node, mod)
                elif mod['type'] == 'stride':
                    apply_stride_modification(node, mod)

        return node

    def visit_List(self, node: ast.List):
        """Visit List nodes to handle architectural mutations like block_setting."""
        self.generic_visit(node)
        
        for mod in self.modifications:
            if mod['type'] == 'architectural':
                loc = mod['location']
                if (hasattr(node, 'lineno') and node.lineno == loc['lineno'] and
                        hasattr(node, 'col_offset') and node.col_offset == loc['col_offset']):
                    
                    if config.DEBUG_MODE:
                        print(f"[CodeMutator] Found AST List node at Line {loc['lineno']}, Col {loc['col_offset']} for architectural modification.")
                    
                    self._apply_architectural_modification(node, mod)
        
        return node

    def get_modified_code(self) -> str:
        if config.DEBUG_MODE:
            print("[CodeMutator] Applying all scheduled modifications to AST.")
        modified_tree = self.visit(self.tree)
        return ast.unparse(modified_tree)
