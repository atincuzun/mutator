import ast
import config

class CodeMutator(ast.NodeTransformer):
    ARG_TO_POS_MAP = {
        'in_channels': 0, 'out_channels': 1,
        'in_features': 0, 'out_features': 1,
        'num_features': 0,
        'normalized_shape': 0
    }

    def __init__(self, code_string: str):
        self.tree = ast.parse(code_string)
        self.modifications = []
        if config.DEBUG_MODE:
            print("[CodeMutator] Initialized.")

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

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        
        for mod in self.modifications:
            loc = mod['location']
            if (hasattr(node, 'lineno') and node.lineno == loc['lineno'] and
                    hasattr(node, 'col_offset') and node.col_offset == loc['col_offset']):
                
                if config.DEBUG_MODE:
                    print(f"[CodeMutator] Found AST Call node at Line {loc['lineno']}, Col {loc['col_offset']} for {mod['type']} modification.")

                if mod['type'] == 'dimension':
                    self._apply_dimension_modification(node, mod)
                elif mod['type'] == 'activation':
                    self._apply_activation_modification(node, mod)
                elif mod['type'] == 'layer_type':
                    self._apply_layer_type_modification(node, mod)

        return node

    def _apply_dimension_modification(self, node: ast.Call, mod: dict):
        """Apply dimension-based modifications (original logic)."""
        modified = False
        for kw in node.keywords:
            if kw.arg == mod['arg_name']:
                old_val = getattr(kw.value, 'value', 'some_variable')
                kw.value = ast.Constant(value=mod['new_value'])
                if config.DEBUG_MODE:
                    print(f"  > Modified keyword arg '{mod['arg_name']}' from ~{old_val} to {mod['new_value']}.")
                modified = True
                break
        
        if not modified and mod['arg_name'] in self.ARG_TO_POS_MAP:
            pos_index = self.ARG_TO_POS_MAP[mod['arg_name']]
            if pos_index < len(node.args):
                old_val = getattr(node.args[pos_index], 'value', 'some_variable')
                node.args[pos_index] = ast.Constant(value=mod['new_value'])
                if config.DEBUG_MODE:
                    print(f"  > Modified positional arg {pos_index} ('{mod['arg_name']}') from ~{old_val} to {mod['new_value']}.")
                modified = True

        if not modified and config.DEBUG_MODE:
             print(f"  > WARNING: Could not find argument '{mod['arg_name']}' to modify at this location.")

    def _apply_activation_modification(self, node: ast.Call, mod: dict):
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
                elif mod['new_activation'] == 'Tanh' or mod['new_activation'] == 'Sigmoid':
                    # These don't have inplace parameter
                    node.keywords = [kw for kw in node.keywords if kw.arg != 'inplace']
                elif mod['new_activation'] == 'LeakyReLU':
                    # Ensure negative_slope parameter exists
                    has_negative_slope = any(kw.arg == 'negative_slope' for kw in node.keywords)
                    if not has_negative_slope:
                        node.keywords.append(ast.keyword(arg='negative_slope', value=ast.Constant(value=0.01)))
                
                if config.DEBUG_MODE:
                    print(f"  > Modified activation from {old_activation} to {mod['new_activation']}")

    def _apply_layer_type_modification(self, node: ast.Call, mod: dict):
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
                    # GroupNorm needs num_groups and num_channels
                    node.args = []
                    node.keywords = [
                        ast.keyword(arg='num_groups', value=ast.Constant(value=mod['params']['num_groups'])),
                        ast.keyword(arg='num_channels', value=ast.Constant(value=mod['params']['num_channels']))
                    ]
                elif mod['new_layer_type'] == 'BatchNorm2d':
                    # BatchNorm2d needs num_features
                    node.args = []
                    node.keywords = [
                        ast.keyword(arg='num_features', value=ast.Constant(value=mod['params']['num_features']))
                    ]
                elif mod['new_layer_type'] in ['AdaptiveMaxPool2d', 'AdaptiveAvgPool2d']:
                    # Adaptive pooling needs output_size
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

    def get_modified_code(self) -> str:
        if config.DEBUG_MODE:
            print("[CodeMutator] Applying all scheduled modifications to AST.")
        modified_tree = self.visit(self.tree)
        return ast.unparse(modified_tree)