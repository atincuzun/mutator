import ast
import config
import utils

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
                    self._apply_dimension_modification(node, mod)
                elif mod['type'] == 'activation':
                    self._apply_activation_modification(node, mod)
                elif mod['type'] == 'layer_type':
                    self._apply_layer_type_modification(node, mod)
                elif mod['type'] == 'architectural':
                    self._apply_architectural_modification(node, mod)
                elif mod['type'] == 'kernel_size':
                    self._apply_kernel_size_modification(node, mod)
                elif mod['type'] == 'stride':
                    self._apply_stride_modification(node, mod)

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

    def _apply_dimension_modification(self, node: ast.Call, mod: dict):
        """Apply dimension-based modifications with convolution validation"""
        modified = False
        additional_changes = {}
        
        # Validate convolution parameters before modification
        if mod['arg_name'] in ['in_channels', 'out_channels']:
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'Conv2d':
                additional_changes = self._validate_conv_params(
                    mod['arg_name'], mod['new_value'], node
                )

        # Apply primary modification
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

        # Apply additional changes (e.g., groups parameter)
        for param, value in additional_changes.items():
            param_found = False
            for kw in node.keywords:
                if kw.arg == param:
                    kw.value = ast.Constant(value=value)
                    param_found = True
                    if config.DEBUG_MODE:
                        print(f"  > Adjusted {param} to {value} for depthwise conv constraint")
                    break
            
            if not param_found:
                node.keywords.append(ast.keyword(arg=param, value=ast.Constant(value=value)))
                if config.DEBUG_MODE:
                    print(f"  > Added {param}={value} for depthwise conv constraint")

        if not modified and config.DEBUG_MODE:
             print(f"  > WARNING: Could not find argument '{mod['arg_name']}' to modify at this location.")

    def _apply_kernel_size_modification(self, node: ast.Call, mod: dict):
        """Apply kernel size modifications."""
        modified = False
        for kw in node.keywords:
            if kw.arg == 'kernel_size':
                kw.value = ast.Constant(value=mod['new_kernel_size'])
                modified = True
                break
        if not modified:
            # Assuming kernel_size is the 2nd positional argument for Conv2d
            if len(node.args) > 2:
                node.args[2] = ast.Constant(value=mod['new_kernel_size'])
                modified = True
        
        if modified and config.DEBUG_MODE:
            print(f"  > Modified kernel_size to {mod['new_kernel_size']}")

    def _apply_stride_modification(self, node: ast.Call, mod: dict):
        """Apply stride modifications."""
        modified = False
        for kw in node.keywords:
            if kw.arg == 'stride':
                kw.value = ast.Constant(value=mod['new_stride'])
                modified = True
                break
        if not modified:
            # Assuming stride is the 3rd positional argument for Conv2d
            if len(node.args) > 3:
                node.args[3] = ast.Constant(value=mod['new_stride'])
                modified = True

        if modified and config.DEBUG_MODE:
            print(f"  > Modified stride to {mod['new_stride']}")

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
                    print(f"  > Modified activation from {old_activation} to {new_activation_name}")
                    if not config.ALLOW_HELPER_FUNCTION_MUTATIONS:
                        print(f"  > Direct instantiation mode: only nn.Module calls mutated")
            
            # When helper mutations are disabled, don't mutate helper function calls
            elif not config.ALLOW_HELPER_FUNCTION_MUTATIONS:
                if config.DEBUG_MODE:
                    func_name = getattr(node.func, 'id', 'unknown')
                    print(f"  > Skipping potential helper function call: {func_name}")

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
                    if not config.ALLOW_HELPER_FUNCTION_MUTATIONS:
                        print(f"  > Direct instantiation mode: only nn.Module calls mutated")
            
            # When helper mutations are disabled, don't mutate helper function calls
            elif not config.ALLOW_HELPER_FUNCTION_MUTATIONS:
                if config.DEBUG_MODE:
                    func_name = getattr(node.func, 'id', 'unknown')
                    print(f"  > Skipping potential helper function call: {func_name}")

    def _apply_architectural_modification(self, node, mod: dict):
        """Apply architectural modifications like changing block configurations."""
        architectural_type = mod['architectural_type']
        params = mod['params']
        
        if architectural_type == 'block_setting':
            # Modify the block_setting list
            if isinstance(node, ast.List):
                # Replace the entire list with new configuration
                new_configs = params.get('new_configs', [])
                
                # Create new AST nodes for the new configuration
                new_elements = []
                for config_tuple in new_configs:
                    # Create CNBlockConfig(input_channels, out_channels, num_layers) call
                    call_node = ast.Call(
                        func=ast.Name(id='CNBlockConfig', ctx=ast.Load()),
                        args=[
                            ast.Constant(value=config_tuple[0]),  # input_channels
                            ast.Constant(value=config_tuple[1]) if config_tuple[1] is not None else ast.Constant(value=None),  # out_channels
                            ast.Constant(value=config_tuple[2])   # num_layers
                        ],
                        keywords=[]
                    )
                    new_elements.append(call_node)
                
                node.elts = new_elements
                
                if config.DEBUG_MODE:
                    print(f"  > Modified block_setting configuration: {new_configs}")
            
            elif isinstance(node, ast.Call) and hasattr(node.func, 'id') and node.func.id == 'CNBlockConfig':
                # Modify individual CNBlockConfig calls
                if 'input_channels' in params:
                    if len(node.args) > 0:
                        node.args[0] = ast.Constant(value=params['input_channels'])
                if 'out_channels' in params:
                    if len(node.args) > 1:
                        node.args[1] = ast.Constant(value=params['out_channels']) if params['out_channels'] is not None else ast.Constant(value=None)
                if 'num_layers' in params:
                    if len(node.args) > 2:
                        node.args[2] = ast.Constant(value=params['num_layers'])
                
                if config.DEBUG_MODE:
                    print(f"  > Modified CNBlockConfig: {params}")
        
        elif architectural_type == 'depth_multiplier':
            # Apply depth multiplier to existing configurations
            multiplier = params.get('multiplier', 1.0)
            if isinstance(node, ast.List):
                for element in node.elts:
                    if (isinstance(element, ast.Call) and 
                        hasattr(element.func, 'id') and 
                        element.func.id == 'CNBlockConfig' and 
                        len(element.args) > 2):
                        # Multiply the num_layers (third argument)
                        current_layers = element.args[2].value if hasattr(element.args[2], 'value') else 3
                        new_layers = max(1, int(current_layers * multiplier))
                        element.args[2] = ast.Constant(value=new_layers)
                
                if config.DEBUG_MODE:
                    print(f"  > Applied depth multiplier {multiplier} to block configurations")
        
        elif architectural_type == 'width_multiplier':
            # Apply width multiplier to channel dimensions
            multiplier = params.get('multiplier', 1.0)
            if isinstance(node, ast.List):
                for element in node.elts:
                    if (isinstance(element, ast.Call) and 
                        hasattr(element.func, 'id') and 
                        element.func.id == 'CNBlockConfig'):
                        # Multiply input_channels and out_channels
                        if len(element.args) > 0 and hasattr(element.args[0], 'value'):
                            current_in = element.args[0].value
                            new_in = max(1, int(current_in * multiplier))
                            element.args[0] = ast.Constant(value=new_in)
                        
                        if len(element.args) > 1 and element.args[1].value is not None:
                            current_out = element.args[1].value
                            new_out = max(1, int(current_out * multiplier))
                            element.args[1] = ast.Constant(value=new_out)
                
                if config.DEBUG_MODE:
                    print(f"  > Applied width multiplier {multiplier} to channel dimensions")

    def get_modified_code(self) -> str:
        if config.DEBUG_MODE:
            print("[CodeMutator] Applying all scheduled modifications to AST.")
        modified_tree = self.visit(self.tree)
        return ast.unparse(modified_tree)
