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
        if location and new_value is not None:
            mod = {
                'location': location,
                'arg_name': arg_name,
                'new_value': new_value
            }
            self.modifications.append(mod)
            if config.DEBUG_MODE:
                print(f"[CodeMutator] Scheduled: {mod}")

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        
        for mod in self.modifications:
            loc = mod['location']
            if (hasattr(node, 'lineno') and node.lineno == loc['lineno'] and
                    hasattr(node, 'col_offset') and node.col_offset == loc['col_offset']):
                
                if config.DEBUG_MODE:
                    print(f"[CodeMutator] Found AST Call node at Line {loc['lineno']}, Col {loc['col_offset']} for modification.")

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

        return node

    def get_modified_code(self) -> str:
        if config.DEBUG_MODE:
            print("[CodeMutator] Applying all scheduled modifications to AST.")
        modified_tree = self.visit(self.tree)
        return ast.unparse(modified_tree)