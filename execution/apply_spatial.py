"""Apply spatial (kernel size, stride) modifications to AST nodes."""

import ast
from mutator import config


def apply_kernel_size_modification(node: ast.Call, mod: dict) -> None:
    modified = False
    for kw in node.keywords:
        if kw.arg == 'kernel_size':
            kw.value = ast.Constant(value=mod['new_kernel_size'])
            modified = True
            break
    if not modified:
        if len(node.args) > 2:
            node.args[2] = ast.Constant(value=mod['new_kernel_size'])
            modified = True
    if modified and config.DEBUG_MODE:
        print(f"  > Modified kernel_size to {mod['new_kernel_size']}")


def apply_stride_modification(node: ast.Call, mod: dict) -> None:
    modified = False
    for kw in node.keywords:
        if kw.arg == 'stride':
            kw.value = ast.Constant(value=mod['new_stride'])
            modified = True
            break
    if not modified:
        if len(node.args) > 3:
            node.args[3] = ast.Constant(value=mod['new_stride'])
            modified = True
    if modified and config.DEBUG_MODE:
        print(f"  > Modified stride to {mod['new_stride']}")
