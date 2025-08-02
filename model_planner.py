import torch
import torch.nn as nn
import torch.fx as fx
from copy import deepcopy
import random
import operator
import hashlib
import os
import config

class ModelPlanner:
    VALID_CHANNEL_SIZES = config.VALID_CHANNEL_SIZES
    MUTABLE_MODULES = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm)

    def __init__(self, model: nn.Module, source_map: dict = None, search_depth: int = 3):
        self.original_model = model
        self.search_depth = search_depth
        self.source_map = source_map if source_map is not None else {}
        self.plan = {}
        try:
            self.graph_module = fx.symbolic_trace(self.original_model, concrete_args={'weights': None} if 'weights' in str(model.forward.__code__.co_varnames) else {})
            self.graph = self.graph_module.graph
            self.submodules = dict(self.original_model.named_modules())
        except Exception as e:
            raise RuntimeError(f"Failed to symbolically trace the model: {e}")

        self.final_layer_name = None
        for name, module in reversed(list(self.original_model.named_modules())):
            if isinstance(module, nn.Linear):
                self.final_layer_name = name
                break

    def plan_random_mutation(self) -> dict:
        self.clear_plan()
        mutation_groups = self._build_mutation_groups()
        if not mutation_groups:
            return {}

        mutation_group = random.choice(mutation_groups)
        original_dim_module = self.submodules[mutation_group[0].target]
        original_dim = original_dim_module.out_channels if isinstance(original_dim_module, nn.Conv2d) else original_dim_module.out_features

        valid_new_sizes = [s for s in self.VALID_CHANNEL_SIZES if s != original_dim]
        if not valid_new_sizes:
            return {}
        new_dim = random.choice(valid_new_sizes)

        consumers, propagators = self._find_downstream_dependencies(mutation_group)
        current_plan = {}
        
        def get_base_plan(node_target):
            return {"new_out": None, "new_in": None, "source_location": self.source_map.get(node_target)}

        for node in mutation_group:
            if node.target == self.final_layer_name:
                continue
            if node.target not in current_plan:
                current_plan[node.target] = get_base_plan(node.target)
            current_plan[node.target]["new_out"] = new_dim

        for consumer_node in consumers:
            if consumer_node.target not in current_plan:
                current_plan[consumer_node.target] = get_base_plan(consumer_node.target)
            current_plan[consumer_node.target]["new_in"] = new_dim

        for propagator_node in propagators:
            if propagator_node.target not in current_plan:
                current_plan[propagator_node.target] = get_base_plan(propagator_node.target)
            current_plan[propagator_node.target]["new_in"] = new_dim

        self.plan = current_plan
        if config.DEBUG_MODE:
            print("[ModelPlanner] Generated mutation plan:")
            import json
            print(json.dumps(self.plan, indent=2))
        return self.plan

    def apply_plan(self) -> nn.Module:
        if not self.plan:
            raise ValueError("No mutation plan exists. Please run 'plan_random_mutation()' first.")
        new_model = deepcopy(self.original_model)
        for name, details in self.plan.items():
            try:
                original_module = new_model.get_submodule(name)
                mutated_copy = self._create_mutated_copy(original_module, details["new_in"], details["new_out"])
                self._set_nested_attr(new_model, name, mutated_copy)
            except AttributeError:
                continue
        return new_model

    def clear_plan(self): self.plan = {}

    def _build_mutation_groups(self) -> list:
        producers = [n for n in self.graph.nodes if n.op == 'call_module' and isinstance(self.submodules.get(n.target), (nn.Conv2d, nn.Linear))]
        if not producers: return []
        parent = {node: node for node in producers}
        def find_set(n):
            if parent[n] == n: return n
            parent[n] = find_set(parent[n])
            return parent[n]
        def unite_sets(a, b):
            a_root, b_root = find_set(a), find_set(b)
            if a_root != b_root: parent[b_root] = a_root
        for node in self.graph.nodes:
            is_add = node.op == 'call_function' and node.target in [operator.add, torch.add]
            if not is_add: continue
            join_producers = []
            for input_node in node.args:
                if isinstance(input_node, fx.Node):
                    p = self._find_nearby_producer_node(input_node)
                    if p and p in parent: join_producers.append(p)
            if len(join_producers) > 1:
                for i in range(1, len(join_producers)): unite_sets(join_producers[0], join_producers[i])
        final_groups = {}
        for p_node in producers:
            root = find_set(p_node)
            if root not in final_groups: final_groups[root] = []
            final_groups[root].append(p_node)
        return list(final_groups.values())

    def _find_downstream_dependencies(self, start_nodes: list) -> tuple[list, list]:
        consumers, propagators = set(), set()
        worklist, visited = list(start_nodes), set(start_nodes)
        while worklist:
            current_node = worklist.pop(0)
            for user in current_node.users:
                if user in visited: continue
                visited.add(user)
                is_consumer = user.op == 'call_module' and isinstance(self.submodules.get(user.target), (nn.Conv2d, nn.Linear))
                is_propagator = user.op == 'call_module' and isinstance(self.submodules.get(user.target), (nn.BatchNorm2d, nn.LayerNorm))
                if is_consumer: consumers.add(user)
                elif is_propagator: propagators.add(user); worklist.append(user)
                else: worklist.append(user)
        return list(consumers), list(propagators)

    def _find_nearby_producer_node(self, start_node: fx.Node) -> fx.Node | None:
        current_node = start_node
        for _ in range(self.search_depth + 1):
            if not isinstance(current_node, fx.Node): return None
            if current_node.op == 'call_module' and isinstance(self.submodules.get(current_node.target), (nn.Conv2d, nn.Linear)): return current_node
            current_node = self._find_tensor_predecessor(current_node)
            if current_node is None: return None
        return None
    @staticmethod
    def _find_tensor_predecessor(node: fx.Node) -> fx.Node | None:
        for arg in node.args:
            if isinstance(arg, fx.Node): return arg
        return None
    @staticmethod
    def _set_nested_attr(obj: nn.Module, name: str, value: nn.Module):
        parts = name.split('.'); parent = obj
        for part in parts[:-1]: parent = getattr(parent, part)
        setattr(parent, parts[-1], value)
    @classmethod
    def _create_mutated_copy(cls, module: nn.Module, new_in_channels, new_out_channels):
        if not isinstance(module, cls.MUTABLE_MODULES): return deepcopy(module)
        if isinstance(module, nn.Conv2d):
            old_out, old_in = module.out_channels, module.in_channels; new_in = new_in_channels or old_in; new_out = new_out_channels or old_out; groups = module.groups
            if (new_in != old_in or new_out != old_out) and groups > 1:
                if new_in % groups != 0 or new_out % groups != 0: groups = 1
            new_module = nn.Conv2d(in_channels=new_in, out_channels=new_out, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=groups, bias=module.bias is not None)
            min_out, min_in = min(old_out, new_out), min(old_in, new_in); copy_in_channels = min_in // (module.groups // new_module.groups)
            new_module.weight.data.zero_(); new_module.weight.data[:min_out, :copy_in_channels, ...] = module.weight.data[:min_out, :copy_in_channels, ...]
            if module.bias is not None: new_module.bias.data.zero_(); new_module.bias.data[:min_out] = module.bias.data[:min_out]
        elif isinstance(module, nn.Linear):
            old_out, old_in = module.out_features, module.in_features; new_in = new_in_channels or old_in; new_out = new_out_channels or old_out
            new_module = nn.Linear(in_features=new_in, out_features=new_out, bias=module.bias is not None)
            min_out, min_in = min(old_out, new_out), min(old_in, new_in)
            new_module.weight.data.zero_(); new_module.weight.data[:min_out, :min_in] = module.weight.data[:min_out, :min_in]
            if module.bias is not None: new_module.bias.data.zero_(); new_module.bias.data[:min_out] = module.bias.data[:min_out]
        elif isinstance(module, nn.BatchNorm2d):
            old_feats = module.num_features; new_feats = new_in_channels or old_feats
            new_module = nn.BatchNorm2d(num_features=new_feats, eps=module.eps, momentum=module.momentum, affine=module.affine, track_running_stats=module.track_running_stats)
            min_feats = min(old_feats, new_feats)
            if new_module.track_running_stats:
                new_module.running_mean.data.zero_(); new_module.running_var.data.fill_(1)
                new_module.running_mean.data[:min_feats] = module.running_mean.data[:min_feats]; new_module.running_var.data[:min_feats] = module.running_var.data[:min_feats]
            if new_module.affine:
                new_module.weight.data.fill_(1); new_module.bias.data.zero_()
                new_module.weight.data[:min_feats] = module.weight.data[:min_feats]; new_module.bias.data[:min_feats] = module.bias.data[:min_feats]
        elif isinstance(module, nn.LayerNorm):
            old_feats = module.normalized_shape[0]; new_feats = new_in_channels or old_feats
            new_module = nn.LayerNorm(normalized_shape=[new_feats], eps=module.eps, elementwise_affine=module.elementwise_affine)
            min_feats = min(old_feats, new_feats)
            if new_module.elementwise_affine:
                new_module.weight.data.fill_(1); new_module.bias.data.zero_()
                new_module.weight.data[:min_feats] = module.weight.data[:min_feats]; new_module.bias.data[:min_feats] = module.bias.data[:min_feats]
        return new_module
    @staticmethod
    def get_model_checksum(model: nn.Module) -> str:
        try:
            if isinstance(model, fx.GraphModule): graph_repr = model.graph.print_tabular()
            else: graph_repr = fx.symbolic_trace(model).graph.print_tabular()
            return hashlib.sha256(graph_repr.encode()).hexdigest()
        except: return os.urandom(16).hex()