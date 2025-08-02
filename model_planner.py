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
    ACTIVATION_MODULES = (nn.ReLU, nn.GELU, nn.ELU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid, nn.SiLU)
    NORMALIZATION_MODULES = (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)
    POOLING_MODULES = (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d)

    def __init__(self, model: nn.Module, source_map: dict = None, search_depth: int = 3):
        self.original_model = model
        self.search_depth = search_depth
        self.source_map = source_map if source_map is not None else {}
        self.plan = {}
        self.fx_compatible = True
        
        try:
            self.graph_module = fx.symbolic_trace(self.original_model, concrete_args={'weights': None} if 'weights' in str(model.forward.__code__.co_varnames) else {})
            self.graph = self.graph_module.graph
            self.submodules = dict(self.original_model.named_modules())
        except Exception as e:
            # FX tracing failed - use fallback mode for ConvNeXT and similar models
            self.fx_compatible = False
            self.graph_module = None
            self.graph = None
            self.submodules = dict(self.original_model.named_modules())
            if config.DEBUG_MODE:
                print(f"[ModelPlanner] FX tracing failed, using fallback mode: {e}")

        self.final_layer_name = None
        for name, module in reversed(list(self.original_model.named_modules())):
            if isinstance(module, nn.Linear):
                self.final_layer_name = name
                break

    def plan_random_mutation(self) -> dict:
        self.clear_plan()
        
        # If FX tracing failed, use fallback mutation strategy
        if not self.fx_compatible:
            if config.DEBUG_MODE:
                print("[ModelPlanner] Using fallback mutation strategy (FX incompatible)")
            return self._plan_fallback_mutation()
        
        # Check if this is a ConvNeXT-style model by looking for characteristic modules
        is_convnext_style = self._detect_convnext_architecture()
        
        if is_convnext_style:
            if config.DEBUG_MODE:
                print("[ModelPlanner] Detected ConvNeXT-style architecture, using compatible mutation strategy")
            # For ConvNeXT models, focus on mutations that work well with FX tracing
            # Prioritize dimension and activation mutations over problematic structural changes
            safe_mutation_types = ['dimension', 'activation']
            safe_weights = [0.6, 0.4]  # Higher weight on dimension mutations
            chosen_mutation_type = random.choices(safe_mutation_types, weights=safe_weights)[0]
        else:
            # Choose mutation type based on configured weights for regular models
            mutation_types = list(config.MUTATION_TYPE_WEIGHTS.keys())
            weights = list(config.MUTATION_TYPE_WEIGHTS.values())
            chosen_mutation_type = random.choices(mutation_types, weights=weights)[0]
        
        if chosen_mutation_type == 'dimension':
            return self._plan_dimension_mutation()
        elif chosen_mutation_type == 'activation':
            return self._plan_activation_mutation()
        elif chosen_mutation_type == 'layer_type':
            return self._plan_layer_type_mutation()
        else:
            return self._plan_dimension_mutation()  # fallback

    def _plan_dimension_mutation(self) -> dict:
        """Original dimension mutation logic."""
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
            return {"mutation_type": "dimension", "new_out": None, "new_in": None, "source_location": self.source_map.get(node_target)}

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
            print("[ModelPlanner] Generated dimension mutation plan:")
            import json
            print(json.dumps(self.plan, indent=2))
        return self.plan

    def _plan_activation_mutation(self) -> dict:
        """Plan mutation of activation functions."""
        activation_candidates = []
        
        # Find all activation function modules
        for name, module in self.original_model.named_modules():
            if isinstance(module, self.ACTIVATION_MODULES):
                module_type = type(module).__name__
                if module_type in config.ACTIVATION_MUTATIONS:
                    # Check if this module has a valid source location
                    if name in self.source_map:
                        # When helper mutations are disabled, ensure this is a direct instantiation
                        if not config.ALLOW_HELPER_FUNCTION_MUTATIONS:
                            # Check if the source location represents a direct instantiation
                            if self._is_direct_instantiation_location(name):
                                activation_candidates.append((name, module_type))
                        else:
                            activation_candidates.append((name, module_type))
        
        if not activation_candidates:
            if config.DEBUG_MODE:
                print(f"[ModelPlanner] No mutable activation functions found (helper_mutations={config.ALLOW_HELPER_FUNCTION_MUTATIONS})")
            return {}
        
        # Choose a random activation to mutate
        target_name, current_activation = random.choice(activation_candidates)
        possible_mutations = config.ACTIVATION_MUTATIONS[current_activation]
        new_activation = random.choice(possible_mutations)
        
        current_plan = {
            target_name: {
                "mutation_type": "activation",
                "current_activation": current_activation,
                "new_activation": new_activation,
                "source_location": self.source_map.get(target_name)
            }
        }
        
        self.plan = current_plan
        if config.DEBUG_MODE:
            print(f"[ModelPlanner] Generated activation mutation plan: {current_activation} -> {new_activation}")
            import json
            print(json.dumps(self.plan, indent=2))
        return self.plan

    def _plan_layer_type_mutation(self) -> dict:
        """Plan mutation of layer types (normalization, pooling)."""
        layer_candidates = []
        
        # Find all mutable layer types
        for name, module in self.original_model.named_modules():
            module_type = type(module).__name__
            if module_type in config.LAYER_TYPE_MUTATIONS:
                # Check if this module has a valid source location
                if name in self.source_map:
                    # When helper mutations are disabled, ensure this is a direct instantiation
                    if not config.ALLOW_HELPER_FUNCTION_MUTATIONS:
                        if self._is_direct_instantiation_location(name):
                            layer_candidates.append((name, module_type, module))
                    else:
                        layer_candidates.append((name, module_type, module))
        
        if not layer_candidates:
            if config.DEBUG_MODE:
                print(f"[ModelPlanner] No mutable layer types found (helper_mutations={config.ALLOW_HELPER_FUNCTION_MUTATIONS})")
            return {}
        
        # Choose a random layer to mutate
        target_name, current_layer_type, module = random.choice(layer_candidates)
        possible_mutations = config.LAYER_TYPE_MUTATIONS[current_layer_type]
        new_layer_type = random.choice(possible_mutations)
        
        # Extract relevant parameters for the mutation
        mutation_params = self._extract_layer_params(module, current_layer_type, new_layer_type)
        
        current_plan = {
            target_name: {
                "mutation_type": "layer_type",
                "current_layer_type": current_layer_type,
                "new_layer_type": new_layer_type,
                "mutation_params": mutation_params,
                "source_location": self.source_map.get(target_name)
            }
        }
        
        self.plan = current_plan
        if config.DEBUG_MODE:
            print(f"[ModelPlanner] Generated layer type mutation plan: {current_layer_type} -> {new_layer_type}")
            import json
            print(json.dumps(self.plan, indent=2))
        return self.plan

    def _is_fx_incompatible_module(self, module_name: str) -> bool:
        """Check if a module is known to be incompatible with torch.fx tracing."""
        module = self.submodules.get(module_name)
        if module is None:
            return False
        
        module_type = type(module).__name__
        return module_type in config.FX_INCOMPATIBLE_MODULES
    
    def _get_fx_compatible_replacement(self, module_name: str) -> str:
        """Get FX-compatible replacement for incompatible modules."""
        module = self.submodules.get(module_name)
        if module is None:
            return None
        
        module_type = type(module).__name__
        return config.FX_COMPATIBLE_REPLACEMENTS.get(module_type)
    
    def _plan_convnext_compatible_mutation(self) -> dict:
        """Plan mutations specifically designed for ConvNeXT-style architectures."""
        mutation_candidates = []
        
        # Find Conv2d layers that we can mutate (avoid FX-incompatible modules)
        for name, module in self.original_model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Skip if this module is in an FX-incompatible context
                if not self._is_fx_incompatible_module(name):
                    if name in self.source_map:
                        mutation_candidates.append((name, module))
        
        if not mutation_candidates:
            # Fallback to regular mutations for compatible modules
            return self._plan_dimension_mutation()
        
        # Choose a Conv2d to mutate
        target_name, target_module = random.choice(mutation_candidates)
        
        # ConvNeXT-specific mutations
        mutation_type = random.choice(['kernel_size', 'groups'])
        
        if mutation_type == 'kernel_size' and hasattr(target_module, 'kernel_size'):
            # Mutate kernel size (common in ConvNeXT depth-wise convs)
            current_kernel = target_module.kernel_size[0] if isinstance(target_module.kernel_size, tuple) else target_module.kernel_size
            new_kernels = [k for k in config.CONVNEXT_MUTATIONS['depthwise_conv']['kernel_sizes'] if k != current_kernel]
            if new_kernels:
                new_kernel = random.choice(new_kernels)
                current_plan = {
                    target_name: {
                        "mutation_type": "convnext_kernel",
                        "current_kernel_size": current_kernel,
                        "new_kernel_size": new_kernel,
                        "source_location": self.source_map.get(target_name)
                    }
                }
                self.plan = current_plan
                return self.plan
        
        # Fallback to dimension mutation
        return self._plan_dimension_mutation()

    def _detect_convnext_architecture(self) -> bool:
        """Detect if this is a ConvNeXT-style architecture based on characteristic modules."""
        # Look for characteristic ConvNeXT modules
        convnext_indicators = [
            'StochasticDepth', 'LayerNorm2d', 'Permute', 'CNBlock'
        ]
        
        for name, module in self.original_model.named_modules():
            module_type = type(module).__name__
            if module_type in convnext_indicators:
                return True
            
            # Also check for depth-wise convolutions (groups == in_channels)
            if isinstance(module, nn.Conv2d) and hasattr(module, 'groups'):
                if module.groups > 1 and module.groups == module.in_channels:
                    return True
        
        return False

    def _plan_fallback_mutation(self) -> dict:
        """Plan mutations for FX-incompatible models using module inspection only."""
        # Focus on simple, safe mutations that don't require FX graph analysis
        mutation_candidates = []
        
        # Find mutable modules that we can safely mutate
        for name, module in self.original_model.named_modules():
            if name in self.source_map:
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    # These are safe to mutate and common in ConvNeXT
                    mutation_candidates.append((name, module, 'dimension'))
                elif isinstance(module, self.ACTIVATION_MODULES):
                    # Activation functions are safe to mutate
                    module_type = type(module).__name__
                    if module_type in config.ACTIVATION_MUTATIONS:
                        mutation_candidates.append((name, module, 'activation'))
                elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                    # Normalization layers are relatively safe
                    module_type = type(module).__name__
                    if module_type in config.LAYER_TYPE_MUTATIONS:
                        mutation_candidates.append((name, module, 'layer_type'))
        
        if not mutation_candidates:
            if config.DEBUG_MODE:
                print("[ModelPlanner] No mutation candidates found in fallback mode")
            return {}
        
        # Choose a random mutation
        target_name, target_module, mutation_type = random.choice(mutation_candidates)
        
        if mutation_type == 'dimension':
            return self._plan_fallback_dimension_mutation(target_name, target_module)
        elif mutation_type == 'activation':
            return self._plan_fallback_activation_mutation(target_name, target_module)
        elif mutation_type == 'layer_type':
            return self._plan_fallback_layer_type_mutation(target_name, target_module)
        
        return {}
    
    def _plan_fallback_dimension_mutation(self, target_name: str, target_module: nn.Module) -> dict:
        """Plan dimension mutation without FX graph analysis."""
        if isinstance(target_module, nn.Conv2d):
            original_dim = target_module.out_channels
        elif isinstance(target_module, nn.Linear):
            original_dim = target_module.out_features
        else:
            return {}
        
        valid_new_sizes = [s for s in self.VALID_CHANNEL_SIZES if s != original_dim]
        if not valid_new_sizes:
            return {}
        
        new_dim = random.choice(valid_new_sizes)
        
        current_plan = {
            target_name: {
                "mutation_type": "dimension",
                "new_out": new_dim,
                "new_in": None,
                "source_location": self.source_map.get(target_name)
            }
        }
        
        self.plan = current_plan
        if config.DEBUG_MODE:
            print(f"[ModelPlanner] Generated fallback dimension mutation plan for {target_name}")
        return self.plan
    
    def _plan_fallback_activation_mutation(self, target_name: str, target_module: nn.Module) -> dict:
        """Plan activation mutation without FX graph analysis."""
        module_type = type(target_module).__name__
        possible_mutations = config.ACTIVATION_MUTATIONS[module_type]
        new_activation = random.choice(possible_mutations)
        
        current_plan = {
            target_name: {
                "mutation_type": "activation",
                "current_activation": module_type,
                "new_activation": new_activation,
                "source_location": self.source_map.get(target_name)
            }
        }
        
        self.plan = current_plan
        if config.DEBUG_MODE:
            print(f"[ModelPlanner] Generated fallback activation mutation plan: {module_type} -> {new_activation}")
        return self.plan
    
    def _plan_fallback_layer_type_mutation(self, target_name: str, target_module: nn.Module) -> dict:
        """Plan layer type mutation without FX graph analysis."""
        module_type = type(target_module).__name__
        possible_mutations = config.LAYER_TYPE_MUTATIONS[module_type]
        new_layer_type = random.choice(possible_mutations)
        
        # Extract relevant parameters for the mutation
        mutation_params = self._extract_layer_params(target_module, module_type, new_layer_type)
        
        current_plan = {
            target_name: {
                "mutation_type": "layer_type",
                "current_layer_type": module_type,
                "new_layer_type": new_layer_type,
                "mutation_params": mutation_params,
                "source_location": self.source_map.get(target_name)
            }
        }
        
        self.plan = current_plan
        if config.DEBUG_MODE:
            print(f"[ModelPlanner] Generated fallback layer type mutation plan: {module_type} -> {new_layer_type}")
        return self.plan

    def _is_direct_instantiation_location(self, module_name: str) -> bool:
        """Check if a module's source location represents a direct nn.Module instantiation."""
        if module_name not in self.source_map:
            return False
        
        # This is a simplified check - in a more sophisticated implementation,
        # we would parse the AST at the source location to determine if it's a direct call
        # For now, we assume all tracked locations are valid when helper mutations are disabled
        return True

    def _extract_layer_params(self, module: nn.Module, current_type: str, new_type: str) -> dict:
        """Extract parameters needed for layer type mutation."""
        params = {}
        
        if current_type == 'BatchNorm2d' and new_type == 'GroupNorm':
            params['num_groups'] = min(32, module.num_features)  # Common default
            params['num_channels'] = module.num_features
        elif current_type == 'GroupNorm' and new_type == 'BatchNorm2d':
            params['num_features'] = module.num_channels
        elif current_type == 'BatchNorm2d' and new_type == 'LayerNorm':
            params['num_features'] = module.num_features
            params['normalized_shape'] = [module.num_features]
        elif current_type == 'LayerNorm' and new_type == 'BatchNorm2d':
            params['num_features'] = module.normalized_shape[0] if hasattr(module, 'normalized_shape') else 64
        elif current_type in ['MaxPool2d', 'AvgPool2d'] and new_type in ['MaxPool2d', 'AvgPool2d']:
            params['kernel_size'] = module.kernel_size
            params['stride'] = module.stride
            params['padding'] = module.padding
        elif current_type in ['MaxPool2d', 'AvgPool2d'] and new_type in ['AdaptiveMaxPool2d', 'AdaptiveAvgPool2d']:
            # Adaptive pooling uses output_size instead of kernel_size/stride/padding
            params['output_size'] = (7, 7)  # Common default for adaptive pooling
        
        return params

    def apply_plan(self) -> nn.Module:
        if not self.plan:
            raise ValueError("No mutation plan exists. Please run 'plan_random_mutation()' first.")
        new_model = deepcopy(self.original_model)
        for name, details in self.plan.items():
            try:
                original_module = new_model.get_submodule(name)
                mutation_type = details.get("mutation_type", "dimension")  # backward compatibility
                
                if mutation_type == "dimension":
                    mutated_copy = self._create_mutated_copy(original_module, details["new_in"], details["new_out"])
                elif mutation_type == "activation":
                    mutated_copy = self._create_activation_mutation(original_module, details["new_activation"])
                elif mutation_type == "layer_type":
                    mutated_copy = self._create_layer_type_mutation(original_module, details["new_layer_type"], details["mutation_params"])
                else:
                    continue  # skip unknown mutation types
                    
                self._set_nested_attr(new_model, name, mutated_copy)
            except AttributeError:
                continue
        return new_model

    def _create_activation_mutation(self, module: nn.Module, new_activation: str) -> nn.Module:
        """Create a new activation module with the specified type."""
        # Preserve common parameters where possible
        inplace = getattr(module, 'inplace', True)
        
        if new_activation == 'ReLU':
            return nn.ReLU(inplace=inplace)
        elif new_activation == 'GELU':
            return nn.GELU()
        elif new_activation == 'ELU':
            return nn.ELU(inplace=inplace)
        elif new_activation == 'LeakyReLU':
            return nn.LeakyReLU(inplace=inplace)
        elif new_activation == 'SiLU':
            return nn.SiLU(inplace=inplace)
        elif new_activation == 'Tanh':
            return nn.Tanh()
        elif new_activation == 'Sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU(inplace=inplace)  # fallback

    def _create_layer_type_mutation(self, module: nn.Module, new_layer_type: str, params: dict) -> nn.Module:
        """Create a new layer module with the specified type."""
        if new_layer_type == 'BatchNorm2d':
            return nn.BatchNorm2d(num_features=params['num_features'])
        elif new_layer_type == 'GroupNorm':
            return nn.GroupNorm(num_groups=params['num_groups'], num_channels=params['num_channels'])
        elif new_layer_type == 'LayerNorm':
            return nn.LayerNorm(normalized_shape=params.get('normalized_shape', [params['num_features']]))
        elif new_layer_type == 'InstanceNorm2d':
            return nn.InstanceNorm2d(num_features=params['num_features'])
        elif new_layer_type == 'MaxPool2d':
            return nn.MaxPool2d(
                kernel_size=params['kernel_size'],
                stride=params['stride'],
                padding=params['padding']
            )
        elif new_layer_type == 'AvgPool2d':
            return nn.AvgPool2d(
                kernel_size=params['kernel_size'],
                stride=params['stride'],
                padding=params['padding']
            )
        elif new_layer_type == 'AdaptiveMaxPool2d':
            return nn.AdaptiveMaxPool2d(output_size=params['output_size'])
        elif new_layer_type == 'AdaptiveAvgPool2d':
            return nn.AdaptiveAvgPool2d(output_size=params['output_size'])
        else:
            return deepcopy(module)  # fallback

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
    
    @staticmethod
    def get_model_parameter_checksum(model: nn.Module) -> str:
        """Generate checksum based on model parameters for FX-incompatible models."""
        try:
            # Collect parameter shapes and names to create a structural signature
            param_info = []
            for name, param in model.named_parameters():
                param_info.append(f"{name}:{param.shape}")
            
            # Also collect module types and names for additional structural info
            module_info = []
            for name, module in model.named_modules():
                if len(name) > 0:  # Skip root module
                    module_info.append(f"{name}:{type(module).__name__}")
            
            # Create combined signature
            signature = "|".join(param_info) + "||" + "|".join(module_info)
            return hashlib.sha256(signature.encode()).hexdigest()
        except Exception:
            return os.urandom(16).hex()