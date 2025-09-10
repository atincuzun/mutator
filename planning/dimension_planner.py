"""
Dimension planner for handling channel and feature size mutations.

This module handles mutations that change the input/output dimensions of layers,
including both numeric and symbolic expression-based changes.
"""

import random
import json
import operator
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.fx as fx

from mutator import config
from mutator.utils import get_available_parameters


class DimensionPlanner:
    """
    Planner for dimension-based mutations (channel/feature size changes).
    
    This class handles mutations that change the input and output dimensions
    of neural network layers, with support for both fixed numeric values
    and symbolic expressions.
    """
    
    def __init__(self, model_planner):
        """
        Initialize the dimension planner.
        
        Args:
            model_planner: Reference to the main ModelPlanner instance
        """
        self.model_planner = model_planner
        self.available_param_cache = {}

    def plan_dimension_mutation(self) -> Dict[str, Any]:
        """
        Plan a dimension mutation with unified in/out channel system and symbolic expressions.
        
        Returns:
            Dictionary containing the dimension mutation plan
        """
        mutation_groups = self._build_mutation_groups()
        if not mutation_groups:
            return {}

        # Find the final layer in the graph to protect its output dimension.
        # We iterate backwards from the 'output' node of the graph.
        final_layer_name = None
        if self.model_planner.graph:
            for node in reversed(self.model_planner.graph.nodes):
                if node.op == 'call_module' and isinstance(
                    self.model_planner.submodules.get(node.target), (nn.Conv2d, nn.Linear)
                ):
                    final_layer_name = node.target
                    if config.DEBUG_MODE:
                        print(f"[DimensionPlanner] Identified final layer: {final_layer_name}. Its output will be protected.")
                    break
        # Attach the final layer name to the main planner instance so helper methods can access it.
        setattr(self.model_planner, 'final_layer_name', final_layer_name)

        # Compute spatial dimensions if not already done
        if not self.model_planner.spatial_tracker:
            self.model_planner._compute_spatial_dimensions()

        # Find a valid mutation group
        valid_mutation_group = None
        original_dim = None
        new_dim = None
        
        # Try up to 20 times to find a valid mutation
        for _ in range(20):
            mutation_group = random.choice(mutation_groups)
            original_dim_module = self.model_planner.submodules[mutation_group[0].target]
            original_dim = (original_dim_module.out_channels if isinstance(original_dim_module, nn.Conv2d) 
                          else original_dim_module.out_features)

            # 1. Get the list of valid sizes directly from the planner's config.
            #    Filter out the current dimension to ensure a change happens.
            possible_new_dims = [
                size for size in self.model_planner.VALID_CHANNEL_SIZES if size != original_dim
            ]
            
            if not possible_new_dims:
                continue # This layer's dimension isn't in our list, or no alternatives exist. Try another group.

            # 2. Directly choose a new dimension from the filtered list.
            new_dim = random.choice(possible_new_dims)
            
            # Validate that this mutation won't break downstream layers (e.g., group convolution)
            consumers, _ = self._find_downstream_dependencies(mutation_group)
            
            valid = True
            for consumer_node in consumers:
                module = self.model_planner.submodules.get(consumer_node.target)
                if isinstance(module, nn.Conv2d) and module.groups > 1 and new_dim % module.groups != 0:
                    valid = False
                    break
                    
            if valid:
                valid_mutation_group = mutation_group
                break
        
        if not valid_mutation_group:
            if config.DEBUG_MODE:
                print("[DimensionPlanner] Could not find valid dimension mutation after 20 attempts")
            return {}
            
        mutation_group = valid_mutation_group
        consumers, propagators = self._find_downstream_dependencies(mutation_group)
        current_plan = {}
        
        # Decide whether to attempt symbolic expressions
        use_symbolic = self._should_use_symbolic_mutation_for_group(mutation_group)
        group_expr: Optional[str] = None
        if use_symbolic:
            common_params = self._find_common_parameters(mutation_group)
            group_expr = self._generate_symbolic_expression_for_group(common_params, new_dim)

        # Collect original dims for each node
        original_dims = self._collect_original_dimensions(mutation_group, consumers, propagators)
        
        # Choose propagation direction based on probability weights
        direction = random.choices(
            ['forward', 'backward'],
            weights=[
                config.PROPAGATION_DIRECTION_WEIGHTS['forward'],
                config.PROPAGATION_DIRECTION_WEIGHTS['backward']
            ],
            k=1
        )[0]
        
        if config.DEBUG_MODE:
            print(f"[DimensionPlanner] Using {direction} propagation")
        
        if direction == 'forward':
            # Forward propagation (existing logic)
            self._apply_mutations_to_producers(
                mutation_group, current_plan, new_dim, use_symbolic, original_dims, group_expr
            )
            self._apply_mutations_to_consumers(
                consumers, current_plan, new_dim, use_symbolic, original_dims, group_expr
            )
            self._apply_mutations_to_propagators(
                propagators, current_plan, new_dim, use_symbolic, original_dims, group_expr
            )
        else:
            # Backward propagation
            self._apply_backward_propagation(
                mutation_group, current_plan, new_dim, use_symbolic, original_dims, group_expr
            )

        self.model_planner.plan = current_plan
        if config.DEBUG_MODE:
            print("[DimensionPlanner] Generated unified dimension mutation plan:")
            print(json.dumps(current_plan, indent=2))
        
        # Clean up the temporary attribute after planning is complete
        if hasattr(self.model_planner, 'final_layer_name'):
            delattr(self.model_planner, 'final_layer_name')

        return current_plan

    def _build_mutation_groups(self) -> List[List[fx.Node]]:
        """Build mutation groups using union-find algorithm."""
        if not self.model_planner.graph:
            return []
            
        producers = [n for n in self.model_planner.graph.nodes 
                    if n.op == 'call_module' and 
                    isinstance(self.model_planner.submodules.get(n.target), (nn.Conv2d, nn.Linear))]
        
        if not producers: 
            return []
        
        parent = {node: node for node in producers}
        
        def find_set(n):
            if parent[n] == n: 
                return n
            parent[n] = find_set(parent[n])
            return parent[n]
        
        def unite_sets(a, b):
            a_root, b_root = find_set(a), find_set(b)
            if a_root != b_root: 
                parent[b_root] = a_root
        
        # Group producers connected by add operations
        for node in self.model_planner.graph.nodes:
            is_add = node.op == 'call_function' and node.target in [torch.add, torch.ops.aten.add]
            if not is_add: 
                continue
                
            join_producers = []
            for input_node in node.args:
                if isinstance(input_node, fx.Node):
                    p = self._find_nearby_producer_node(input_node)
                    if p and p in parent: 
                        join_producers.append(p)
                        
            if len(join_producers) > 1:
                for i in range(1, len(join_producers)): 
                    unite_sets(join_producers[0], join_producers[i])
        
        # Collect final groups
        final_groups = {}
        for p_node in producers:
            root = find_set(p_node)
            if root not in final_groups: 
                final_groups[root] = []
            final_groups[root].append(p_node)
            
        return list(final_groups.values())

    def _find_downstream_dependencies(self, start_nodes: List[fx.Node]) -> Tuple[List[fx.Node], List[fx.Node]]:
        """Find downstream consumers and propagators."""
        consumers, propagators = set(), set()
        worklist, visited = list(start_nodes), set(start_nodes)
        
        while worklist:
            current_node = worklist.pop(0)
            for user in current_node.users:
                if user in visited: 
                    continue
                visited.add(user)
                
                is_consumer = (user.op == 'call_module' and 
                             isinstance(self.model_planner.submodules.get(user.target), (nn.Conv2d, nn.Linear)))
                is_propagator = (user.op == 'call_module' and 
                               isinstance(self.model_planner.submodules.get(user.target), (nn.BatchNorm2d, nn.LayerNorm)))
                
                if is_consumer: 
                    consumers.add(user)
                elif is_propagator: 
                    propagators.add(user)
                    worklist.append(user)
                else: 
                    worklist.append(user)
                    
        return list(consumers), list(propagators)

    def _find_nearby_producer_node(self, start_node: fx.Node) -> Optional[fx.Node]:
        """Find a nearby producer node in the graph."""
        current_node = start_node
        for _ in range(self.model_planner.search_depth + 1):
            if not isinstance(current_node, fx.Node): 
                return None
            if (current_node.op == 'call_module' and 
                isinstance(self.model_planner.submodules.get(current_node.target), (nn.Conv2d, nn.Linear))): 
                return current_node
            current_node = self._find_tensor_predecessor(current_node)
            if current_node is None: 
                return None
        return None

    @staticmethod
    def _find_tensor_predecessor(node: fx.Node) -> Optional[fx.Node]:
        """Find the tensor predecessor of a node."""
        for arg in node.args:
            if isinstance(arg, fx.Node): 
                return arg
        return None

    def _get_available_params(self, module_name: str) -> List[str]:
        """Get available parameters for symbolic expressions with caching."""
        if module_name in self.available_param_cache:
            return self.available_param_cache[module_name]
            
        source_location = self.model_planner.source_map.get(module_name)
        if not source_location:
            self.available_param_cache[module_name] = []
            return []
            
        source_code = self.model_planner._get_source_code_for_location(source_location)
        if not source_code:
            self.available_param_cache[module_name] = []
            return []
            
        call_node = self.model_planner._find_call_node_at_line(source_code, source_location.get('lineno', -1))
        if not call_node:
            self.available_param_cache[module_name] = []
            return []
            
        params = get_available_parameters(call_node, source_code)
        self.available_param_cache[module_name] = params
        return params

    def _synthesize_symbolic(self, old_dim: int, new_dim_val: int, params: List[str]) -> Optional[str]:
        """
        Deterministic simple expression builder.
        Always returns an expression referencing one param if possible.
        Order of preference:
          1. param (if equals new)
          2. param * k (exact)
          3. param // k (exact shrink, small k)
          4. (param * p)//q (p,q <= 8)
          5. Fallback: (param * new)//param
        """
        if not params or new_dim_val <= 0:
            return None
            
        # Prioritize common nn param names for readability
        priority = ['planes', 'in_channels', 'out_channels', 'width', 'channels', 'features']
        sorted_params = sorted(params, key=lambda x: (x not in priority, len(x), x))
        base_param = sorted_params[0]
        
        # Without knowing runtime param value, we still guarantee correctness using fallback.
        # Try nicer forms only if we can infer ratio from old_dim.
        if old_dim and old_dim > 0:
            # 1. direct multiply
            if new_dim_val == old_dim:
                return base_param
            if new_dim_val % old_dim == 0:
                k = new_dim_val // old_dim
                if k <= 32:
                    return f"{base_param} * {k}"
            # 2. division
            if old_dim % new_dim_val == 0:
                k = old_dim // new_dim_val
                if k <= 32:
                    return f"{base_param} // {k}"
            # 3. rational (param * p)//q approximating new_dim
            # We cannot know base_param's value here (could differ from old_dim if different identifier),
            # so skip to fallback to avoid incorrect mapping.
            
        # 4. fallback guaranteed symbolic
        return f"({base_param} * {new_dim_val}) // {base_param}"

    def _collect_original_dimensions(self, mutation_group: List[fx.Node], 
                                   consumers: List[fx.Node], 
                                   propagators: List[fx.Node]) -> Dict[str, Optional[int]]:
        """Collect original dimensions for each node."""
        original_dims = {}
        
        # Producers use output dimensions
        for node in mutation_group:
            mod = self.model_planner.submodules.get(node.target)
            if isinstance(mod, nn.Conv2d):
                original_dims[node.target] = mod.out_channels
            elif isinstance(mod, nn.Linear):
                original_dims[node.target] = mod.out_features
            else:
                original_dims[node.target] = None
                
        # Consumers and propagators use input dimensions
        for dep_list in (consumers, propagators):
            for n in dep_list:
                mod = self.model_planner.submodules.get(n.target)
                if isinstance(mod, nn.Conv2d):
                    original_dims[n.target] = mod.in_channels
                elif isinstance(mod, nn.Linear):
                    original_dims[n.target] = mod.in_features
                elif isinstance(mod, (nn.BatchNorm2d, nn.LayerNorm)):
                    original_dims[n.target] = getattr(mod, 'num_features', None)
                else:
                    original_dims[n.target] = None
                    
        return original_dims

    def _get_base_plan(self, node_target: str) -> Dict[str, Any]:
        """Get base plan structure for a node."""
        return {
            "mutation_type": "dimension", 
            "new_out": None, 
            "new_in": None, 
            "source_location": self.model_planner.source_map.get(node_target)
        }

    def _apply_mutations_to_producers(self, mutation_group: List[fx.Node], 
                                    current_plan: Dict[str, Any], 
                                    new_dim: int, 
                                    use_symbolic: bool, 
                                    original_dims: Dict[str, Optional[int]],
                                    group_expr: Optional[str]):
        """Apply mutations to producer nodes."""
        for node in mutation_group:
            # Check if this node is the final layer. If so, do not mutate its output.
            if hasattr(self.model_planner, 'final_layer_name') and node.target == self.model_planner.final_layer_name:
                continue
                
            if node.target not in current_plan:
                current_plan[node.target] = self._get_base_plan(node.target)
                
            # Always set numeric new_out for producer nodes
            current_plan[node.target]["new_out"] = new_dim
            
            if use_symbolic:
                if group_expr:
                    current_plan[node.target]["symbolic"] = True
                    current_plan[node.target]["symbolic_expression"] = group_expr
                else:
                    params = self._get_available_params(node.target)
                    sym_expr = self._synthesize_symbolic(original_dims.get(node.target), new_dim, params)
                    if sym_expr:
                        current_plan[node.target]["symbolic"] = True
                        current_plan[node.target]["symbolic_expression"] = sym_expr
                    else:
                        current_plan[node.target]["symbolic"] = False
            else:
                current_plan[node.target]["symbolic"] = False

    def _apply_mutations_to_consumers(self, consumers: List[fx.Node], 
                                    current_plan: Dict[str, Any], 
                                    new_dim: int, 
                                    use_symbolic: bool, 
                                    original_dims: Dict[str, Optional[int]],
                                    group_expr: Optional[str]):
        """Apply mutations to consumer nodes."""
        for consumer_node in consumers:
            if consumer_node.target not in current_plan:
                current_plan[consumer_node.target] = self._get_base_plan(consumer_node.target)
                
            # Set numeric new_in
            current_plan[consumer_node.target]["new_in"] = new_dim
            
            if use_symbolic:
                if group_expr:
                    current_plan[consumer_node.target]["symbolic"] = True
                    current_plan[consumer_node.target]["symbolic_expression"] = group_expr
                else:
                    params = self._get_available_params(consumer_node.target)
                    sym_expr = self._synthesize_symbolic(original_dims.get(consumer_node.target), new_dim, params)
                    if sym_expr:
                        current_plan[consumer_node.target]["symbolic"] = True
                        current_plan[consumer_node.target]["symbolic_expression"] = sym_expr
                    else:
                        current_plan[consumer_node.target]["symbolic"] = False
            else:
                current_plan[consumer_node.target]["symbolic"] = False

    def _apply_mutations_to_propagators(self, propagators: List[fx.Node], 
                                      current_plan: Dict[str, Any], 
                                      new_dim: int, 
                                      use_symbolic: bool, 
                                      original_dims: Dict[str, Optional[int]],
                                      group_expr: Optional[str]):
        """Apply mutations to propagator nodes."""
        for propagator_node in propagators:
            if propagator_node.target not in current_plan:
                current_plan[propagator_node.target] = self._get_base_plan(propagator_node.target)
                
            current_plan[propagator_node.target]["new_in"] = new_dim
            
            if use_symbolic:
                if group_expr:
                    current_plan[propagator_node.target]["symbolic"] = True
                    current_plan[propagator_node.target]["symbolic_expression"] = group_expr
                else:
                    params = self._get_available_params(propagator_node.target)
                    sym_expr = self._synthesize_symbolic(original_dims.get(propagator_node.target), new_dim, params)
                    if sym_expr:
                        current_plan[propagator_node.target]["symbolic"] = True
                        current_plan[propagator_node.target]["symbolic_expression"] = sym_expr
                    else:
                        current_plan[propagator_node.target]["symbolic"] = False
            else:
                current_plan[propagator_node.target]["symbolic"] = False

    def _find_common_parameters(self, mutation_group: List[fx.Node]) -> List[str]:
        """
        Find parameters that are common across all nodes in the mutation group.
        This helps ensure consistent symbolic expressions across the group.
        """
        common_params = None

        for node in mutation_group:
            module_name = node.target
            if module_name not in self.model_planner.source_map:
                continue

            params = self._get_available_params(module_name)
            if params is None:
                continue

            if common_params is None:
                common_params = set(params)
            else:
                common_params = common_params.intersection(set(params))

        # Return common parameters as a list, prioritizing common neural network parameters
        if not common_params:
            return []

        priority_params = ['in_channels', 'out_channels', 'planes', 'width', 'depth', 'expansion']
        sorted_params = sorted(common_params, key=lambda x: (x not in priority_params, x))
        return list(sorted_params)

    def _generate_symbolic_expression_for_group(self, common_params: List[str], target_value: int) -> str:
        """
        Generate a single symbolic expression for the entire mutation group.
        Ensures dimensional consistency across the group by using the same expression.
        """
        if not common_params:
            return str(target_value)

        # Use the most relevant common parameter (prioritize neural network patterns)
        priority_params = ['in_channels', 'out_channels', 'planes', 'width', 'depth', 'expansion']
        relevant_param = None
        for param in priority_params:
            if param in common_params:
                relevant_param = param
                break
        if relevant_param is None:
            relevant_param = common_params[0]

        # Try multiplier that divides target_value
        for multiplier in [2, 4, 8, 16, 32, 64, 128]:
            if target_value % multiplier == 0:
                return f"{relevant_param} * {multiplier}"

        # Try scaled expression that remains reasonable
        for divisor in [2, 4, 8, 16, 32, 64, 128]:
            if target_value * divisor <= 1024:
                return f"{relevant_param} * {target_value} // {divisor}"

        # Fallback simple expression
        return f"{relevant_param} * 2"

    def _should_use_symbolic_mutation_for_group(self, mutation_group: List[fx.Node]) -> bool:
        """
        Determine if symbolic mutation should be used for the entire mutation group.
        Uses configuration settings to make a consistent decision for the whole group.
        """
        # Check configuration mode first
        if config.MUTATION_MODE == 'always_symbolic':
            if config.DEBUG_MODE:
                print(f"[DimensionPlanner] Using symbolic mutation for group (always_symbolic mode)")
            return True
            
        if config.MUTATION_MODE == 'always_fixed':
            if config.DEBUG_MODE:
                print(f"[DimensionPlanner] Using fixed-number mutation for group (always_fixed mode)")
            return False
            
        # For 'auto' mode, use weighted probability from SYMBOLIC_MUTATION_WEIGHTS
        choices = ['symbolic', 'fixed']
        weights = [config.SYMBOLIC_MUTATION_WEIGHTS['symbolic'], config.SYMBOLIC_MUTATION_WEIGHTS['fixed']]
        decision = random.choices(choices, weights=weights, k=1)[0]
        
        if decision == 'symbolic':
            if config.DEBUG_MODE:
                print(f"[DimensionPlanner] Using symbolic mutation for group (weighted probability)")
            return True
        else:
            if config.DEBUG_MODE:
                print(f"[DimensionPlanner] Using fixed-number mutation for group (weighted probability)")
            return False

    def _apply_backward_propagation(self, mutation_group: List[fx.Node], 
                                  current_plan: Dict[str, Any], 
                                  new_dim: int, 
                                  use_symbolic: bool, 
                                  original_dims: Dict[str, Optional[int]],
                                  group_expr: Optional[str]):
        """Apply mutations using backward propagation strategy."""
        # Find input dependencies for the mutation group
        input_dependencies = self._find_input_dependencies(mutation_group)
        
        # Apply mutations to input dependencies first
        for input_node in input_dependencies:
            if input_node.target not in current_plan:
                current_plan[input_node.target] = self._get_base_plan(input_node.target)
            
            # Set output dimension for input dependencies
            current_plan[input_node.target]["new_out"] = new_dim
            
            if use_symbolic:
                if group_expr:
                    current_plan[input_node.target]["symbolic"] = True
                    current_plan[input_node.target]["symbolic_expression"] = group_expr
                else:
                    params = self._get_available_params(input_node.target)
                    sym_expr = self._synthesize_symbolic(original_dims.get(input_node.target), new_dim, params)
                    if sym_expr:
                        current_plan[input_node.target]["symbolic"] = True
                        current_plan[input_node.target]["symbolic_expression"] = sym_expr
                    else:
                        current_plan[input_node.target]["symbolic"] = False
            else:
                current_plan[input_node.target]["symbolic"] = False

        # Then apply to the mutation group itself
        for node in mutation_group:
            if node.target not in current_plan:
                current_plan[node.target] = self._get_base_plan(node.target)
            
            # Set input dimension for the group
            current_plan[node.target]["new_in"] = new_dim
            
            if use_symbolic:
                if group_expr:
                    current_plan[node.target]["symbolic"] = True
                    current_plan[node.target]["symbolic_expression"] = group_expr
                else:
                    params = self._get_available_params(node.target)
                    sym_expr = self._synthesize_symbolic(original_dims.get(node.target), new_dim, params)
                    if sym_expr:
                        current_plan[node.target]["symbolic"] = True
                        current_plan[node.target]["symbolic_expression"] = sym_expr
                    else:
                        current_plan[node.target]["symbolic"] = False
            else:
                current_plan[node.target]["symbolic"] = False

    def _find_input_dependencies(self, mutation_group: List[fx.Node]) -> List[fx.Node]:
        """Find nodes that provide input to the mutation group."""
        input_dependencies = set()
        
        for node in mutation_group:
            # Trace backwards through input nodes
            for input_node in node.all_input_nodes:
                if (input_node.op == 'call_module' and 
                    isinstance(self.model_planner.submodules.get(input_node.target), 
                              (nn.Conv2d, nn.Linear))):
                    input_dependencies.add(input_node)
        
        return list(input_dependencies)