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
        Enhanced with block-aware novel architecture generation.
        
        Returns:
            Dictionary containing the dimension mutation plan
        """
        # Try block-aware mutations first for ResNet-like architectures
        block_aware_plan = self._try_block_aware_mutation()
        if block_aware_plan:
            return block_aware_plan
            
        # Fallback to original mutation system
        return self._plan_traditional_mutation()
        
    def _try_block_aware_mutation(self) -> Optional[Dict[str, Any]]:
        """Attempt block-aware mutations for novel architectures."""
        from mutator.planning.architecture_patterns import pattern_detector
        
        if not self.model_planner.graph:
            if config.DEBUG_MODE:
                print("[DimensionPlanner] No FX graph available for block-aware mutation")
            return None
            
        # Find block instance groups
        try:
            block_groups = pattern_detector.find_block_instance_groups(
                self.model_planner.original_model, 
                self.model_planner.graph
            )
            
            if config.DEBUG_MODE:
                print(f"[DimensionPlanner] Found {len(block_groups)} block groups: {list(block_groups.keys())}")
            
            if not block_groups:
                if config.DEBUG_MODE:
                    print("[DimensionPlanner] No block groups found, falling back to traditional mutation")
                return None
                
            # Select a random block group to mutate
            import random
            group_id = random.choice(list(block_groups.keys()))
            block_modules = block_groups[group_id]
            
            if config.DEBUG_MODE:
                print(f"[DimensionPlanner] Selected block group {group_id} with modules: {block_modules[:3]}...")
            
            # Generate novel mutations for this block
            base_params = {'planes': None}  # Will be determined contextually
            novel_mutations = pattern_detector.generate_novel_block_mutations(
                block_modules, base_params
            )
            
            if config.DEBUG_MODE:
                print(f"[DimensionPlanner] Generated {len(novel_mutations)} novel mutations")
            
            if novel_mutations:
                # Apply source location information
                current_plan = {}
                for module_name, mutation_info in novel_mutations.items():
                    if module_name in self.model_planner.source_map:
                        mutation_info['source_location'] = self.model_planner.source_map[module_name]
                    current_plan[module_name] = mutation_info
                
                if config.DEBUG_MODE:
                    print(f"[DimensionPlanner] Generated block-aware mutation for group {group_id}")
                    print(f"[DimensionPlanner] Novel mutations: {list(novel_mutations.keys())}")
                
                # Store the plan in the model planner
                self.model_planner.plan = current_plan
                return current_plan
        
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"[DimensionPlanner] Error in block-aware mutation: {e}")
        
        return None
        
    def _plan_traditional_mutation(self) -> Dict[str, Any]:
        """Traditional mutation planning (fallback method)."""
        mutation_groups = self._build_mutation_groups()
        if not mutation_groups:
            return {}

        # Compute spatial dimensions if not already done
        if not self.model_planner.spatial_tracker:
            self.model_planner._compute_spatial_dimensions()

        # Find a valid mutation group
        valid_mutation_group = None
        original_dim = None
        new_dim = None
        
        # Try up to 10 times to find a valid mutation (increased for more flexibility)
        for _ in range(10):
            mutation_group = random.choice(mutation_groups)
            original_dim_module = self.model_planner.submodules[mutation_group[0].target]
            original_dim = (original_dim_module.out_channels if isinstance(original_dim_module, nn.Conv2d) 
                          else original_dim_module.out_features)

            # Use unified channel dimension changer: start with random mutation
            # Choose a mutation factor from a wider range for more diversity
            mutation_factors = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2, 3, 4, 5, 6, 7, 8]
            mutation_factor = random.choice(mutation_factors)
        mutation_groups = self._build_mutation_groups()
        if not mutation_groups:
            return {}

        # Compute spatial dimensions if not already done
        if not self.model_planner.spatial_tracker:
            self.model_planner._compute_spatial_dimensions()

        # Find a valid mutation group
        valid_mutation_group = None
        original_dim = None
        new_dim = None
        
        # Try up to 10 times to find a valid mutation (increased for more flexibility)
        for _ in range(10):
            mutation_group = random.choice(mutation_groups)
            original_dim_module = self.model_planner.submodules[mutation_group[0].target]
            original_dim = (original_dim_module.out_channels if isinstance(original_dim_module, nn.Conv2d) 
                          else original_dim_module.out_features)

            # Use unified channel dimension changer: start with random mutation
            # Choose a mutation factor from a wider range for more diversity
            mutation_factors = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2, 3, 4, 5, 6, 7, 8]
            mutation_factor = random.choice(mutation_factors)
            
            # Calculate proposed new dimension
            proposed_dim = int(original_dim * mutation_factor)
            
            # Round to nearest valid channel size
            new_dim = min(self.model_planner.VALID_CHANNEL_SIZES, 
                         key=lambda x: abs(x - proposed_dim))
            
            # Ensure it's different from original
            if new_dim == original_dim:
                continue
                
            # Validate that this mutation won't break downstream layers
            consumers, propagators = self._find_downstream_dependencies(mutation_group)
            
            # Check if all consumers can accept the new dimension
            valid = True
            for consumer_node in consumers:
                module = self.model_planner.submodules.get(consumer_node.target)
                if isinstance(module, nn.Conv2d) and new_dim % module.groups != 0:
                    valid = False
                    break
                    
            if valid:
                valid_mutation_group = mutation_group
                break
        
        if not valid_mutation_group:
            if config.DEBUG_MODE:
                print("[DimensionPlanner] Could not find valid dimension mutation after 10 attempts")
            return {}
            
        mutation_group = valid_mutation_group
        consumers, propagators = self._find_downstream_dependencies(mutation_group)
        current_plan = {}
        
        # Calculate scaling factor for the mutation
        scaling_factor = new_dim / original_dim
        
        # Decide whether to attempt symbolic expressions
        use_symbolic = self._should_use_symbolic_mutation_for_group(mutation_group)
        group_expr: Optional[str] = None
        if use_symbolic:
            common_params = self._find_common_parameters(mutation_group)
            group_expr = self._generate_symbolic_expression_for_group(common_params, new_dim)

        # Collect original dims for each node
        original_dims = self._collect_original_dimensions(mutation_group, consumers, propagators)
        
        # Unified propagation: apply the same mutation pattern throughout the group
        self._apply_mutations_to_producers(
            mutation_group, current_plan, new_dim, use_symbolic, original_dims, group_expr
        )
        self._apply_mutations_to_consumers(
            consumers, current_plan, new_dim, use_symbolic, original_dims, group_expr
        )
        self._apply_mutations_to_propagators(
            propagators, current_plan, new_dim, use_symbolic, original_dims, group_expr
        )

        self.model_planner.plan = current_plan
        if config.DEBUG_MODE:
            print("[DimensionPlanner] Generated unified dimension mutation plan:")
            print(json.dumps(current_plan, indent=2))
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
