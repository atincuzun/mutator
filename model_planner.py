import torch
import torch.nn as nn
import torch.fx as fx
from copy import deepcopy
import random
import operator
import hashlib
import os
import ast
from . import config
import inspect
from .utils import is_block_definition_context, is_top_level_net_context, get_available_parameters
from .plan_uniqueness_tracker import get_plan_tracker

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
        self.spatial_tracker = {}  # Track spatial dimensions: {module_name: (height, width)}
        self.input_shape = (3, 224, 224)  # Default input shape (C, H, W)
        
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
        plan_tracker = get_plan_tracker()
        
        # Try to generate a unique plan (max 10 attempts to avoid infinite loops)
        for attempt in range(10):
            # Compute spatial dimensions before planning any mutations
            self._compute_spatial_dimensions()
            
            # If FX tracing failed, use fallback mutation strategy
            if not self.fx_compatible:
                if config.DEBUG_MODE:
                    print("[ModelPlanner] Using fallback mutation strategy (FX incompatible)")
                plan = self._plan_fallback_mutation()
            else:
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
                    plan = self._plan_dimension_mutation()
                elif chosen_mutation_type == 'activation':
                    plan = self._plan_activation_mutation()
                elif chosen_mutation_type == 'layer_type':
                    plan = self._plan_layer_type_mutation()
                elif chosen_mutation_type == 'architectural':
                    plan = self._plan_architectural_mutation()
                elif chosen_mutation_type == 'kernel_size':
                    plan = self._plan_kernel_size_mutation()
                elif chosen_mutation_type == 'stride':
                    plan = self._plan_stride_mutation()
                else:
                    plan = self._plan_dimension_mutation()  # fallback
            
            # Check if we got a valid plan and if it's unique
            if plan and plan_tracker.is_unique_plan(plan):
                plan_tracker.register_plan(plan)
                if config.DEBUG_MODE:
                    print(f"[ModelPlanner] Generated unique plan (attempt {attempt + 1})")
                return plan
            elif not plan:
                if config.DEBUG_MODE:
                    print(f"[ModelPlanner] Failed to generate valid plan (attempt {attempt + 1})")
            else:
                if config.DEBUG_MODE:
                    print(f"[ModelPlanner] Generated duplicate plan, retrying (attempt {attempt + 1})")
        
        # If we reach here, we couldn't generate a unique plan after 10 attempts
        if config.DEBUG_MODE:
            print("[ModelPlanner] Could not generate unique plan after 10 attempts")
        return {}

    def _plan_dimension_mutation(self) -> dict:
        """Dimension mutation with unified in/out channel system and free symbolic combinations."""
        mutation_groups = self._build_mutation_groups()
        if not mutation_groups:
            return {}

        # Compute spatial dimensions if not already done
        if not self.spatial_tracker:
            self._compute_spatial_dimensions()

        # Find a valid mutation group
        valid_mutation_group = None
        original_dim = None
        new_dim = None
        
        # Try up to 10 times to find a valid mutation (increased for more flexibility)
        for _ in range(10):
            mutation_group = random.choice(mutation_groups)
            original_dim_module = self.submodules[mutation_group[0].target]
            original_dim = original_dim_module.out_channels if isinstance(original_dim_module, nn.Conv2d) else original_dim_module.out_features

            # Use unified channel dimension changer: start with random mutation
            # Choose a mutation factor from a wider range for more diversity
            mutation_factors = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2, 3, 4, 5, 6, 7, 8]
            mutation_factor = random.choice(mutation_factors)
            
            # Calculate proposed new dimension
            proposed_dim = int(original_dim * mutation_factor)
            
            # Round to nearest valid channel size
            new_dim = min(self.VALID_CHANNEL_SIZES, key=lambda x: abs(x - proposed_dim))
            
            # Ensure it's different from original
            if new_dim == original_dim:
                continue
                
            # Validate that this mutation won't break downstream layers
            consumers, propagators = self._find_downstream_dependencies(mutation_group)
            
            # Check if all consumers can accept the new dimension
            valid = True
            for consumer_node in consumers:
                module = self.submodules.get(consumer_node.target)
                if isinstance(module, nn.Conv2d) and new_dim % module.groups != 0:
                    valid = False
                    break
                    
            if valid:
                valid_mutation_group = mutation_group
                break
        
        if not valid_mutation_group:
            if config.DEBUG_MODE:
                print("[ModelPlanner] Could not find valid dimension mutation after 10 attempts")
            return {}
            
        mutation_group = valid_mutation_group
        consumers, propagators = self._find_downstream_dependencies(mutation_group)
        current_plan = {}
        
        def get_base_plan(node_target):
            return {"mutation_type": "dimension", "new_out": None, "new_in": None, "source_location": self.source_map.get(node_target)}

        # Unified propagation: apply the same mutation pattern throughout
        for node in mutation_group:
            if node.target == self.final_layer_name:
                continue
            if node.target not in current_plan:
                current_plan[node.target] = get_base_plan(node.target)
            
            # Check if this mutation should be symbolic based on context
            if self._should_use_symbolic_mutation(node.target):
                current_plan[node.target]["symbolic"] = True
                current_plan[node.target]["symbolic_expression"] = self._generate_symbolic_expression(node.target, new_dim)
            else:
                current_plan[node.target]["symbolic"] = False
                current_plan[node.target]["new_out"] = new_dim

        for consumer_node in consumers:
            if consumer_node.target not in current_plan:
                current_plan[consumer_node.target] = get_base_plan(consumer_node.target)
            
            # Check if this mutation should be symbolic based on context
            if self._should_use_symbolic_mutation(consumer_node.target):
                current_plan[consumer_node.target]["symbolic"] = True
                current_plan[consumer_node.target]["symbolic_expression"] = self._generate_symbolic_expression(consumer_node.target, new_dim)
            else:
                current_plan[consumer_node.target]["symbolic"] = False
                current_plan[consumer_node.target]["new_in"] = new_dim

        for propagator_node in propagators:
            if propagator_node.target not in current_plan:
                current_plan[propagator_node.target] = get_base_plan(propagator_node.target)
            
            # Check if this mutation should be symbolic based on context
            if self._should_use_symbolic_mutation(propagator_node.target):
                current_plan[propagator_node.target]["symbolic"] = True
                current_plan[propagator_node.target]["symbolic_expression"] = self._generate_symbolic_expression(propagator_node.target, new_dim)
            else:
                current_plan[propagator_node.target]["symbolic"] = False
                current_plan[propagator_node.target]["new_in"] = new_dim

        self.plan = current_plan
        if config.DEBUG_MODE:
            print("[ModelPlanner] Generated unified dimension mutation plan:")
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

    def _plan_kernel_size_mutation(self) -> dict:
        """Plan mutation of kernel sizes for Conv2d layers with spatial validation."""
        conv2d_candidates = []
        for name, module in self.original_model.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.source_map:
                module_type = type(module).__name__
                current_kernel = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                if module_type in config.KERNEL_SIZE_MUTATIONS and current_kernel in config.KERNEL_SIZE_MUTATIONS[module_type]:
                    conv2d_candidates.append((name, module, current_kernel))

        if not conv2d_candidates:
            if config.DEBUG_MODE:
                print("[ModelPlanner] No mutable Conv2d layers found for kernel size mutation.")
            return {}

        # Compute spatial dimensions if not already done
        if not self.spatial_tracker:
            self._compute_spatial_dimensions()

        valid_candidates = []
        for name, module, current_kernel in conv2d_candidates:
            possible_mutations = config.KERNEL_SIZE_MUTATIONS[type(module).__name__][current_kernel]
            for new_kernel in possible_mutations:
                # Validate the kernel size maintains valid dimensions
                if self._validate_spatial_change(name, {'kernel_size': new_kernel}):
                    valid_candidates.append((name, module, current_kernel, new_kernel))
        
        if not valid_candidates:
            if config.DEBUG_MODE:
                print("[ModelPlanner] No valid kernel size mutations after spatial validation")
            return {}
            
        target_name, module, current_kernel, new_kernel = random.choice(valid_candidates)

        current_plan = {
            target_name: {
                "mutation_type": "kernel_size",
                "new_kernel_size": new_kernel,
                "source_location": self.source_map.get(target_name)
            }
        }
        self.plan = current_plan
        if config.DEBUG_MODE:
            print(f"[ModelPlanner] Generated kernel size mutation plan: {current_kernel} -> {new_kernel} for {target_name}")
        return self.plan

    def _plan_stride_mutation(self) -> dict:
        """Plan mutation of strides for Conv2d layers with spatial validation."""
        conv2d_candidates = []
        for name, module in self.original_model.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.source_map:
                module_type = type(module).__name__
                current_stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                if module_type in config.STRIDE_MUTATIONS and current_stride in config.STRIDE_MUTATIONS[module_type]:
                    conv2d_candidates.append((name, module, current_stride))

        if not conv2d_candidates:
            if config.DEBUG_MODE:
                print("[ModelPlanner] No mutable Conv2d layers found for stride mutation.")
            return {}

        # Compute spatial dimensions if not already done
        if not self.spatial_tracker:
            self._compute_spatial_dimensions()

        valid_candidates = []
        for name, module, current_stride in conv2d_candidates:
            possible_mutations = config.STRIDE_MUTATIONS[type(module).__name__][current_stride]
            for new_stride in possible_mutations:
                # Validate the stride maintains valid dimensions
                if self._validate_spatial_change(name, {'stride': new_stride}):
                    valid_candidates.append((name, module, current_stride, new_stride))
        
        if not valid_candidates:
            if config.DEBUG_MODE:
                print("[ModelPlanner] No valid stride mutations after spatial validation")
            return {}
            
        target_name, module, current_stride, new_stride = random.choice(valid_candidates)

        current_plan = {
            target_name: {
                "mutation_type": "stride",
                "new_stride": new_stride,
                "source_location": self.source_map.get(target_name)
            }
        }
        self.plan = current_plan
        if config.DEBUG_MODE:
            print(f"[ModelPlanner] Generated stride mutation plan: {current_stride} -> {new_stride} for {target_name}")
        return self.plan

    def _plan_architectural_mutation(self) -> dict:
        """Plan high-level architectural mutations for ConvNeXT-style models."""
        # Check if this is a ConvNeXT-style model first
        if not self._detect_convnext_architecture():
            # Fall back to regular mutations for non-ConvNeXT models
            return self._plan_dimension_mutation()
        
        # Look for high-level architectural patterns in the source map
        architectural_candidates = []
        
        # Find patterns that match high-level architectural parameters
        for name, location in self.source_map.items():
            if location and 'lineno' in location:
                # Look for common architectural patterns in Net.__init__
                if any(pattern in name.lower() for pattern in [
                    'block_setting', 'stage_configs', 'stochastic_depth_prob', 
                    'layer_scale', 'kernel_size', 'stride'
                ]):
                    architectural_candidates.append((name, location))
        
        # If no high-level patterns found, look for fixed parameter assignments
        if not architectural_candidates:
            for name, module in self.original_model.named_modules():
                if name in self.source_map and isinstance(module, (nn.Conv2d, nn.Linear)):
                    # Target parameters that look like architectural choices
                    if any(keyword in name for keyword in ['stem', 'downsample', 'classifier']):
                        architectural_candidates.append((name, self.source_map[name]))
        
        if not architectural_candidates:
            if config.DEBUG_MODE:
                print("[ModelPlanner] No architectural mutation candidates found, falling back to dimension mutation")
            return self._plan_dimension_mutation()
        
        # Choose an architectural parameter to mutate
        target_name, target_location = random.choice(architectural_candidates)
        
        # Determine the type of architectural mutation
        mutation_type = self._determine_architectural_mutation_type(target_name)
        
        current_plan = {
            target_name: {
                "mutation_type": "architectural",
                "architectural_type": mutation_type,
                "source_location": target_location,
                **self._get_architectural_mutation_params(mutation_type)
            }
        }
        
        self.plan = current_plan
        if config.DEBUG_MODE:
            print(f"[ModelPlanner] Generated architectural mutation plan: {mutation_type}")
            import json
            print(json.dumps(self.plan, indent=2))
        return self.plan
    
    def _determine_architectural_mutation_type(self, target_name: str) -> str:
        """Determine what type of architectural mutation to apply."""
        name_lower = target_name.lower()
        
        if 'block_setting' in name_lower or 'stage' in name_lower:
            return 'block_configuration'
        elif 'stochastic_depth' in name_lower:
            return 'stochastic_depth_prob'
        elif 'layer_scale' in name_lower:
            return 'layer_scale'
        elif 'kernel' in name_lower and 'stem' in name_lower:
            return 'stem_kernel_size'
        elif 'stride' in name_lower and 'stem' in name_lower:
            return 'stem_stride'
        else:
            return 'dimension'  # fallback to dimension mutation
    
    def _get_architectural_mutation_params(self, mutation_type: str) -> dict:
        """Get parameters for architectural mutations."""
        params = {}
        
        if mutation_type == 'block_configuration':
            stage_configs = config.ARCHITECTURAL_MUTATIONS['convnext_block_settings']['stage_configs']
            params['new_block_setting'] = random.choice(stage_configs)
        elif mutation_type == 'stochastic_depth_prob':
            params['new_value'] = random.choice(config.ARCHITECTURAL_MUTATIONS['fixed_parameters']['stochastic_depth_prob'])
        elif mutation_type == 'layer_scale':
            params['new_value'] = random.choice(config.ARCHITECTURAL_MUTATIONS['fixed_parameters']['layer_scale'])
        elif mutation_type == 'stem_kernel_size':
            params['new_value'] = random.choice(config.ARCHITECTURAL_MUTATIONS['fixed_parameters']['kernel_sizes'])
        elif mutation_type == 'stem_stride':
            params['new_value'] = random.choice(config.ARCHITECTURAL_MUTATIONS['fixed_parameters']['strides'])
        
        return params

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
        
        # For ConvNeXT models, prioritize architectural mutations
        if (config.PRIORITIZE_ARCHITECTURAL_MUTATIONS and 
            self._detect_convnext_architecture() and 
            random.random() < 0.3):  # 30% chance to force architectural mutation
            return self._plan_fallback_architectural_mutation()
        
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
    
    def _plan_fallback_architectural_mutation(self) -> dict:
        """Plan architectural mutations in fallback mode (FX-incompatible models)."""
        # Prioritize block_setting for ConvNeXT models, then other architectural parameters
        mutation_options = [
            ('block_setting', 'convnext_block_settings'),  # HIGH PRIORITY for ConvNeXT
            ('stochastic_depth_prob', 'stochastic_depth_prob'),
            ('layer_scale', 'layer_scale'),
            ('stem_kernel_size', 'stem_kernel_size'),
            ('stem_stride', 'stem_stride'),
        ]
        
        # Prioritize block_setting for ConvNeXT models (70% chance)
        if self._is_convnext_model() and random.random() < 0.7:
            param_name, mutation_type = 'block_setting', 'convnext_block_settings'
        else:
            # Choose a random architectural mutation
            param_name, mutation_type = random.choice(mutation_options)
        
        # Create a synthetic target since we're doing architectural mutations
        synthetic_target = f"Net.__init__.{param_name}"
        
        current_plan = {
            synthetic_target: {
                "mutation_type": "architectural",
                "architectural_type": mutation_type,
                "source_location": None,  # Will be resolved during code mutation
                **self._get_architectural_mutation_params(mutation_type)
            }
        }
        
        self.plan = current_plan
        if config.DEBUG_MODE:
            print(f"[ModelPlanner] Generated fallback architectural mutation plan: {mutation_type}")
        return self.plan

    def _is_direct_instantiation_location(self, module_name: str) -> bool:
        """Check if a module's source location represents a direct nn.Module instantiation."""
        if module_name not in self.source_map:
            return False
        
        # This is a simplified check - in a more sophisticated implementation,
        # we would parse the AST at the source location to determine if it's a direct call
        # For now, we assume all tracked locations are valid when helper mutations are disabled
        return True

    def _validate_spatial_change(self, module_name, new_params):
        """Validate if mutation maintains valid spatial dimensions"""
        if module_name not in self.spatial_tracker:
            return True  # Skip validation if no dimension info
        
        current_h, current_w = self.spatial_tracker[module_name]
        input_h, input_w = current_h, current_w
        
        # Compute new dimensions based on mutation type
        if 'kernel_size' in new_params:
            kernel_size = new_params['kernel_size']
            stride = new_params.get('stride', self.submodules[module_name].stride)
            padding = new_params.get('padding', self.submodules[module_name].padding)
            dilation = new_params.get('dilation', self.submodules[module_name].dilation)
            
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            if isinstance(stride, int):
                stride = (stride, stride)
            if isinstance(padding, int):
                padding = (padding, padding)
            if isinstance(dilation, int):
                dilation = (dilation, dilation)
                
            new_h = (input_h + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) // stride[0] + 1
            new_w = (input_w + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) // stride[1] + 1
        elif 'stride' in new_params:
            stride = new_params['stride']
            kernel_size = new_params.get('kernel_size', self.submodules[module_name].kernel_size)
            padding = new_params.get('padding', self.submodules[module_name].padding)
            dilation = new_params.get('dilation', self.submodules[module_name].dilation)
            
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            if isinstance(stride, int):
                stride = (stride, stride)
            if isinstance(padding, int):
                padding = (padding, padding)
            if isinstance(dilation, int):
                dilation = (dilation, dilation)
                
            new_h = (input_h + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) // stride[0] + 1
            new_w = (input_w + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) // stride[1] + 1
        else:
            return True  # Non-spatial mutation
            
        # Validate dimensions
        return new_h >= 1 and new_w >= 1
        
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
    
    def _compute_spatial_dimensions(self):
        """Compute spatial dimensions for all layers in the network"""
        current_h, current_w = self.input_shape[1], self.input_shape[2]
        
        for name, module in self.original_model.named_modules():
            if name == '':  # Skip root module
                continue
                
            # Store current dimensions
            self.spatial_tracker[name] = (current_h, current_w)
            
            # Update dimensions based on layer type
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                kernel_size = module.kernel_size
                stride = module.stride
                padding = module.padding
                dilation = module.dilation
                
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size, kernel_size)
                if isinstance(stride, int):
                    stride = (stride, stride)
                if isinstance(padding, int):
                    padding = (padding, padding)
                if isinstance(dilation, int):
                    dilation = (dilation, dilation)
                
                # Compute output dimensions
                current_h = (current_h + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) // stride[0] + 1
                current_w = (current_w + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) // stride[1] + 1
                
                # Ensure valid dimensions
                current_h = max(1, current_h)
                current_w = max(1, current_w)
            elif isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.AdaptiveMaxPool2d):
                # Adaptive pooling sets fixed output size
                if isinstance(module.output_size, int):
                    current_h = module.output_size
                    current_w = module.output_size
                else:
                    current_h, current_w = module.output_size
            elif isinstance(module, nn.Linear):
                # Linear layers flatten spatial dimensions
                current_h, current_w = 1, 1
                
            self.spatial_tracker[name] = (current_h, current_w)

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

    def _should_use_symbolic_mutation(self, module_name: str) -> bool:
        """
        Determine if a mutation should be symbolic based on configuration.
        Respects the MUTATION_MODE setting: 'auto', 'always_symbolic', or 'always_fixed'.
        """
        # Check configuration mode first
        if config.MUTATION_MODE == 'always_symbolic':
            if config.DEBUG_MODE:
                print(f"[ModelPlanner] Using symbolic mutation for {module_name} (always_symbolic mode)")
            return True
            
        if config.MUTATION_MODE == 'always_fixed':
            if config.DEBUG_MODE:
                print(f"[ModelPlanner] Using fixed-number mutation for {module_name} (always_fixed mode)")
            return False
            
        # For 'auto' mode, use weighted probability from SYMBOLIC_MUTATION_WEIGHTS
        choices = ['symbolic', 'fixed']
        weights = [config.SYMBOLIC_MUTATION_WEIGHTS['symbolic'], config.SYMBOLIC_MUTATION_WEIGHTS['fixed']]
        decision = random.choices(choices, weights=weights, k=1)[0]
        
        if decision == 'symbolic':
            if config.DEBUG_MODE:
                print(f"[ModelPlanner] Using symbolic mutation for {module_name} (weighted probability)")
            return True
        else:
            if config.DEBUG_MODE:
                print(f"[ModelPlanner] Using fixed-number mutation for {module_name} (weighted probability)")
            return False

    def _generate_symbolic_expression(self, module_name: str, target_value: int) -> str:
        """
        Generate a symbolic expression for parameter-dependent mutations.
        Creates expressions like 'in_channels * 2' or 'planes * 4' based on context.
        """
        if module_name not in self.source_map:
            return str(target_value)
            
        source_location = self.source_map[module_name]
        if not source_location or 'lineno' not in source_location:
            return str(target_value)
            
        # Get the source code for context analysis
        source_code = self._get_source_code_for_location(source_location)
        if not source_code:
            return str(target_value)
            
        # Find the AST call node at this location
        call_node = self._find_call_node_at_line(source_code, source_location['lineno'])
        if not call_node:
            return str(target_value)
            
        # Get available parameters in the current context
        available_params = get_available_parameters(call_node, source_code)
        
        # Try to find a meaningful symbolic expression
        symbolic_expr = self._create_symbolic_expression(target_value, available_params)
        
        if config.DEBUG_MODE:
            print(f"[ModelPlanner] Generated symbolic expression for {module_name}: {symbolic_expr}")
            
        return symbolic_expr

    def _get_source_code_for_location(self, source_location: dict) -> str:
        """Get source code for the given location by reading the source file."""
        if not source_location or 'filename' not in source_location:
            if config.DEBUG_MODE:
                print(f"[ModelPlanner] No filename in source_location: {source_location}")
            return ""
        
        filename = source_location['filename']
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except (IOError, OSError, UnicodeDecodeError) as e:
            if config.DEBUG_MODE:
                print(f"[ModelPlanner] Failed to read source file {filename}: {e}")
            return ""

    def _create_mock_frame_info(self, source_location: dict, module_name: str) -> object:
        """Create a mock frame info object for context analysis."""
        class MockFrameInfo:
            def __init__(self, lineno, function_name):
                self.lineno = lineno
                self.function = function_name
                self.code_context = [f"# Mock frame for {module_name} at line {lineno}"]
                
        # Extract function name from module path (e.g., "Net.conv1" -> "Net")
        function_name = module_name.split('.')[0] if '.' in module_name else "__init__"
        return MockFrameInfo(source_location['lineno'], function_name)

    def _find_call_node_at_line(self, source_code: str, lineno: int):
        """Find the AST call node at the given line number."""
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and 
                    hasattr(node, 'lineno') and 
                    node.lineno == lineno):
                    return node
        except (SyntaxError, AttributeError):
            pass
        return None

    def _create_symbolic_expression(self, target_value: int, available_params: list) -> str:
        """
        Create free symbolic combinations that evaluate to the target value.
        Generates complex expressions using multiple operations and parameters.
        """
        if not available_params:
            return str(target_value)
            
        # Common neural network parameter patterns (prioritize these)
        common_params = ['in_channels', 'out_channels', 'planes', 'width', 'depth', 'expansion']
        
        # Filter available params to prioritize common ones
        prioritized_params = [p for p in available_params if p in common_params]
        if not prioritized_params:
            prioritized_params = available_params
        
        # Try multiple times to create a meaningful expression
        for attempt in range(5):
            # Choose expression complexity (1-3 operations)
            complexity = random.randint(1, 3)
            
            if complexity == 1:
                # Simple expression: param op value
                param = random.choice(prioritized_params)
                operation = random.choice(config.SYMBOLIC_OPERATIONS)
                operand = random.choice(config.SYMBOLIC_OPERANDS)
                
                # For division operations, ensure it's meaningful
                if operation in ['//', '/'] and operand == 0:
                    continue
                    
                expression = f"{param} {operation} {operand}"
                
            elif complexity == 2:
                # Medium complexity: (param1 op1 value1) op2 (param2 op3 value2) or similar
                param1 = random.choice(prioritized_params)
                param2 = random.choice(prioritized_params)
                op1 = random.choice(config.SYMBOLIC_OPERATIONS)
                op2 = random.choice(['+', '-', '*', '//'])
                op3 = random.choice(config.SYMBOLIC_OPERATIONS)
                val1 = random.choice(config.SYMBOLIC_OPERANDS)
                val2 = random.choice(config.SYMBOLIC_OPERANDS)
                
                # Avoid division by zero
                if (op1 in ['//', '/'] and val1 == 0) or (op3 in ['//', '/'] and val2 == 0):
                    continue
                
                # Choose expression pattern randomly
                patterns = [
                    f"({param1} {op1} {val1}) {op2} ({param2} {op3} {val2})",
                    f"{param1} {op1} {val1} {op2} {param2} {op3} {val2}",
                    f"({param1} {op2} {param2}) {op1} {val1}",
                    f"{param1} {op1} ({val1} {op2} {val2})"
                ]
                expression = random.choice(patterns)
                
            else:
                # High complexity: nested expressions
                param1 = random.choice(prioritized_params)
                param2 = random.choice(prioritized_params)
                param3 = random.choice(prioritized_params) if len(prioritized_params) > 2 else param1
                
                op1 = random.choice(config.SYMBOLIC_OPERATIONS)
                op2 = random.choice(config.SYMBOLIC_OPERATIONS)
                op3 = random.choice(config.SYMBOLIC_OPERATIONS)
                val1 = random.choice(config.SYMBOLIC_OPERANDS)
                val2 = random.choice(config.SYMBOLIC_OPERANDS)
                val3 = random.choice(config.SYMBOLIC_OPERANDS)
                
                # Avoid division by zero
                if any(op in ['//', '/'] and val == 0 for op, val in [(op1, val1), (op2, val2), (op3, val3)]):
                    continue
                
                patterns = [
                    f"({param1} {op1} {val1}) {op2} ({param2} {op3} {val2}) + {param3}",
                    f"{param1} {op1} {val1} * ({param2} {op2} {val2}) // {val3}",
                    f"({param1} + {param2}) {op1} {val1} - {param3} {op2} {val2}",
                    f"{param1} << {val1} {op1} {param2} >> {val2}"
                ]
                expression = random.choice(patterns)
            
            # Try to evaluate the expression safely to check if it's reasonable
            try:
                # Create a mock environment with reasonable parameter values
                # Use the target_value as a baseline for parameter values
                mock_env = {}
                for param in set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)):
                    if param not in ['and', 'or', 'not', 'if', 'else', 'True', 'False', 'None']:
                        # Assign a reasonable value based on target_value
                        mock_env[param] = max(8, min(1024, target_value // random.randint(1, 8)))
                
                # Evaluate the expression
                result = eval(expression, {}, mock_env)
                
                # Check if the result is close to the target value and reasonable
                if (isinstance(result, (int, float)) and 
                    result > 0 and 
                    abs(result - target_value) <= target_value * 0.5 and  # Allow 50% variance
                    8 <= result <= 2048):  # Reasonable range for neural network parameters
                    return expression
                    
            except (SyntaxError, NameError, TypeError, ZeroDivisionError, ValueError):
                continue
        
        # Fallback: try simple expressions with common parameters
        for param in prioritized_params:
            for operation in config.SYMBOLIC_OPERATIONS:
                for operand in config.SYMBOLIC_OPERANDS:
                    if operation in ['//', '/'] and operand == 0:
                        continue
                        
                    expression = f"{param} {operation} {operand}"
                    try:
                        mock_env = {param: max(8, min(1024, target_value // 2))}
                        result = eval(expression, {}, mock_env)
                        if (isinstance(result, (int, float)) and 
                            result > 0 and 
                            abs(result - target_value) <= target_value * 0.3 and
                            8 <= result <= 1024):
                            return expression
                    except:
                        continue
        
        # Final fallback: return target value as string
        return str(target_value)
