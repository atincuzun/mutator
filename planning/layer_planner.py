"""
Layer type planner for handling normalization and pooling layer mutations.

This module handles mutations that change layer types while maintaining
compatibility with the network architecture.
"""

import random
import json
from typing import Dict, Any

import torch.nn as nn

from mutator import config


class LayerTypePlanner:
    """
    Planner for layer type mutations (normalization, pooling).
    
    This class handles mutations that swap layer types like normalization
    and pooling layers while maintaining network compatibility.
    """
    
    def __init__(self, model_planner):
        """
        Initialize the layer type planner.
        
        Args:
            model_planner: Reference to the main ModelPlanner instance
        """
        self.model_planner = model_planner

    def plan_layer_type_mutation(self) -> Dict[str, Any]:
        """
        Plan mutation of layer types (normalization, pooling).
        
        Returns:
            Dictionary containing the layer type mutation plan
        """
        layer_candidates = self._find_layer_candidates()
        
        if not layer_candidates:
            if config.DEBUG_MODE:
                print("[LayerTypePlanner] No mutable layer types found")
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
                "source_location": self.model_planner.source_map.get(target_name)
            }
        }
        
        self.model_planner.plan = current_plan
        if config.DEBUG_MODE:
            print(f"[LayerTypePlanner] Generated layer type mutation plan: {current_layer_type} -> {new_layer_type}")
            print(json.dumps(current_plan, indent=2))
        return current_plan

    def _find_layer_candidates(self):
        """Find all mutable layer types."""
        layer_candidates = []
        
        for name, module in self.model_planner.original_model.named_modules():
            module_type = type(module).__name__
            if module_type in config.LAYER_TYPE_MUTATIONS:
                # Check if this module has a valid source location
                if name in self.model_planner.source_map:
                    layer_candidates.append((name, module_type, module))
        
        return layer_candidates

    def _extract_layer_params(self, module: nn.Module, current_type: str, new_type: str) -> Dict[str, Any]:
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

    def plan_fallback_layer_type_mutation(self, target_name: str, target_module: nn.Module) -> Dict[str, Any]:
        """
        Plan layer type mutation without FX graph analysis.
        
        Args:
            target_name: Name of the target module
            target_module: The module to mutate
            
        Returns:
            Dictionary containing the layer type mutation plan
        """
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
                "source_location": self.model_planner.source_map.get(target_name)
            }
        }
        
        self.model_planner.plan = current_plan
        if config.DEBUG_MODE:
            print(f"[LayerTypePlanner] Generated fallback layer type mutation plan: {module_type} -> {new_layer_type}")
        return current_plan

    def validate_layer_type_mutation(self, current_type: str, new_type: str, module: nn.Module) -> bool:
        """
        Validate if a layer type mutation is compatible.
        
        Args:
            current_type: Current layer type
            new_type: Proposed new layer type  
            module: The module to be mutated
            
        Returns:
            True if the mutation is valid
        """
        # Basic compatibility checks
        normalization_layers = ['BatchNorm2d', 'GroupNorm', 'LayerNorm', 'InstanceNorm2d']
        pooling_layers = ['MaxPool2d', 'AvgPool2d', 'AdaptiveMaxPool2d', 'AdaptiveAvgPool2d']
        
        # Normalization to normalization is generally safe
        if current_type in normalization_layers and new_type in normalization_layers:
            return True
            
        # Pooling to pooling is generally safe
        if current_type in pooling_layers and new_type in pooling_layers:
            return True
            
        # Cross-category mutations may need special handling
        return False
