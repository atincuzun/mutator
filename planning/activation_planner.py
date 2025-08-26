"""
Activation planner for handling activation function mutations.

This module handles swapping activation functions in neural networks,
ensuring compatibility and maintaining proper semantics.
"""

import random
import json
from typing import Dict, List, Tuple, Any

import torch.nn as nn

from mutator import config


class ActivationPlanner:
    """
    Planner for activation function mutations.
    
    This class handles mutations that swap activation functions
    while maintaining network compatibility and semantics.
    """
    
    def __init__(self, model_planner):
        """
        Initialize the activation planner.
        
        Args:
            model_planner: Reference to the main ModelPlanner instance
        """
        self.model_planner = model_planner

    def plan_activation_mutation(self) -> Dict[str, Any]:
        """
        Plan mutation of activation functions.
        
        Returns:
            Dictionary containing the activation mutation plan
        """
        activation_candidates = self._find_activation_candidates()
        
        if not activation_candidates:
            if config.DEBUG_MODE:
                print(f"[ActivationPlanner] No mutable activation functions found (helper_mutations={config.ALLOW_HELPER_FUNCTION_MUTATIONS})")
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
                "source_location": self.model_planner.source_map.get(target_name)
            }
        }
        
        self.model_planner.plan = current_plan
        if config.DEBUG_MODE:
            print(f"[ActivationPlanner] Generated activation mutation plan: {current_activation} -> {new_activation}")
            print(json.dumps(current_plan, indent=2))
        return current_plan

    def _find_activation_candidates(self) -> List[Tuple[str, str]]:
        """
        Find all activation function modules that can be mutated.
        
        Returns:
            List of tuples containing (module_name, activation_type)
        """
        activation_candidates = []
        
        # Find all activation function modules
        for name, module in self.model_planner.original_model.named_modules():
            if isinstance(module, self.model_planner.ACTIVATION_MODULES):
                module_type = type(module).__name__
                if module_type in config.ACTIVATION_MUTATIONS:
                    # Check if this module has a valid source location
                    if name in self.model_planner.source_map:
                        # When helper mutations are disabled, ensure this is a direct instantiation
                        if not config.ALLOW_HELPER_FUNCTION_MUTATIONS:
                            # Check if the source location represents a direct instantiation
                            if self._is_direct_instantiation_location(name):
                                activation_candidates.append((name, module_type))
                        else:
                            activation_candidates.append((name, module_type))
        
        return activation_candidates

    def _is_direct_instantiation_location(self, module_name: str) -> bool:
        """
        Check if a module's source location represents a direct nn.Module instantiation.
        
        Args:
            module_name: Name of the module to check
            
        Returns:
            True if the module represents a direct instantiation
        """
        if module_name not in self.model_planner.source_map:
            return False
        
        # This is a simplified check - in a more sophisticated implementation,
        # we would parse the AST at the source location to determine if it's a direct call
        # For now, we assume all tracked locations are valid when helper mutations are disabled
        return True

    def plan_fallback_activation_mutation(self, target_name: str, target_module: nn.Module) -> Dict[str, Any]:
        """
        Plan activation mutation without FX graph analysis.
        
        Args:
            target_name: Name of the target module
            target_module: The module to mutate
            
        Returns:
            Dictionary containing the activation mutation plan
        """
        module_type = type(target_module).__name__
        possible_mutations = config.ACTIVATION_MUTATIONS[module_type]
        new_activation = random.choice(possible_mutations)
        
        current_plan = {
            target_name: {
                "mutation_type": "activation",
                "current_activation": module_type,
                "new_activation": new_activation,
                "source_location": self.model_planner.source_map.get(target_name)
            }
        }
        
        self.model_planner.plan = current_plan
        if config.DEBUG_MODE:
            print(f"[ActivationPlanner] Generated fallback activation mutation plan: {module_type} -> {new_activation}")
        return current_plan

    def get_activation_compatibility_score(self, current_activation: str, new_activation: str) -> float:
        """
        Get a compatibility score between two activation functions.
        
        Args:
            current_activation: Current activation function name
            new_activation: Proposed new activation function name
            
        Returns:
            Compatibility score between 0.0 and 1.0 (higher is more compatible)
        """
        # Define compatibility groups
        similar_groups = [
            ['ReLU', 'LeakyReLU', 'ELU'],  # ReLU family
            ['Tanh', 'Sigmoid'],  # Sigmoid family  
            ['GELU', 'SiLU'],  # Smooth activations
        ]
        
        # Same activation = perfect compatibility
        if current_activation == new_activation:
            return 1.0
        
        # Check if they're in the same compatibility group
        for group in similar_groups:
            if current_activation in group and new_activation in group:
                return 0.8
        
        # Different groups = lower compatibility
        return 0.5

    def validate_activation_mutation(self, target_name: str, new_activation: str) -> bool:
        """
        Validate if an activation mutation is safe to apply.
        
        Args:
            target_name: Name of the target module
            new_activation: Proposed new activation function
            
        Returns:
            True if the mutation is safe to apply
        """
        # Basic validation - ensure the activation is supported
        supported_activations = ['ReLU', 'GELU', 'ELU', 'LeakyReLU', 'SiLU', 'Tanh', 'Sigmoid']
        if new_activation not in supported_activations:
            return False
        
        # Check if the module exists
        if target_name not in self.model_planner.submodules:
            return False
        
        # Additional model-specific validations could be added here
        return True
