"""
Architectural planner for high-level architectural mutations (ConvNeXT-style, etc.).
"""

import random
import json
from typing import Dict, Any, List, Tuple

import torch.nn as nn

from mutator import config


class ArchitecturalPlanner:
    """Planner for high-level architectural mutations."""

    def __init__(self, model_planner):
        self.model_planner = model_planner

    def plan_architectural_mutation(self) -> Dict[str, Any]:
        """Plan high-level architectural mutations for ConvNeXT-style models."""
        # Check if this is a ConvNeXT-style model first
        if not self._detect_convnext_architecture():
            # Fall back to regular mutations for non-ConvNeXT models
            return self.model_planner.dimension_planner.plan_dimension_mutation()

        # Look for high-level architectural patterns in the source map
        architectural_candidates: List[Tuple[str, dict]] = []

        # Find patterns that match high-level architectural parameters
        for name, location in self.model_planner.source_map.items():
            if location and 'lineno' in location:
                lname = name.lower()
                if any(k in lname for k in ['block_setting', 'stage', 'stochastic_depth', 'layer_scale', 'stem']):
                    architectural_candidates.append((name, location))

        # If no high-level patterns found, look for fixed parameter assignments
        if not architectural_candidates:
            for name, location in self.model_planner.source_map.items():
                if location and 'lineno' in location:
                    lname = name.lower()
                    if any(k in lname for k in ['kernel', 'stride']):
                        architectural_candidates.append((name, location))

        if not architectural_candidates:
            if config.DEBUG_MODE:
                print("[ArchitecturalPlanner] No architectural candidates found; falling back to dimension mutation")
            return self.model_planner.dimension_planner.plan_dimension_mutation()

        # Choose an architectural parameter to mutate
        target_name, target_location = random.choice(architectural_candidates)

        # Determine the type of architectural mutation
        mutation_type = self._determine_architectural_mutation_type(target_name)

        current_plan = {
            target_name: {
                "mutation_type": "architectural",
                "architectural_type": mutation_type,
                "source_location": target_location,
                **self._get_architectural_mutation_params(mutation_type),
            }
        }

        self.model_planner.plan = current_plan
        if config.DEBUG_MODE:
            print(json.dumps(self.model_planner.plan, indent=2))
        return current_plan

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
        params: Dict[str, Any] = {}

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

    def _detect_convnext_architecture(self) -> bool:
        """Detect if this is a ConvNeXT-style architecture based on characteristic modules."""
        # Look for characteristic ConvNeXT modules
        convnext_indicators = [
            'StochasticDepth', 'LayerNorm2d', 'Permute', 'CNBlock'
        ]

        found_indicator = False
        for name, module in self.model_planner.original_model.named_modules():
            module_type = type(module).__name__
            if module_type in convnext_indicators:
                found_indicator = True

            # Also check for depth-wise convolutions (groups == in_channels)
            if isinstance(module, nn.Conv2d) and hasattr(module, 'groups'):
                if module.groups == getattr(module, 'in_channels', -1):
                    found_indicator = True

        return found_indicator
