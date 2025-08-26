"""
Fallback planner for FX-incompatible models.

Provides safe mutations using module inspection only (no FX graph).
"""

import random
import json
from typing import Dict, Any, List, Tuple

import torch.nn as nn

from mutator import config


class FallbackPlanner:
    """Planner for mutations when FX tracing is not available."""

    def __init__(self, model_planner):
        self.model_planner = model_planner

    def plan_fallback_mutation(self) -> Dict[str, Any]:
        """Plan mutations for FX-incompatible models using module inspection only."""
        mutation_candidates: List[Tuple[str, nn.Module, str]] = []

        # Find mutable modules that we can safely mutate
        for name, module in self.model_planner.original_model.named_modules():
            if name in self.model_planner.source_map:
                module_type = type(module).__name__
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    mutation_candidates.append((name, module, 'dimension'))
                elif module_type in getattr(config, 'ACTIVATION_MUTATIONS', {}):
                    mutation_candidates.append((name, module, 'activation'))
                elif module_type in getattr(config, 'LAYER_TYPE_MUTATIONS', {}):
                    mutation_candidates.append((name, module, 'layer_type'))

        if not mutation_candidates:
            if config.DEBUG_MODE:
                print("[FallbackPlanner] No mutation candidates found in fallback mode")
            return {}

        # Choose a random mutation
        target_name, target_module, mutation_type = random.choice(mutation_candidates)

        # For ConvNeXT models, prioritize architectural mutations
        if (
            getattr(config, 'PRIORITIZE_ARCHITECTURAL_MUTATIONS', False)
            and self._is_convnext_model()
            and random.random() < 0.3
        ):
            return self._plan_fallback_architectural_mutation()

        if mutation_type == 'dimension':
            return self._plan_fallback_dimension_mutation(target_name, target_module)
        elif mutation_type == 'activation':
            return self.model_planner.activation_planner.plan_fallback_activation_mutation(target_name, target_module)
        elif mutation_type == 'layer_type':
            return self.model_planner.layer_planner.plan_fallback_layer_type_mutation(target_name, target_module)

        return {}

    def _plan_fallback_dimension_mutation(self, target_name: str, target_module: nn.Module) -> Dict[str, Any]:
        """Plan dimension mutation without FX graph analysis."""
        if isinstance(target_module, nn.Conv2d):
            original_dim = target_module.out_channels
        elif isinstance(target_module, nn.Linear):
            original_dim = target_module.out_features
        else:
            return {}

        valid_new_sizes = [s for s in self.model_planner.VALID_CHANNEL_SIZES if s != original_dim]
        if not valid_new_sizes:
            return {}

        new_dim = random.choice(valid_new_sizes)

        current_plan = {
            target_name: {
                "mutation_type": "dimension",
                "new_out": new_dim,
                "new_in": None,
                "source_location": self.model_planner.source_map.get(target_name),
            }
        }

        self.model_planner.plan = current_plan
        if config.DEBUG_MODE:
            print(f"[FallbackPlanner] Generated fallback dimension mutation plan for {target_name}")
            print(json.dumps(current_plan, indent=2))
        return current_plan

    def _plan_fallback_architectural_mutation(self) -> Dict[str, Any]:
        """Plan architectural mutations in fallback mode (FX-incompatible models)."""
        mutation_options = [
            ('block_setting', 'convnext_block_settings'),
            ('stochastic_depth_prob', 'stochastic_depth_prob'),
            ('layer_scale', 'layer_scale'),
            ('stem_kernel_size', 'stem_kernel_size'),
            ('stem_stride', 'stem_stride'),
        ]

        if self._is_convnext_model() and random.random() < 0.7:
            param_name, mutation_type = 'block_setting', 'convnext_block_settings'
        else:
            param_name, mutation_type = random.choice(mutation_options)

        synthetic_target = f"Net.__init__.{param_name}"

        current_plan = {
            synthetic_target: {
                "mutation_type": "architectural",
                "architectural_type": mutation_type,
                "source_location": None,
                **self._get_architectural_mutation_params(mutation_type),
            }
        }

        self.model_planner.plan = current_plan
        if config.DEBUG_MODE:
            print(f"[FallbackPlanner] Generated fallback architectural mutation plan: {mutation_type}")
            print(json.dumps(current_plan, indent=2))
        return current_plan

    def _get_architectural_mutation_params(self, mutation_type: str) -> dict:
        params: Dict[str, Any] = {}
        if mutation_type == 'convnext_block_settings':
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

    def _is_convnext_model(self) -> bool:
        """Heuristic detection of ConvNeXT-like models."""
        indicators = ['StochasticDepth', 'LayerNorm2d', 'Permute', 'CNBlock']
        for _, module in self.model_planner.original_model.named_modules():
            if type(module).__name__ in indicators:
                return True
            if isinstance(module, nn.Conv2d) and getattr(module, 'groups', 1) == getattr(module, 'in_channels', -1):
                return True
        return False
