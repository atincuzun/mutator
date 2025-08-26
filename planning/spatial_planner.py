"""
Spatial planner for handling kernel size and stride mutations on Conv2d.

Validates spatial dimensions using the model planner's spatial tracker.
"""

import random
import json
from typing import Dict, Any, List, Tuple

import torch.nn as nn

from mutator import config


class SpatialPlanner:
    """
    Planner for spatial parameter mutations (kernel_size, stride) on Conv2d layers.
    """

    def __init__(self, model_planner):
        self.model_planner = model_planner

    def plan_kernel_size_mutation(self) -> Dict[str, Any]:
        """Plan mutation of kernel sizes for Conv2d layers with spatial validation."""
        conv2d_candidates: List[Tuple[str, nn.Conv2d, int]] = []
        for name, module in self.model_planner.original_model.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.model_planner.source_map:
                module_type = type(module).__name__
                current_kernel = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                if (
                    hasattr(config, 'KERNEL_SIZE_MUTATIONS')
                    and module_type in config.KERNEL_SIZE_MUTATIONS
                    and current_kernel in config.KERNEL_SIZE_MUTATIONS[module_type]
                ):
                    conv2d_candidates.append((name, module, current_kernel))

        if not conv2d_candidates:
            if config.DEBUG_MODE:
                print("[SpatialPlanner] No mutable Conv2d layers found for kernel size mutation.")
            return {}

        # Compute spatial dimensions if not already done
        if not self.model_planner.spatial_tracker:
            self.model_planner._compute_spatial_dimensions()

        valid_candidates: List[Tuple[str, nn.Conv2d, int, int]] = []
        for name, module, current_kernel in conv2d_candidates:
            possible_mutations = config.KERNEL_SIZE_MUTATIONS[type(module).__name__][current_kernel]
            for new_kernel in possible_mutations:
                # Validate the kernel size maintains valid dimensions
                if self.model_planner._validate_spatial_change(name, {'kernel_size': new_kernel}):
                    valid_candidates.append((name, module, current_kernel, new_kernel))

        if not valid_candidates:
            if config.DEBUG_MODE:
                print("[SpatialPlanner] No valid kernel size mutations after spatial validation")
            return {}

        target_name, module, current_kernel, new_kernel = random.choice(valid_candidates)

        current_plan = {
            target_name: {
                "mutation_type": "kernel_size",
                "new_kernel_size": new_kernel,
                "source_location": self.model_planner.source_map.get(target_name),
            }
        }
        self.model_planner.plan = current_plan
        if config.DEBUG_MODE:
            print(
                f"[SpatialPlanner] Generated kernel size mutation plan: {current_kernel} -> {new_kernel} for {target_name}"
            )
            print(json.dumps(current_plan, indent=2))
        return current_plan

    def plan_stride_mutation(self) -> Dict[str, Any]:
        """Plan mutation of strides for Conv2d layers with spatial validation."""
        conv2d_candidates: List[Tuple[str, nn.Conv2d, int]] = []
        for name, module in self.model_planner.original_model.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.model_planner.source_map:
                module_type = type(module).__name__
                current_stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                if (
                    hasattr(config, 'STRIDE_MUTATIONS')
                    and module_type in config.STRIDE_MUTATIONS
                    and current_stride in config.STRIDE_MUTATIONS[module_type]
                ):
                    conv2d_candidates.append((name, module, current_stride))

        if not conv2d_candidates:
            if config.DEBUG_MODE:
                print("[SpatialPlanner] No mutable Conv2d layers found for stride mutation.")
            return {}

        # Compute spatial dimensions if not already done
        if not self.model_planner.spatial_tracker:
            self.model_planner._compute_spatial_dimensions()

        valid_candidates: List[Tuple[str, nn.Conv2d, int, int]] = []
        for name, module, current_stride in conv2d_candidates:
            possible_mutations = config.STRIDE_MUTATIONS[type(module).__name__][current_stride]
            for new_stride in possible_mutations:
                # Validate the stride maintains valid dimensions
                if self.model_planner._validate_spatial_change(name, {'stride': new_stride}):
                    valid_candidates.append((name, module, current_stride, new_stride))

        if not valid_candidates:
            if config.DEBUG_MODE:
                print("[SpatialPlanner] No valid stride mutations after spatial validation")
            return {}

        target_name, module, current_stride, new_stride = random.choice(valid_candidates)

        current_plan = {
            target_name: {
                "mutation_type": "stride",
                "new_stride": new_stride,
                "source_location": self.model_planner.source_map.get(target_name),
            }
        }
        self.model_planner.plan = current_plan
        if config.DEBUG_MODE:
            print(
                f"[SpatialPlanner] Generated stride mutation plan: {current_stride} -> {new_stride} for {target_name}"
            )
            print(json.dumps(current_plan, indent=2))
        return current_plan
