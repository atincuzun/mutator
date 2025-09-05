"""
Pattern detection utilities for neural network architecture analysis.
Provides tools to identify block factories and architectural patterns.
"""

import ast
import torch
import torch.nn as nn
import torch.fx as fx
from typing import Dict, List, Set, Optional, Tuple
import re


class ArchitecturePatternDetector:
    """
    Detects architectural patterns in neural network models.
    Enhanced to track block instances and generate novel architectures.
    """
    
    def __init__(self):
        self.block_factory_methods = {
            '_make_layer', 'make_layer', '_build_block', 'build_block',
            '_create_stage', 'create_stage', '_make_residual_block'
        }
        
        self.block_parameter_names = {
            'planes', 'width', 'features', 'channels', 'filters',
            'out_channels', 'in_channels', 'embed_dim', 'hidden_size'
        }
        
        # Track block instances created by factory methods
        self.block_instance_groups = {}  # Maps factory call -> list of module names
    
    def find_block_instance_groups(self, model, graph: fx.Graph) -> Dict[str, List[str]]:
        """
        Track which modules belong to the same block factory call.
        
        This is the key method that solves your problem by grouping modules
        that were created by the same _make_layer() call.
        
        Returns:
            Dict mapping factory_call_id -> list of module names in that group
        """
        # Get all named modules for reference
        named_modules = dict(model.named_modules())
        
        # Find all sequential containers that likely came from _make_layer calls
        block_groups = {}
        
        for name, module in named_modules.items():
            # Look for sequential patterns that indicate block factory output
            if isinstance(module, torch.nn.Sequential):
                # Check if this sequential contains blocks with similar patterns
                if self._is_likely_block_factory_output(module, name):
                    group_id = f"factory_{name}"
                    block_groups[group_id] = self._extract_block_modules(module, name)
        
        # Also detect by naming patterns (ResNet style: layer1.0, layer1.1, etc.)
        pattern_groups = self._detect_by_naming_patterns(named_modules)
        block_groups.update(pattern_groups)
        
        return block_groups
    
    def _is_likely_block_factory_output(self, sequential_module, name: str) -> bool:
        """Check if a Sequential module looks like output from _make_layer."""
        # ResNet pattern: layer1, layer2, etc.
        if re.match(r'layer\d+', name):
            return True
            
        # Check if children have consistent block patterns
        children = list(sequential_module.children())
        if len(children) >= 2:
            # All children should be instances of the same block type
            first_type = type(children[0])
            if all(isinstance(child, first_type) for child in children):
                return True
        
        return False
    
    def _extract_block_modules(self, sequential_module, base_name: str) -> List[str]:
        """Extract all module names within a block factory's output."""
        module_names = []
        
        # Get all submodules of this sequential
        for child_name, child_module in sequential_module.named_modules():
            if child_name:  # Skip the root module itself
                full_name = f"{base_name}.{child_name}"
                module_names.append(full_name)
        
        return module_names
    
    def _detect_by_naming_patterns(self, named_modules: Dict) -> Dict[str, List[str]]:
        """Detect block groups by naming patterns (layer1.0, layer1.1, etc.)."""
        pattern_groups = {}
        
        # Group by layer patterns
        layer_pattern = re.compile(r'(layer\d+)\.(\d+)')
        layer_blocks = {}
        
        for name, module in named_modules.items():
            match = layer_pattern.match(name)
            if match:
                layer_name, block_idx = match.groups()
                
                if layer_name not in layer_blocks:
                    layer_blocks[layer_name] = []
                
                # Get all submodules of this block
                block_modules = []
                for sub_name, sub_module in named_modules.items():
                    if sub_name.startswith(f"{layer_name}.{block_idx}."):
                        block_modules.append(sub_name)
                
                if block_modules:
                    group_id = f"block_{layer_name}_{block_idx}"
                    pattern_groups[group_id] = block_modules
        
        return pattern_groups
        """
        Find all block factory method calls in the FX graph.
        
        Returns:
            List of FX nodes that represent block factory calls
        """
        factory_calls = []
        
        for node in graph.nodes:
            if self._is_block_factory_call(node):
                factory_calls.append(node)
        
        return factory_calls
    
    def _is_block_factory_call(self, node: fx.Node) -> bool:
        """
        Check if a node represents a block factory method call.
        """
        # Method calls (e.g., self._make_layer)
        if node.op == 'call_method' and node.target in self.block_factory_methods:
            return True
        
        # Function calls with known factory patterns
        if node.op == 'call_function' and hasattr(node.target, '__name__'):
            if node.target.__name__ in self.block_factory_methods:
                return True
        
        return False
    
    def extract_factory_parameters(self, node: fx.Node) -> Dict[str, any]:
        """
        Extract parameters from a block factory call.
        
        Returns:
            Dictionary of parameter names and their values/expressions
        """
        params = {}
        
        # Extract positional arguments
        for i, arg in enumerate(node.args):
            params[f'arg_{i}'] = arg
        
        # Extract keyword arguments
        if hasattr(node, 'kwargs'):
            for kwarg in node.kwargs:
                params[kwarg.arg] = kwarg.value
        
        return params
    
    def is_parameter_symbolic_candidate(self, param_name: str, param_value: any) -> bool:
        """
        Check if a parameter is a good candidate for symbolic mutation.
        """
        # Skip non-symbolic values (constants, etc.)
        if not isinstance(param_value, (ast.Name, ast.Attribute, ast.BinOp)):
            return False
        
        # Check if parameter name suggests it's architectural
        param_name_lower = param_name.lower()
        if any(pattern in param_name_lower for pattern in ['planes', 'width', 'channel', 'feature', 'dim']):
            return True
        
        return False
    
    def find_downstream_consumers(self, graph: fx.Graph, producer_node: fx.Node) -> List[fx.Node]:
        """
        Find all nodes that consume the output of a producer node.
        Useful for tracking parameter propagation.
        """
        consumers = []
        
        for node in graph.nodes:
            if producer_node in node.all_input_nodes:
                consumers.append(node)
        
        return consumers
    
    def detect_architecture_constraints(self, graph: fx.Graph) -> Dict[str, any]:
        """
        Detect architectural constraints that should be preserved during mutation.
        """
        constraints = {
            'attention_heads_divisible': False,
            'group_conv_constraints': False,
            'embedding_dim_constraints': False
        }
        
        # Check for attention mechanisms
        for node in graph.nodes:
            if node.op == 'call_module':
                module_name = str(node.target).lower()
                if 'attention' in module_name or 'mha' in module_name:
                    constraints['attention_heads_divisible'] = True
        
        # Check for group convolutions
        for node in graph.nodes:
            if node.op == 'call_module' and hasattr(node, 'kwargs'):
                for kwarg in node.kwargs:
                    if kwarg.arg == 'groups' and kwarg.value != 1:
                        constraints['group_conv_constraints'] = True
        
        return constraints
    
    def generate_novel_block_mutations(self, block_group: List[str], base_params: Dict[str, any]) -> Dict[str, Dict]:
        """
        Generate novel architectural mutations for a block group.
        
        This creates genuinely different architectures like bottlenecks, 
        inverted bottlenecks, squeeze-excite patterns, etc.
        
        Args:
            block_group: List of module names in the same block instance
            base_params: Base parameters (like 'planes') available for expressions
        
        Returns:
            Dict mapping module_name -> mutation plan
        """
        mutation_plans = {}
        
        # Identify the structure of this block
        conv_layers = [name for name in block_group if '.conv' in name]
        bn_layers = [name for name in block_group if '.bn' in name or 'norm' in name]
        downsample_layers = [name for name in block_group if 'downsample' in name]
        
        if len(conv_layers) >= 2:  # Typical ResNet block
            # Generate various novel patterns
            pattern_type = self._select_novel_pattern()
            
            if pattern_type == "bottleneck":
                mutation_plans.update(self._generate_bottleneck_pattern(conv_layers, bn_layers, base_params))
            elif pattern_type == "inverted_bottleneck":
                mutation_plans.update(self._generate_inverted_bottleneck_pattern(conv_layers, bn_layers, base_params))
            elif pattern_type == "squeeze_excite":
                mutation_plans.update(self._generate_squeeze_excite_pattern(conv_layers, bn_layers, base_params))
            elif pattern_type == "asymmetric_expansion":
                mutation_plans.update(self._generate_asymmetric_expansion_pattern(conv_layers, bn_layers, base_params))
        
        # Handle downsample layers to maintain residual connection validity
        if downsample_layers and conv_layers:
            final_conv = conv_layers[-1]  # Last conv layer determines output
            for ds_layer in downsample_layers:
                if final_conv in mutation_plans:
                    final_output = mutation_plans[final_conv].get('symbolic_expression', 'planes')
                    mutation_plans[ds_layer] = {
                        'mutation_type': 'dimension',
                        'symbolic': True,
                        'symbolic_expression': final_output,
                        'new_out': None,
                        'new_in': None
                    }
        
        return mutation_plans
    
    def _select_novel_pattern(self) -> str:
        """Randomly select a novel architectural pattern to apply."""
        import random
        patterns = ["bottleneck", "inverted_bottleneck", "squeeze_excite", "asymmetric_expansion"]
        return random.choice(patterns)
    
    def _generate_bottleneck_pattern(self, conv_layers: List[str], bn_layers: List[str], base_params: Dict) -> Dict:
        """Generate bottleneck pattern: compress -> expand -> compress."""
        mutations = {}
        
        if len(conv_layers) >= 2:
            # First conv: compress
            mutations[conv_layers[0]] = {
                'mutation_type': 'dimension',
                'symbolic': True, 
                'symbolic_expression': 'planes // 2',
                'new_out': None,
                'new_in': None
            }
            
            # Last conv: expand back to original
            mutations[conv_layers[-1]] = {
                'mutation_type': 'dimension',
                'symbolic': True,
                'symbolic_expression': 'planes',
                'new_out': None,
                'new_in': None
            }
            
            # Intermediate convs: maintain compressed size
            for conv in conv_layers[1:-1]:
                mutations[conv] = {
                    'mutation_type': 'dimension', 
                    'symbolic': True,
                    'symbolic_expression': 'planes // 2',
                    'new_out': None,
                    'new_in': None
                }
        
        return mutations
    
    def _generate_inverted_bottleneck_pattern(self, conv_layers: List[str], bn_layers: List[str], base_params: Dict) -> Dict:
        """Generate inverted bottleneck: expand -> compress."""
        mutations = {}
        
        if len(conv_layers) >= 2:
            # First conv: expand
            mutations[conv_layers[0]] = {
                'mutation_type': 'dimension',
                'symbolic': True,
                'symbolic_expression': 'planes * 4',
                'new_out': None,
                'new_in': None
            }
            
            # Last conv: compress back to original
            mutations[conv_layers[-1]] = {
                'mutation_type': 'dimension', 
                'symbolic': True,
                'symbolic_expression': 'planes',
                'new_out': None,
                'new_in': None
            }
        
        return mutations
    
    def _generate_squeeze_excite_pattern(self, conv_layers: List[str], bn_layers: List[str], base_params: Dict) -> Dict:
        """Generate squeeze-excite inspired pattern with channel attention."""
        mutations = {}
        
        if len(conv_layers) >= 2:
            # Create a squeeze-like reduction in middle layers
            middle_idx = len(conv_layers) // 2
            
            mutations[conv_layers[middle_idx]] = {
                'mutation_type': 'dimension',
                'symbolic': True,
                'symbolic_expression': 'planes // 4',  # Heavy squeeze
                'new_out': None,
                'new_in': None
            }
            
            # Final layer expands back
            mutations[conv_layers[-1]] = {
                'mutation_type': 'dimension',
                'symbolic': True, 
                'symbolic_expression': 'planes',
                'new_out': None,
                'new_in': None
            }
        
        return mutations
    
    def _generate_asymmetric_expansion_pattern(self, conv_layers: List[str], bn_layers: List[str], base_params: Dict) -> Dict:
        """Generate asymmetric expansion patterns for novel architectures."""
        mutations = {}
        
        if len(conv_layers) >= 2:
            # Create complex asymmetric patterns
            import random
            
            # Random expansion factors that still resolve to original size
            factors = ['planes * 3 // 2', 'planes * 5 // 4', 'planes * 7 // 8']
            
            for i, conv in enumerate(conv_layers[:-1]):
                mutations[conv] = {
                    'mutation_type': 'dimension',
                    'symbolic': True,
                    'symbolic_expression': random.choice(factors),
                    'new_out': None,
                    'new_in': None
                }
            
            # Final layer always resolves to original
            mutations[conv_layers[-1]] = {
                'mutation_type': 'dimension',
                'symbolic': True,
                'symbolic_expression': 'planes',
                'new_out': None,
                'new_in': None
            }
        
        return mutations


# Global instance for convenience
pattern_detector = ArchitecturePatternDetector()