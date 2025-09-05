"""
Architecture Classifier for adaptive mutation strategies.
Determines the optimal mutation approach based on neural network patterns.
"""

import torch.nn as nn
import torch.fx as fx
from typing import Dict, List, Optional, Set, Literal

ArchitectureType = Literal["block-based", "sequential", "transformer", "specialized"]


class ArchitectureClassifier:
    """
    Classifies neural network architectures into mutation strategy categories.
    """
    
    def __init__(self):
        self.block_factory_patterns = {
            '_make_layer', '_build_block', '_create_stage', '_make_block',
            'make_layer', 'build_block', 'create_stage', 'make_block'
        }
        
        self.attention_patterns = {
            'MultiheadAttention', 'SelfAttention', 'ScaledDotProductAttention',
            'attention', 'attn', 'mha'
        }
        
        self.specialized_architectures = {
            'BayesianNet', 'MoE', 'MixtureOfExperts', 'Bayesian', 'Probabilistic'
        }
    
    def classify(self, model: nn.Module, graph: Optional[fx.Graph] = None) -> ArchitectureType:
        """
        Classify a neural network architecture into mutation strategy category.
        
        Args:
            model: The neural network model to classify
            graph: Optional FX graph for deeper analysis
            
        Returns:
            ArchitectureType: The classified architecture pattern
        """
        model_name = model.__class__.__name__
        
        # Check for specialized architectures first
        if any(special_name in model_name for special_name in self.specialized_architectures):
            return "specialized"
        
        # Use FX graph analysis if available for pattern detection
        if graph is not None:
            return self._classify_with_graph(model, graph)
        
        # Fallback to name-based classification
        return self._classify_by_name(model_name)
    
    def _classify_with_graph(self, model: nn.Module, graph: fx.Graph) -> ArchitectureType:
        """
        Classify using FX graph analysis for precise pattern detection.
        """
        # Check for block factory patterns
        if self._has_block_factory_pattern(graph):
            return "block-based"
        
        # Check for attention/transformer patterns
        if self._has_attention_pattern(graph):
            return "transformer"
        
        # Default to sequential for simple layer stacks
        return "sequential"
    
    def _classify_by_name(self, model_name: str) -> ArchitectureType:
        """
        Fallback classification based on model name patterns.
        """
        lower_name = model_name.lower()
        
        # Known block-based architectures
        block_based_keywords = {'resnet', 'densenet', 'dpn', 'darknet', 'shufflenet'}
        if any(keyword in lower_name for keyword in block_based_keywords):
            return "block-based"
        
        # Known transformer architectures
        transformer_keywords = {'transformer', 'vit', 'swin', 'attention', 'bert'}
        if any(keyword in lower_name for keyword in transformer_keywords):
            return "transformer"
        
        # Default to sequential
        return "sequential"
    
    def _has_block_factory_pattern(self, graph: fx.Graph) -> bool:
        """
        Check if the graph contains block factory method patterns.
        """
        for node in graph.nodes:
            if node.op == 'call_method' and node.target in self.block_factory_patterns:
                return True
            if node.op == 'call_function' and hasattr(node.target, '__name__'):
                if node.target.__name__ in self.block_factory_patterns:
                    return True
        return False
    
    def _has_attention_pattern(self, graph: fx.Graph) -> bool:
        """
        Check if the graph contains attention mechanism patterns.
        """
        for node in graph.nodes:
            if node.op == 'call_module':
                module_name = str(node.target).lower()
                if any(pattern in module_name for pattern in self.attention_patterns):
                    return True
        return False
    
    def get_mutation_strategy(self, architecture_type: ArchitectureType) -> str:
        """
        Get the recommended mutation strategy for an architecture type.
        
        Returns:
            Strategy name: 'block-propagation', 'direct-layer', 'transformer-aware', or 'specialized'
        """
        strategy_map = {
            "block-based": "block-propagation",
            "sequential": "direct-layer", 
            "transformer": "transformer-aware",
            "specialized": "specialized"
        }
        return strategy_map[architecture_type]


# Global instance for convenience
classifier = ArchitectureClassifier()