"""
Utilities for advanced torch.fx graph manipulation, including custom tracers.
"""
import torch.nn as nn
import torch.fx as fx

class LeafTracer(fx.Tracer):
    """
    A custom FX tracer that does NOT treat nn.Sequential as a leaf module.
    This allows the tracer to step into nn.Sequential blocks and map their
    internal layers, producing a fully "flattened" graph.
    """
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # If the module is a Sequential block, we must trace inside it.
        if isinstance(m, nn.Sequential):
            return False
        # For all other modules, use the default FX behavior.
        return super().is_leaf_module(m, module_qualified_name)


def get_detailed_graph(model: nn.Module) -> fx.Graph:
    """
    Traces a PyTorch model using the LeafTracer to produce a detailed
    graph that includes layers inside nn.Sequential containers.

    Args:
        model: The nn.Module instance to trace.

    Returns:
        A detailed torch.fx.Graph object.
    """
    tracer = LeafTracer()
    # The trace method returns the graph object
    graph = tracer.trace(model)
    # Re-constitute a GraphModule to ensure graph integrity, then return the graph
    traced_model = fx.GraphModule(tracer.root, graph)
    return traced_model.graph