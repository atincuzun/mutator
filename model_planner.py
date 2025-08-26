"""
Compatibility facade for legacy imports.

This module preserves the old `mutator.model_planner` import path while delegating
to the new modular implementation in `mutator.planning`.

Example:
    from mutator.model_planner import ModelPlanner  # still works

Preferred modern usage:
    from mutator.planning import ModelPlanner
"""

from .planning import ModelPlanner

__all__ = ["ModelPlanner"]
