import os
import json
import hashlib
from functools import lru_cache

class PlanTracker:
    """Tracks unique mutation plans to prevent redundant plan generation.
    Uses a file-based cache to maintain persistence between executions."""
    
    def __init__(self, storage_path="mutation_plans/unique_plans.json"):
        """Initialize with storage path for persistent plan tracking.
        
        Args:
            storage_path: Path to store unique plan identifiers
        """
        self.storage_path = storage_path
        self.seen_plans = self._load_plans()
        
        # Create directory if it doesn't exist
        if not os.path.exists(os.path.dirname(self.storage_path)):
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
    
    def _load_plans(self):
        """Load previously seen plan identifiers from storage file."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return set(json.load(f))
            except (IOError, json.JSONDecodeError):
                # Return empty set if file is corrupted or unreadable
                return set()
        return set()
    
    def _save_plans(self):
        """Save current set of seen plan identifiers to storage file."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(list(self.seen_plans), f)
        except IOError:
            # If we can't save, just continue - we'll try again next time
            pass
    
    def is_unique_plan(self, plan: dict) -> bool:
        """Check if a mutation plan is unique (not seen before).
        
        Args:
            plan: The mutation plan dictionary
            
        Returns:
            bool: True if plan is unique, False if duplicate
        """
        # Create a hashable representation of the plan
        plan_hash = self._hash_plan(plan)
        return plan_hash not in self.seen_plans
    
    def register_plan(self, plan: dict):
        """Register a new unique mutation plan.
        
        Args:
            plan: The mutation plan dictionary
        """
        plan_hash = self._hash_plan(plan)
        self.seen_plans.add(plan_hash)
        self._save_plans()
    
    def _hash_plan(self, plan: dict) -> str:
        """Create a consistent hash of a mutation plan.
        This handles the nested structure of plans to create a reliable hash.
        
        Args:
            plan: The mutation plan dictionary
            
        Returns:
            str: SHA-256 hash of the plan
        """
        # Sort keys to ensure consistent ordering
        ordered_plan = json.dumps(plan, sort_keys=True)
        return hashlib.sha256(ordered_plan.encode()).hexdigest()
    
    def get_unique_count(self) -> int:
        """Get count of unique plans tracked.
        
        Returns:
            int: Number of unique plans
        """
        return len(self.seen_plans)

# Singleton instance for global access
_instance = None

def get_plan_tracker():
    """Get the global plan tracker instance.
    
    Returns:
        PlanTracker: The singleton tracker instance
    """
    global _instance
    if _instance is None:
        _instance = PlanTracker()
    return _instance

@lru_cache(maxsize=128)
def is_plan_unique(plan: dict) -> bool:
    """Cached check for plan uniqueness (for performance).
    
    Args:
        plan: The mutation plan dictionary
        
    Returns:
        bool: True if plan is unique, False if duplicate
    """
    return get_plan_tracker().is_unique_plan(plan)
