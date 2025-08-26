import os
import json
from functools import lru_cache

class MutationTracker:
    """Tracks unique mutations across runs to prevent duplicates.
    Uses a file-based set to maintain persistence between executions."""
    
    def __init__(self, storage_path="mutation_plans/unique_mutations.json"):
        """Initialize with storage path for persistent mutation tracking.
        
        Args:
            storage_path: Path to store unique mutation checksums
        """
        self.storage_path = storage_path
        self.seen_mutations = self._load_mutations()
        
        # Create directory if it doesn't exist
        if not os.path.exists(os.path.dirname(self.storage_path)):
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
    
    def _load_mutations(self):
        """Load previously seen mutations from storage file."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return set(json.load(f))
            except (IOError, json.JSONDecodeError):
                # Return empty set if file is corrupted or unreadable
                return set()
        return set()
    
    def _save_mutations(self):
        """Save current set of seen mutations to storage file."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(list(self.seen_mutations), f)
        except IOError:
            # If we can't save, just continue - we'll try again next time
            pass
    
    def is_unique_mutation(self, checksum: str) -> bool:
        """Check if a mutation is unique (not seen before).
        
        Args:
            checksum: SHA-256 hash of the mutation's code
            
        Returns:
            bool: True if mutation is unique, False if duplicate
        """
        return checksum not in self.seen_mutations
    
    def register_mutation(self, checksum: str):
        """Register a new unique mutation.
        
        Args:
            checksum: SHA-256 hash of the mutation's code
        """
        self.seen_mutations.add(checksum)
        self._save_mutations()
    
    def get_unique_count(self) -> int:
        """Get count of unique mutations tracked.
        
        Returns:
            int: Number of unique mutations
        """
        return len(self.seen_mutations)

# Singleton instance for global access
_instance = None

def get_mutation_tracker():
    """Get the global mutation tracker instance.
    
    Returns:
        MutationTracker: The singleton tracker instance
    """
    global _instance
    if _instance is None:
        _instance = MutationTracker()
    return _instance

@lru_cache(maxsize=128)
def is_mutation_unique(checksum: str) -> bool:
    """Cached check for mutation uniqueness (for performance).
    
    Args:
        checksum: SHA-256 hash of the mutation's code
        
    Returns:
        bool: True if mutation is unique, False if duplicate
    """
    return get_mutation_tracker().is_unique_mutation(checksum)
