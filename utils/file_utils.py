"""
File system utilities for saving mutation plans and handling file operations.
"""

import os
import json
import time
from typing import Dict, Any

from mutator import config


def save_plan_to_file(model_name: str, status: str, plan: dict, details: dict) -> None:
    """
    Save a mutation plan to a JSON file.
    
    Args:
        model_name: Name of the model being mutated
        status: Status of the mutation (e.g., 'success', 'failed')
        plan: The mutation plan dictionary
        details: Additional details about the mutation
    """
    output_dir = os.path.join(config.PLANS_OUTPUT_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.time_ns()
    filename = f"{status}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    report = {
        "model_name": model_name,
        "status": status,
        "timestamp_ns": timestamp,
        "plan": plan,
        "details": details
    }
    
    # Add debug output to confirm plan saving
    if config.DEBUG_MODE:
        print(f"[Utils] Saving mutation plan to: {filepath}")
        print(f"[Utils] Plan content: {json.dumps(report, indent=2)}")

    class CustomEncoder(json.JSONEncoder):
        """Custom JSON encoder that handles Exception objects."""
        def default(self, obj):
            if isinstance(obj, Exception):
                return str(obj)
            return json.JSONEncoder.default(self, obj)

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=4, cls=CustomEncoder)
        if config.DEBUG_MODE:
            print(f"[Utils] Successfully saved mutation plan to: {filepath}")


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory to create
    """
    os.makedirs(directory_path, exist_ok=True)


def get_file_size(filepath: str) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        filepath: Path to the file
        
    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    try:
        return os.path.getsize(filepath)
    except (OSError, FileNotFoundError):
        return 0


def read_source_file(filepath: str) -> str:
    """
    Read source code from a file.
    
    Args:
        filepath: Path to the source file
        
    Returns:
        Contents of the file as a string, or empty string if failed
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except (IOError, OSError, UnicodeDecodeError) as e:
        if config.DEBUG_MODE:
            print(f"[FileUtils] Failed to read source file {filepath}: {e}")
        return ""
