"""
CryptoForge — JSON serialization utility (Python 3.13 compatible)

In Python 3.13, numpy.bool_ is a subclass of numpy.integer.
A naive convert function that checks np.integer first will convert
numpy bools to Python bools via int(), which re-enters the JSON
default handler and creates a circular reference.

Fix: check bool types FIRST, before int/float types.
"""

import json
import numpy as np


def numpy_safe_convert(obj):
    """Convert numpy types to Python natives for JSON serialization.
    
    CRITICAL: bool checks must come BEFORE int checks because
    bool is a subclass of int in Python, and np.bool_ is a subclass
    of np.integer in Python 3.13+.
    """
    # Bool FIRST (before int, because bool is subclass of int)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: dict, path: str):
    """Save dict to JSON with numpy-safe serialization."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=numpy_safe_convert)