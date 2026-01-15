"""
Random seed management for reproducibility.
"""

import numpy as np
from typing import Optional

_GLOBAL_SEED: Optional[int] = None
_GLOBAL_RNG: Optional[np.random.Generator] = None


def set_seed(seed: int) -> None:
    """
    Set global random seed for reproducibility.
    
    Parameters
    ----------
    seed : int
        Random seed value
    """
    global _GLOBAL_SEED, _GLOBAL_RNG
    _GLOBAL_SEED = seed
    _GLOBAL_RNG = np.random.default_rng(seed)
    
    # Also set numpy's legacy random state for compatibility
    np.random.seed(seed)


def get_seed() -> Optional[int]:
    """Get the current global seed."""
    return _GLOBAL_SEED


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Get a random number generator.
    
    Parameters
    ----------
    seed : int, optional
        If provided, create a new RNG with this seed.
        Otherwise, return the global RNG.
        
    Returns
    -------
    np.random.Generator
        Random number generator instance
    """
    if seed is not None:
        return np.random.default_rng(seed)
    
    global _GLOBAL_RNG
    if _GLOBAL_RNG is None:
        _GLOBAL_RNG = np.random.default_rng(42)  # Default seed
    
    return _GLOBAL_RNG


def spawn_rng(n: int, seed: Optional[int] = None) -> list:
    """
    Create multiple independent RNGs for parallel processing.
    
    Parameters
    ----------
    n : int
        Number of RNGs to create
    seed : int, optional
        Base seed
        
    Returns
    -------
    list
        List of independent np.random.Generator instances
    """
    base_rng = get_rng(seed)
    return [np.random.default_rng(base_rng.integers(0, 2**32)) for _ in range(n)]
