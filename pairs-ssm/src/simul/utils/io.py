"""
I/O utilities for loading configs and saving/loading results.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to YAML config file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    config_path : str or Path
        Output path
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_results(obj: Any, path: Union[str, Path]) -> None:
    """
    Save object to pickle file.
    
    Parameters
    ----------
    obj : Any
        Object to save
    path : str or Path
        Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_results(path: Union[str, Path]) -> Any:
    """
    Load object from pickle file.
    
    Parameters
    ----------
    path : str or Path
        Path to pickle file
        
    Returns
    -------
    Any
        Loaded object
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating if necessary.
    
    Parameters
    ----------
    path : str or Path
        Directory path
        
    Returns
    -------
    Path
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
