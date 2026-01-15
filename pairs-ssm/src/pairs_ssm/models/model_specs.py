"""
Model specifications from Zhang (2021) Table 1.

This module provides REFERENCE DOCUMENTATION for the model parameters.
For actual simulation, use `pairs_ssm.optimization.table1.simulate_paths()`
which uses numba-optimized code for performance.

Paper's Model Specifications:
- Model 1: x_{t+1} = 0.9590 x_t + 0.0049 η_t, η_t ~ N(0,1)
- Model 2: x_{t+1} = 0.9 x_t + 0.2590 x_t² + 0.0049 η_t, η_t ~ N(0,1)
- Model 3: x_{t+1} = 0.9590 x_t + sqrt(0.00089 + 0.08 x_t²) η_t, η_t ~ N(0,1)
- Model 4: x_{t+1} = 0.9590 x_t + (0.0049/√3) η_t, η_t ~ t(3)
- Model 5: x_{t+1} = 0.9 x_t + 0.2590 x_t² + (0.0049/√3) η_t, η_t ~ t(3)

Usage
-----
>>> from pairs_ssm.models import get_model_spec
>>> spec = get_model_spec("model_3")
>>> print(spec.is_heteroscedastic)
True
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Literal


@dataclass
class SimulationSpec:
    """Specification for a simulation model."""
    
    name: str
    description: str
    theta0: float = 0.0
    theta1: float = 0.959
    theta2: float = 0.0           # Quadratic term (nonlinearity)
    sigma: float = 0.0049         # Base noise std
    q_base: float = 0.0           # Base variance for heteroscedastic (0.00089 for Model 3)
    q_het: float = 0.0            # Heteroscedasticity coefficient (0.08 for Model 3)
    noise_type: Literal["gaussian", "t"] = "gaussian"  # Noise distribution
    t_df: int = 3                 # Degrees of freedom for t-distribution
    
    @property
    def is_nonlinear(self) -> bool:
        return abs(self.theta2) > 1e-10
    
    @property
    def is_heteroscedastic(self) -> bool:
        return abs(self.q_het) > 1e-10
    
    @property
    def is_t_distributed(self) -> bool:
        return self.noise_type == "t"


# =============================================================================
# SIMULATION MODEL REGISTRY (matching Zhang 2021 exactly)
# =============================================================================

SIMULATION_MODELS: Dict[str, SimulationSpec] = {
    
    "model_1": SimulationSpec(
        name="Model 1",
        description="Linear + Gaussian + Homoscedastic: x_{t+1} = 0.9590 x_t + 0.0049 η_t",
        theta0=0.0,
        theta1=0.9590,
        sigma=0.0049,
        noise_type="gaussian",
    ),
    
    "model_2": SimulationSpec(
        name="Model 2",
        description="Nonlinear + Gaussian: x_{t+1} = 0.9 x_t + 0.2590 x_t² + 0.0049 η_t",
        theta0=0.0,
        theta1=0.9,
        theta2=0.2590,
        sigma=0.0049,
        noise_type="gaussian",
    ),
    
    "model_3": SimulationSpec(
        name="Model 3",
        description="Linear + Heteroscedastic: x_{t+1} = 0.9590 x_t + sqrt(0.00089 + 0.08 x_t²) η_t",
        theta0=0.0,
        theta1=0.9590,
        q_base=0.00089,  # Base variance term
        q_het=0.08,      # Heteroscedastic coefficient
        noise_type="gaussian",
    ),
    
    "model_4": SimulationSpec(
        name="Model 4",
        description="Linear + t-distributed: x_{t+1} = 0.9590 x_t + (0.0049/√3) η_t, η_t ~ t(3)",
        theta0=0.0,
        theta1=0.9590,
        sigma=0.0049 / np.sqrt(3),  # Scaled for t-distribution
        noise_type="t",
        t_df=3,
    ),
    
    "model_5": SimulationSpec(
        name="Model 5",
        description="Nonlinear + t-distributed: x_{t+1} = 0.9 x_t + 0.2590 x_t² + (0.0049/√3) η_t, η_t ~ t(3)",
        theta0=0.0,
        theta1=0.9,
        theta2=0.2590,
        sigma=0.0049 / np.sqrt(3),  # Scaled for t-distribution
        noise_type="t",
        t_df=3,
    ),
}


def get_model_spec(name: str) -> SimulationSpec:
    """
    Get simulation model specification by name.
    
    This is a reference/documentation module. For actual simulation,
    use `pairs_ssm.optimization.table1.simulate_paths()` which uses
    numba-optimized code for performance.
    
    Parameters
    ----------
    name : str
        Model name (e.g., "model_1", "model1", "Model 1", etc.)
        
    Returns
    -------
    SimulationSpec
        Model specification with all parameters
        
    Example
    -------
    >>> spec = get_model_spec("model_3")
    >>> print(spec.description)
    'Linear + Heteroscedastic: x_{t+1} = 0.9590 x_t + sqrt(0.00089 + 0.08 x_t²) η_t'
    >>> print(spec.is_heteroscedastic)
    True
    """
    # Normalize name: "Model 1", "model1", "model_1" all -> "model_1"
    name = name.lower().replace(" ", "_")
    if not name.startswith("model_"):
        name = name.replace("model", "model_")
    
    if name not in SIMULATION_MODELS:
        available = list(SIMULATION_MODELS.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    
    return SIMULATION_MODELS[name]


# Alias for backward compatibility
get_simulation_model = get_model_spec
