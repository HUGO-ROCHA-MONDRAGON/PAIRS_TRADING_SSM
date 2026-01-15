"""
Simulation model specifications from Zhang (2021) Table 1.

These are used for Monte Carlo optimization of thresholds,
not for fitting to real data.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Callable, Optional


@dataclass
class SimulationSpec:
    """Specification for a simulation model."""
    
    name: str
    description: str
    theta0: float = 0.0
    theta1: float = 0.959
    theta2: float = 0.0           # Quadratic term
    sigma: float = 0.0049         # Base noise std
    q_het: float = 0.0            # Heteroscedasticity
    mix_prob: float = 0.0         # Mixture probability
    mix_scale: float = 3.0        # Outlier scale
    
    @property
    def is_nonlinear(self) -> bool:
        return abs(self.theta2) > 1e-10
    
    @property
    def is_heteroscedastic(self) -> bool:
        return abs(self.q_het) > 1e-10
    
    @property
    def is_non_gaussian(self) -> bool:
        return self.mix_prob > 1e-10


# =============================================================================
# SIMULATION MODEL REGISTRY
# =============================================================================

SIMULATION_MODELS: Dict[str, SimulationSpec] = {
    
    "model_1": SimulationSpec(
        name="Model 1",
        description="Simple AR(1): x_{t+1} = 0.9590 x_t + 0.0049 η_t",
        theta0=0.0,
        theta1=0.9590,
        sigma=0.0049,
    ),
    
    "model_2": SimulationSpec(
        name="Model 2",
        description="Nonlinear AR(1): x_{t+1} = 0.9 x_t + 0.259 x_t² + 0.0049 η_t",
        theta0=0.0,
        theta1=0.9,
        theta2=0.259,
        sigma=0.0049,
    ),
    
    "model_3": SimulationSpec(
        name="Model 3",
        description="Heteroscedastic: x_{t+1} = 0.959 x_t + sqrt(0.00089 + 0.08 x_t²) η_t",
        theta0=0.0,
        theta1=0.9590,
        sigma=np.sqrt(0.00089),  # Base volatility
        q_het=0.08,
    ),
    
    "model_4": SimulationSpec(
        name="Model 4",
        description="Non-Gaussian: AR(1) with mixture of normals",
        theta0=0.0,
        theta1=0.9590,
        sigma=0.0049,
        mix_prob=0.05,
        mix_scale=3.0,
    ),
    
    "model_5": SimulationSpec(
        name="Model 5",
        description="Combined: nonlinear + heteroscedastic + non-Gaussian",
        theta0=0.0,
        theta1=0.9,
        theta2=0.259,
        sigma=np.sqrt(0.00089),
        q_het=0.08,
        mix_prob=0.05,
        mix_scale=3.0,
    ),
}


def get_simulation_model(name: str) -> SimulationSpec:
    """
    Get simulation model specification by name.
    
    Parameters
    ----------
    name : str
        Model name (e.g., "model_1", "model_2", etc.)
        
    Returns
    -------
    SimulationSpec
        Model specification
    """
    name = name.lower().replace(" ", "_")
    
    if name not in SIMULATION_MODELS:
        available = list(SIMULATION_MODELS.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    
    return SIMULATION_MODELS[name]


def simulate_from_spec(
    spec: SimulationSpec,
    n_steps: int,
    n_paths: int,
    x0: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate paths from a model specification.
    
    Parameters
    ----------
    spec : SimulationSpec
        Model specification
    n_steps : int
        Number of time steps
    n_paths : int
        Number of paths
    x0 : float
        Initial state
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Simulated paths, shape (n_paths, n_steps)
    """
    rng = np.random.default_rng(seed)
    
    x = np.zeros((n_paths, n_steps))
    x[:, 0] = x0
    
    for t in range(n_steps - 1):
        # Mean reversion
        x_next = spec.theta0 + spec.theta1 * x[:, t]
        
        # Nonlinearity
        if spec.is_nonlinear:
            x_next += spec.theta2 * x[:, t]**2
        
        # Noise
        if spec.is_non_gaussian:
            # Mixture of normals
            is_outlier = rng.random(n_paths) < spec.mix_prob
            scale = np.where(is_outlier, spec.sigma * spec.mix_scale, spec.sigma)
            noise = scale * rng.standard_normal(n_paths)
        else:
            if spec.is_heteroscedastic:
                # State-dependent volatility
                vol = np.sqrt(spec.sigma**2 + spec.q_het * x[:, t]**2)
            else:
                vol = spec.sigma
            noise = vol * rng.standard_normal(n_paths)
        
        x[:, t + 1] = x_next + noise
    
    return x


def simulate_model(
    model_name: str,
    n_steps: int,
    n_paths: int,
    x0: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function to simulate from named model.
    
    Parameters
    ----------
    model_name : str
        Model name
    n_steps : int
        Number of time steps
    n_paths : int
        Number of paths
    x0 : float
        Initial state
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Simulated paths
    """
    spec = get_simulation_model(model_name)
    return simulate_from_spec(spec, n_steps, n_paths, x0, seed)
