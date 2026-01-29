"""State-space model definitions."""

from pairs_ssm.models.params import ModelParams, FilterResult
from pairs_ssm.models.base_ssm import BaseSSM
from pairs_ssm.models.model_I import ModelI
from pairs_ssm.models.model_II import ModelII
from pairs_ssm.models.model_specs import (
    get_model_spec,
    get_simulation_model,  # Alias for backward compatibility
    SIMULATION_MODELS,
    SimulationSpec,
)

__all__ = [
    "ModelParams",
    "FilterResult",
    "BaseSSM",
    "ModelI",
    "ModelII",
    "get_model_spec",
    "get_simulation_model",
    "SIMULATION_MODELS",
    "SimulationSpec",
]
