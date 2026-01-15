"""
Kalman filtering implementations.

Provides:
- kalman_filter: Standard KF for linear Gaussian (Model I)
- extended_kalman_filter: EKF for nonlinear/heteroscedastic (Model II)
- unscented_kalman_filter: UKF alternative to EKF
- qmc_kalman_filter: QMCKF for non-Gaussian (Models 4-5)
- particle_filter: Bootstrap SMC for general cases
"""

from pairs_ssm.filtering.kalman_linear import kalman_filter, kalman_smoother
from pairs_ssm.filtering.kalman_extended import extended_kalman_filter, unscented_kalman_filter
from pairs_ssm.filtering.qmckf import (
    qmc_kalman_filter,
    particle_filter,
    select_filter,
    state_transition,
    state_noise_std,
)
from pairs_ssm.filtering.mle import fit_model, compare_models

__all__ = [
    # Linear Kalman
    "kalman_filter",
    "kalman_smoother",
    # Extended/Unscented
    "extended_kalman_filter",
    "unscented_kalman_filter",
    # QMC / Particle
    "qmc_kalman_filter",
    "particle_filter",
    "select_filter",
    # State functions
    "state_transition",
    "state_noise_std",
    # MLE
    "fit_model",
    "compare_models",
]
