"""Mode 3 V3: trigger-induced representation-region experiments."""

from .metrics import Mode3Metrics, evaluate_mode3
from .support import BenignSupportModel, fit_spherical_kmeans

__all__ = ["BenignSupportModel", "Mode3Metrics", "evaluate_mode3", "fit_spherical_kmeans"]
