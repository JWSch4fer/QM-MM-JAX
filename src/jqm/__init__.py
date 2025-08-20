### src/project/__init__.py
"""Project package initializer."""

__version__ = "0.1.0"

from .qm_models.base import QMModel, LBFGSOptions
from .qm_models.tight_binding import TightBinding, TBParams
from .optimizer import optimize_geometry, OptResult

__all__ = [
    "QMModel",
    "LBFGSOptions",
    "OptResult",
    "optimize_geometry",
    "TightBinding",
    "TBParams",
]
