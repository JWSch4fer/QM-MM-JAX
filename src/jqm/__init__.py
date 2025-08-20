### src/project/__init__.py
"""Project package initializer."""

__version__ = "0.1.0"

from .qm_models.base import QMModel, LBFGSOptions
from .qm_models.tight_binding import TightBinding, TBParams
from .qm_models.sk_tight_binding import SKTightBinding, SKTBParams, SKPairParams
from .optimizer import optimize_geometry, OptResult

__all__ = [
    "QMModel",
    "LBFGSOptions",
    "OptResult",
    "optimize_geometry",
    "TightBinding",
    "TBParams",
    "SKTightBinding",
    "SKTBParams",
    "SKPairParams",
]
