# src/jqm/base.py
from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp

# Use double precision for stable geometry optimization
jax.config.update("jax_enable_x64", True)


class QMModel:
    """Abstract base class for quantum models."""

    def energy(self, R: jnp.ndarray, Z: jnp.ndarray, params) -> jnp.ndarray:
        """Return total energy (scalar). R: (N,3) in Å, Z: (N,) int."""
        raise NotImplementedError

    def forces(self, R: jnp.ndarray, Z: jnp.ndarray, params) -> jnp.ndarray:
        """Return forces as -∂E/∂R with shape (N,3)."""
        e_of_R = lambda _R: self.energy(_R, Z, params)
        return -jax.grad(e_of_R)(R)


@dataclass
class LBFGSOptions:
    maxiter: int = 500
    gtol: float = 1e-4  # max |force| (Ha/Å) stopping criterion
    history_size: int = 10
    # backtracking line search is fine for most TB systems
    linesearch: str = "backtracking"  # ("backtracking" | "zoom")
    has_aux: bool = False
