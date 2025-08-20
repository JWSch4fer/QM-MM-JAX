# qm_models/base.py
from abc import ABC, abstractmethod
import jax.numpy as jnp


class QMModel(ABC):
    def __init__(self, nuclear_coords: jnp.ndarray, nuclear_charges: jnp.ndarray):
        self.coords = nuclear_coords  # shape (N, 3)
        self.charges = nuclear_charges  # shape (N,)

    @abstractmethod
    def energy(self) -> float:
        pass

    @abstractmethod
    def forces(self) -> jnp.ndarray:
        pass
