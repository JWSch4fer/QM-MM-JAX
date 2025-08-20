# qm_models/tight_binding.py
from .base import QMModel
import jax.numpy as jnp
from jax import grad


class TightBinding(QMModel):
    def __init__(self, coords, charges):
        super().__init__(coords, charges)

    def hopping_matrix(self):
        dists = jnp.linalg.norm(
            self.coords[:, None, :] - self.coords[None, :, :], axis=-1
        )
        return -1.0 * jnp.exp(-dists)  # toy exponential decay

    def energy(self):
        H = self.hopping_matrix()
        return jnp.sum(H)  # toy model: sum of off-diagonal interactions

    def forces(self):
        energy_fn = lambda coords: TightBinding(coords, self.charges).energy()
        return -grad(energy_fn)(self.coords)
