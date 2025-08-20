# water_geometry.py
import jax.numpy as jnp


def load_water_geometry():
    # H2O in angstroms (bent ~104.5 deg)
    coords = jnp.array(
        [
            [0.0000, 0.0000, 0.0000],  # O
            [0.7586, 0.0000, 0.5043],  # H
            [-0.7586, 0.0000, 0.5043],  # H
        ]
    )
    charges = jnp.array([8, 1, 1])
    return coords, charges
