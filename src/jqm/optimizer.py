# src/jqm/optimize.py
from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp
from jax import value_and_grad
from jaxopt import LBFGS

from .qm_models.base import LBFGSOptions, QMModel


def _pack(R: jnp.ndarray, mask_free: jnp.ndarray) -> jnp.ndarray:
    """Flatten free coordinates only."""
    return R[mask_free].reshape(-1)


def _unpack(x: jnp.ndarray, Rref: jnp.ndarray, mask_free: jnp.ndarray) -> jnp.ndarray:
    """Scatter flat vector back to full (N,3) coords using mask."""
    return Rref.at[mask_free].set(x.reshape((-1, 3)))


@dataclass
class OptResult:
    R_opt: jnp.ndarray
    E: float
    grad_max: float
    n_iter: int
    converged: bool


def optimize_geometry(
    model: QMModel,
    R0: jnp.ndarray,
    Z: jnp.ndarray,
    params,
    mask_free: jnp.ndarray | None = None,
    options: LBFGSOptions = LBFGSOptions(),
) -> OptResult:
    """
    Geometry optimization with JAXopt L-BFGS on free atoms (mask_free=True).
    """
    if mask_free is None:
        mask_free = jnp.ones(R0.shape[0], dtype=bool)

    x0 = _pack(R0, mask_free)

    # Define energy as a function of x (flattened coords of free atoms)
    def energy_x(x):
        R = _unpack(x, R0, mask_free)
        return model.energy(R, Z, params)

    # L-BFGS solver (JIT by default)
    solver = LBFGS(
        fun=energy_x,
        value_and_grad=True,
        tol=options.gtol,  # gradient infinity-norm tol
        maxiter=options.maxiter,
        history_size=options.history_size,
        linesearch=options.linesearch,
    )

    res = solver.run(x0)
    xopt = res.params
    state = res.state

    R_opt = _unpack(xopt, R0, mask_free)
    E_opt, g_opt = value_and_grad(energy_x)(xopt)
    grad_max = float(jnp.max(jnp.abs(g_opt)))
    return OptResult(
        R_opt=R_opt,
        E=float(E_opt),
        grad_max=grad_max,
        n_iter=int(state.iter_num),
        converged=bool(state.error <= options.gtol),
    )
