# src/jqm/examples/optimize_tb.py
"""
Run from anywhere after installing jqm:
    python -m jqm.examples.optimize_tb

Or import in your own script:
    from jqm.examples.optimize_tb import main
    main()
"""
import jax.numpy as jnp
from jqm import TightBinding, TBParams, optimize_geometry, LBFGSOptions


def main():
    # Small demo: H(1) - C(8) - H(1)
    Z = jnp.array([1, 6, 1], dtype=jnp.int32)
    R0 = jnp.array(
        [[0.0, 0.0, -1.1], [0.0, 0.0, 0.0], [0.0, 0.0, 1.1]], dtype=jnp.float64
    )

    params = TBParams(
        Z_ref=jnp.array([1, 6, 7, 8, 16], dtype=jnp.int32),
        eps_ref=jnp.array(
            [-0.50, -1.00, -1.10, -1.30, -1.00], dtype=jnp.float64
        ),  # demo values
        valence_ref=jnp.array([1, 4, 5, 6, 6], dtype=jnp.float64),
        t0=-2.5,
        r0=1.4,
        decay=2.2,
        A_rep=8.0,
        r_rep=1.2,
        kT=5e-3,
    )

    model = TightBinding()

    # Example: relax all atoms (set False to freeze any atom)
    mask_free = jnp.ones(Z.shape[0], dtype=bool)

    result = optimize_geometry(
        model,
        R0,
        Z,
        params,
        mask_free=mask_free,
        options=LBFGSOptions(maxiter=300, gtol=2e-4, history_size=12),
    )

    print(
        f"E_opt = {result.E:.6f} Ha | grad_max = {result.grad_max:.3e} "
        f"| steps = {result.n_iter} | converged = {result.converged}"
    )
    print("R_opt (Å):\n", result.R_opt)


if __name__ == "__main__":
    main()
