# src/jqm/examples/optimize_tb_sk.py
"""
Multi-molecule Slater–Koster TB optimizations:
- H2
- H2O
- CH4
- NH3

Run:
    python -m jqm.examples.optimize_tb_sk
"""
import jax.numpy as jnp
from jqm import SKTBParams, SKPairParams, LBFGSOptions


def _ev(x):  # eV -> Hartree
    return x * 0.03674932217565499


def default_params() -> SKTBParams:
    # DEMO parameters (Hartree). Calibrate for production use.
    pair = {}

    # H-H
    pair[(1, 1)] = SKPairParams(
        V_ss_sigma0=_ev(-6.0),
        V_sp_sigma0=0.0,
        V_pp_sigma0=0.0,
        V_pp_pi0=0.0,
        r0=0.75,
        beta=2.5,
        Rc=3.0,
        w=0.5,
        A_rep=6.0,
        r_rep=0.6,
    )

    # C-H
    pair[(1, 6)] = SKPairParams(
        V_ss_sigma0=_ev(-7.0),
        V_sp_sigma0=_ev(6.0),
        V_pp_sigma0=_ev(3.0),
        V_pp_pi0=_ev(-1.0),
        r0=1.10,
        beta=2.3,
        Rc=3.0,
        w=0.6,
        A_rep=10.0,
        r_rep=0.80,
    )

    # N-H
    pair[(1, 7)] = SKPairParams(
        V_ss_sigma0=_ev(-7.5),
        V_sp_sigma0=_ev(7.0),
        V_pp_sigma0=_ev(3.0),
        V_pp_pi0=_ev(-1.2),
        r0=1.02,
        beta=2.4,
        Rc=3.0,
        w=0.6,
        A_rep=9.0,
        r_rep=0.75,
    )

    # O-H
    pair[(1, 8)] = SKPairParams(
        V_ss_sigma0=_ev(-8.0),
        V_sp_sigma0=_ev(7.2),
        V_pp_sigma0=_ev(3.2),
        V_pp_pi0=_ev(-1.3),
        r0=0.98,
        beta=2.6,
        Rc=3.0,
        w=0.6,
        A_rep=10.0,
        r_rep=0.70,
    )

    # Optional homonuclear p-block pairs if needed later (not used by these 4 molecules)
    pair[(6, 6)] = SKPairParams(
        V_ss_sigma0=_ev(-8.0),
        V_sp_sigma0=_ev(7.5),
        V_pp_sigma0=_ev(5.0),
        V_pp_pi0=_ev(-1.5),
        r0=1.45,
        beta=2.2,
        Rc=3.5,
        w=0.7,
        A_rep=30.0,
        r_rep=1.30,
    )
    pair[(7, 7)] = SKPairParams(
        V_ss_sigma0=_ev(-8.5),
        V_sp_sigma0=_ev(7.8),
        V_pp_sigma0=_ev(5.2),
        V_pp_pi0=_ev(-1.6),
        r0=1.40,
        beta=2.2,
        Rc=3.5,
        w=0.7,
        A_rep=32.0,
        r_rep=1.25,
    )
    pair[(8, 8)] = SKPairParams(
        V_ss_sigma0=_ev(-9.0),
        V_sp_sigma0=_ev(8.0),
        V_pp_sigma0=_ev(5.5),
        V_pp_pi0=_ev(-1.7),
        r0=1.35,
        beta=2.3,
        Rc=3.5,
        w=0.7,
        A_rep=35.0,
        r_rep=1.20,
    )

    return SKTBParams(
        # Onsites (Ha) — rough demo values
        eps_H_1s=-0.55,
        eps_C_2s=-1.15,
        eps_C_2p=-0.60,
        eps_N_2s=-1.30,
        eps_N_2p=-0.65,
        eps_O_2s=-1.45,
        eps_O_2p=-0.70,
        pair=pair,
        kT=2e-3,
    )


def run_case(
    name, Z, R0, params, opt=LBFGSOptions(maxiter=400, gtol=2e-4, history_size=12)
):
    from jqm import SKTightBinding, optimize_geometry

    model = SKTightBinding()
    res = optimize_geometry(
        model, R0, Z, params, mask_free=jnp.ones(Z.shape[0], bool), options=opt
    )
    print(
        f"[{name}] E_opt = {res.E:.6f} Ha | grad_max = {res.grad_max:.3e} "
        f"| steps = {res.n_iter} | converged = {res.converged}"
    )
    print(f"[{name}] R_opt (Å):\n{res.R_opt}\n")


def main():
    p = default_params()

    # H2 (≈0.74 Å bond in reality; start near 0.7 Å)
    Z_h2 = jnp.array([1, 1], dtype=jnp.int32)
    R_h2 = jnp.array([[0.0, 0.0, -0.70], [0.0, 0.0, 0.70]], dtype=jnp.float64)

    # H2O: start near 1.0 Å, ~104.5°; place O at origin, two H in xz-plane
    Z_h2o = jnp.array([8, 1, 1], dtype=jnp.int32)
    OH = 1.00
    theta = jnp.deg2rad(104.5 / 2.0)
    hx = OH * jnp.sin(theta)
    hz = OH * jnp.cos(theta)
    R_h2o = jnp.array(
        [[0.0, 0.0, 0.0], [hx, 0.0, hz], [-hx, 0.0, hz]], dtype=jnp.float64
    )

    # CH4 (tetrahedral): C at origin; 1.09 Å along tetrahedral directions
    Z_ch4 = jnp.array([6, 1, 1, 1, 1], dtype=jnp.int32)
    rCH = 1.09
    a = rCH / jnp.sqrt(3.0)
    R_ch4 = jnp.array(
        [[0.0, 0.0, 0.0], [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]],
        dtype=jnp.float64,
    )

    # NH3 (trigonal pyramidal): N at origin; ~1.01 Å, ~107°
    Z_nh3 = jnp.array([7, 1, 1, 1], dtype=jnp.int32)
    rNH = 1.02
    # Put 3 H in an equilateral triangle slightly below N (z<0)
    R_nh3 = jnp.array(
        [
            [0.00, 0.00, 0.10],  # small N offset +z for a pyramid
            [rNH, 0.0, -0.20],
            [-0.5 * rNH, 0.8660254 * rNH, -0.20],
            [-0.5 * rNH, -0.8660254 * rNH, -0.20],
        ],
        dtype=jnp.float64,
    )

    run_case("H2", Z_h2, R_h2, p)
    run_case("H2O", Z_h2o, R_h2o, p)
    run_case("CH4", Z_ch4, R_ch4, p)
    run_case("NH3", Z_nh3, R_nh3, p)


if __name__ == "__main__":
    main()
