# src/jqm/tight_binding.py
from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from .base import QMModel


@dataclass
class TBParams:
    """
    Minimal orthogonal TB, 1 orbital per atom (extend later).
    Units: energies in Hartree (Ha), distances in Å.
    """

    # Per-element onsite energies and valence electron count (arrays are JAX-friendly)
    Z_ref: jnp.ndarray  # e.g., jnp.array([1, 6, 7, 8, 16], dtype=jnp.int32)
    eps_ref: jnp.ndarray  # onsite energies for Z_ref, in Ha
    valence_ref: (
        jnp.ndarray
    )  # valence electron counts for Z_ref (spin-deg total electrons = valence)

    # Hopping form: t(r) = t0 * exp(-decay * max(r - r0, 0))
    t0: float = -2.5  # Ha
    r0: float = 1.4  # Å
    decay: float = 2.2  # 1/Å

    # Simple short-range repulsion: A * exp(-r / r_rep)
    A_rep: float = 8.0  # Ha
    r_rep: float = 1.2  # Å

    # Electronic smearing for occupations (kT in Ha) to smooth gradients
    kT: float = 5e-3


class TightBinding(QMModel):
    """Orthogonal 1-orbital-per-atom TB Hamiltonian with repulsion."""

    # ---------- helpers ----------
    @staticmethod
    def _lookup_by_Z(
        Z: jnp.ndarray, Z_ref: jnp.ndarray, vals: jnp.ndarray
    ) -> jnp.ndarray:
        """Map integer atomic numbers Z to values using Z_ref/vals arrays, JAX-friendly."""

        # For each z_i, sum vals * 1[Z_ref == z_i]
        def one(z):
            return jnp.sum(jnp.where(Z_ref == z, vals, 0.0))

        return jax.vmap(one)(Z)

    @staticmethod
    def _pairwise(R: jnp.ndarray):
        dR = R[:, None, :] - R[None, :, :]
        r = jnp.linalg.norm(dR + 1e-15, axis=-1)  # avoid NaN in grad at r=0
        return dR, r

    @staticmethod
    def _hopping(r: jnp.ndarray, t0: float, r0: float, decay: float) -> jnp.ndarray:
        t = t0 * jnp.exp(-decay * jnp.maximum(r - r0, 0.0))
        return t * (1.0 - jnp.eye(r.shape[0]))  # zero diagonal

    @staticmethod
    def _repulsion(r: jnp.ndarray, A: float, r_rep: float) -> jnp.ndarray:
        rep = A * jnp.exp(-r / r_rep) * (1.0 - jnp.eye(r.shape[0]))
        return 0.5 * jnp.sum(rep)

    @staticmethod
    def _electronic_energy(H: jnp.ndarray, nelec: float, kT: float) -> jnp.ndarray:
        # Diagonalize (symmetric) Hamiltonian
        eps, _ = jnp.linalg.eigh(H)  # (M,)

        # Fermi-Dirac occupations with bisection for μ
        def occ(mu):
            return 1.0 / (1.0 + jnp.exp((eps - mu) / kT))

        def nelec_at(mu):
            # spin degeneracy 2
            return 2.0 * jnp.sum(occ(mu))

        mu_lo = eps[0] - 20.0
        mu_hi = eps[-1] + 20.0

        def body_fun(_, state):
            lo, hi = state
            mid = 0.5 * (lo + hi)
            # if electron count at mid is too high, lower mu_hi
            return (lo, mid) if nelec_at(mid) > nelec else (mid, hi)

        lo, hi = jax.lax.fori_loop(0, 60, body_fun, (mu_lo, mu_hi))
        mu = 0.5 * (lo + hi)
        f = occ(mu)
        return 2.0 * jnp.sum(f * eps)  # spin factor

    def energy(self, R: jnp.ndarray, Z: jnp.ndarray, params: TBParams) -> jnp.ndarray:
        _, r = self._pairwise(R)

        # onsite energies and valence electrons
        eps_on = self._lookup_by_Z(Z, params.Z_ref, params.eps_ref)  # (N,)
        val_e = self._lookup_by_Z(Z, params.Z_ref, params.valence_ref)  # (N,)
        nelec = jnp.sum(val_e)

        # Hamiltonian and repulsion
        H = jnp.diag(eps_on) + self._hopping(r, params.t0, params.r0, params.decay)
        E_elec = self._electronic_energy(H, nelec, params.kT)
        E_rep = self._repulsion(r, params.A_rep, params.r_rep)
        return E_elec + E_rep
