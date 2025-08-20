# src/jqm/qm_models/sk_tight_binding.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp

from .base import QMModel

# ------------------------- Parameters -------------------------


@dataclass
class SKPairParams:
    """Two-center Slater–Koster parameters for a given element pair."""

    # Amplitudes (Hartree) for SK integrals evaluated near r0
    V_ss_sigma0: float
    V_sp_sigma0: float
    V_pp_sigma0: float
    V_pp_pi0: float

    # Radial decay and reference bond length
    r0: float  # Å where amplitudes are nominal
    beta: float  # Å^-1 exponential decay: exp(-beta*(r - r0))

    # Smooth cutoff: 1 for r <= Rc-w, cosine-like switch to 0 by r >= Rc
    Rc: float  # cutoff radius (Å)
    w: float  # switch width (Å)

    # Short-range pair repulsion: A * exp(-r / r_rep)
    A_rep: float  # Hartree
    r_rep: float  # Å


@dataclass
class SKTBParams:
    """
    Global SK-TB parameters. Energies: Hartree (Ha). Distances: Å.
    Elements supported: H (Z=1), C (Z=6), N (Z=7), O (Z=8).
    """

    # Onsite energies (Hartree)
    eps_H_1s: float
    eps_C_2s: float
    eps_C_2p: float
    eps_N_2s: float
    eps_N_2p: float
    eps_O_2s: float
    eps_O_2p: float

    # Valence electron counts (spin *not* included here)
    valence_H: int = 1
    valence_C: int = 4
    valence_N: int = 5
    valence_O: int = 6

    # Pair-specific Slater–Koster sets. Keys are sorted (Z1, Z2).
    pair: Dict[Tuple[int, int], SKPairParams] = None

    # Electronic smearing (Hartree) to keep occupations smooth
    kT: float = 5e-3


# ------------------------- Model -------------------------
class SKTightBinding(QMModel):
    # ---------- existing helpers (_orbital_count_for_Z, _onsite_for_Z, ...) can stay,
    # but we'll avoid calling them with Python ints inside energy(). ----------

    @staticmethod
    def _pairwise_vectors(R: jnp.ndarray):
        dR = R[:, None, :] - R[None, :, :]
        r = jnp.linalg.norm(dR + 1e-15, axis=-1)
        L = dR / r[..., None]
        return dR, r, L

    @staticmethod
    def _cutoff(r: jnp.ndarray, Rc: float, w: float) -> jnp.ndarray:
        x = (r - (Rc - w)) / jnp.maximum(w, 1e-12)
        x = jnp.clip(x, 0.0, 1.0)
        return 1.0 - (3.0 * x**2 - 2.0 * x**3)

    @staticmethod
    def _radial(V0: float, r: jnp.ndarray, r0: float, beta: float, Rc: float, w: float):
        return V0 * jnp.exp(-beta * (r - r0)) * SKTightBinding._cutoff(r, Rc, w)

    @staticmethod
    def _pair_fields(Zi, Zj, params: SKTBParams):
        """
        JAX-safe selection of SK pair parameters for (Zi, Zj).
        Returns a tuple of scalars:
          (Vss0, Vsp0, Vpps0, Vppp0, r0, beta, Rc, w, A_rep, r_rep)
        Uses sum of masked constants (no Python dict lookup on tracers).
        """
        # List of available pairs from the Python dict (static at trace time)
        # We accumulate each field with jnp.where over all known keys.
        # Prepare zeros as JAX scalars:
        zero = jnp.array(0.0, dtype=jnp.float64)

        Vss = zero
        Vsp = zero
        Vpps = zero
        Vppp = zero
        r0 = zero
        beta = zero
        Rc = zero
        w = zero
        A = zero
        rrep = zero

        for (a, b), sk in (params.pair or {}).items():
            # match if (Zi,Zj) == (a,b) or (b,a)
            match = jnp.logical_or(
                jnp.logical_and(Zi == a, Zj == b), jnp.logical_and(Zi == b, Zj == a)
            )
            m = match.astype(jnp.float64)

            Vss = Vss + m * jnp.array(sk.V_ss_sigma0, dtype=jnp.float64)
            Vsp = Vsp + m * jnp.array(sk.V_sp_sigma0, dtype=jnp.float64)
            Vpps = Vpps + m * jnp.array(sk.V_pp_sigma0, dtype=jnp.float64)
            Vppp = Vppp + m * jnp.array(sk.V_pp_pi0, dtype=jnp.float64)
            r0 = r0 + m * jnp.array(sk.r0, dtype=jnp.float64)
            beta = beta + m * jnp.array(sk.beta, dtype=jnp.float64)
            Rc = Rc + m * jnp.array(sk.Rc, dtype=jnp.float64)
            w = w + m * jnp.array(sk.w, dtype=jnp.float64)
            A = A + m * jnp.array(sk.A_rep, dtype=jnp.float64)
            rrep = rrep + m * jnp.array(sk.r_rep, dtype=jnp.float64)

        return (Vss, Vsp, Vpps, Vppp, r0, beta, Rc, w, A, rrep)

    @staticmethod
    def _onsite_scalar(Zi: jnp.ndarray, orb_idx: int, p: SKTBParams) -> jnp.ndarray:
        """
        Return the onsite energy for atom with atomic number Zi at orbital index orb_idx.
        Orbital order:
          H: [s]                      -> idx 0
          C/N/O: [2s, 2p_x, 2p_y, 2p_z] -> idx 0..3 (0 is s, others are p)
        """
        Zi = Zi.astype(jnp.int32)
        isH = Zi == 1
        isC = Zi == 6
        isN = Zi == 7
        isO = Zi == 8

        idx0 = orb_idx == 0
        # For H: only s (idx0)
        e_H = jnp.array(p.eps_H_1s, dtype=jnp.float64)
        # For C/N/O: s if idx0, else p
        e_C = jnp.where(
            idx0,
            jnp.array(p.eps_C_2s, dtype=jnp.float64),
            jnp.array(p.eps_C_2p, dtype=jnp.float64),
        )
        e_N = jnp.where(
            idx0,
            jnp.array(p.eps_N_2s, dtype=jnp.float64),
            jnp.array(p.eps_N_2p, dtype=jnp.float64),
        )
        e_O = jnp.where(
            idx0,
            jnp.array(p.eps_O_2s, dtype=jnp.float64),
            jnp.array(p.eps_O_2p, dtype=jnp.float64),
        )
        return isH * e_H + isC * e_C + isN * e_N + isO * e_O

    @staticmethod
    def _ni_from_Z(Z: jnp.ndarray) -> jnp.ndarray:
        """Number of orbitals per atom: 1 for H, 4 for C/N/O."""
        Zi = Z.astype(jnp.int32)
        return jnp.where(Zi == 1, 1, 4).astype(jnp.int32)

    @staticmethod
    def _is_p_orb(Zi: jnp.ndarray, a: int) -> jnp.ndarray:
        """True if orbital index a is a p-orbital for element Zi."""
        Zi = Zi.astype(jnp.int32)
        return jnp.logical_and(Zi != 1, a > 0)

    @staticmethod
    def _p_component_index(a: int) -> int:
        """Map p orbital index to component index: a=1->0 (x), 2->1 (y), 3->2 (z)."""
        return a - 1  # safe to use as Python int in fixed 0..3 loops

    def energy(self, R: jnp.ndarray, Z: jnp.ndarray, params: SKTBParams) -> jnp.ndarray:
        """
        JAX-safe Hamiltonian build:
          - static loop bounds only
          - fixed-size padded basis (4 orbitals/atom), mask H p-orbitals
          - no Python int() on tracers or dict lookups keyed by tracers
        """
        N = int(Z.shape[0])  # static at trace time
        M = 4 * N  # padded basis size
        offsets = 4 * jnp.arange(N, dtype=jnp.int32)  # (N,)
        ni_vec = self._ni_from_Z(Z)  # (N,)  1 for H, 4 for C/N/O

        H = jnp.zeros((M, M), dtype=jnp.float64)
        diag = jnp.zeros((M,), dtype=jnp.float64)

        # ---- Onsite diagonal (loop a=0..3 with mask a<ni)
        def fill_diag_i(i, dvec):
            Zi = Z[i]
            oi = offsets[i]
            ni = ni_vec[i]

            def fill_a(a, dv):
                valid = a < ni
                idx = oi + a
                e_a = self._onsite_scalar(Zi, a, params)
                val = jnp.where(valid, e_a, 0.0)
                dv = dv.at[idx].set(jnp.where(valid, val, dv[idx]))
                return dv

            return lax.fori_loop(0, 4, fill_a, dvec)

        diag = lax.fori_loop(0, N, fill_diag_i, diag)
        H = H + jnp.diag(diag)

        # ---- Pairwise blocks with static loops; mask j>i
        _, rmat, Lmat = self._pairwise_vectors(R)
        E_rep = jnp.array(0.0, dtype=jnp.float64)

        def outer_i(i, carry):
            H_acc, Erep_acc = carry
            Zi = Z[i]
            ni = ni_vec[i]
            oi = offsets[i]

            def inner_j(j, carry2):
                H_loc, Erep_loc = carry2
                Zj = Z[j]
                nj = ni_vec[j]
                oj = offsets[j]
                rij = rmat[i, j]
                Lij = Lmat[i, j]  # (3,)
                pair_active = j > i  # mask, keeps bounds static

                # Select SK parameters via masked sums (no dict indexing with tracers)
                Vss, Vsp, Vpps, Vppp, r0, beta, Rc, w, A, rrep = self._pair_fields(
                    Zi, Zj, params
                )

                # Radial SK values
                V_ss = self._radial(Vss, rij, r0, beta, Rc, w)
                V_sp = self._radial(Vsp, rij, r0, beta, Rc, w)
                V_pps = self._radial(Vpps, rij, r0, beta, Rc, w)
                V_ppp = self._radial(Vppp, rij, r0, beta, Rc, w)
                V_pp_common = V_pps - V_ppp
                lvec = Lij  # (3,)

                # Fill 4x4 block with masks
                # def loop_a(a, Htmp):
                #     va = a < ni
                #     is_p_a = jnp.logical_and(Zi != 1, a > 0)
                #     comp_a = self._p_component_index(a)  # 0..2 (python int OK here)
                def loop_a(a, Htmp):
                    va = a < ni
                    is_p_a = jnp.logical_and(Zi != 1, a > 0)
                    def loop_b(b, Htmp2):
                        vb = b < nj
                        is_p_b = jnp.logical_and(Zj != 1, b > 0)

                        # Components l_a and l_b without Python indexing on tracers
                        # a==1→x, a==2→y, a==3→z; 0 if s-orbital
                        l_a = (
                            jnp.where(a == 1, lvec[0], 0.0)
                            + jnp.where(a == 2, lvec[1], 0.0)
                            + jnp.where(a == 3, lvec[2], 0.0)
                        )
                        l_b = (
                            jnp.where(b == 1, lvec[0], 0.0)
                            + jnp.where(b == 2, lvec[1], 0.0)
                            + jnp.where(b == 3, lvec[2], 0.0)
                        )

                        s_s = jnp.logical_not(is_p_a) & jnp.logical_not(is_p_b)
                        s_p = jnp.logical_not(is_p_a) & is_p_b
                        p_s = is_p_a & jnp.logical_not(is_p_b)
                        p_p = is_p_a & is_p_b

                        val_ss = V_ss
                        val_sp = l_b * V_sp
                        val_ps = -l_a * V_sp
                        val_pp = (l_a * l_b) * V_pp_common + (a == b).astype(
                            jnp.float64
                        ) * V_ppp

                        val = jnp.where(s_s, val_ss, 0.0)
                        val = val + jnp.where(s_p, val_sp, 0.0)
                        val = val + jnp.where(p_s, val_ps, 0.0)
                        val = val + jnp.where(p_p, val_pp, 0.0)

                        use = pair_active & va & vb
                        ii = oi + a
                        jj = oj + b
                        Htmp2 = Htmp2.at[ii, jj].set(jnp.where(use, val, Htmp2[ii, jj]))
                        Htmp2 = Htmp2.at[jj, ii].set(jnp.where(use, val, Htmp2[jj, ii]))
                        return Htmp2
                    return lax.fori_loop(0, 4, loop_b, Htmp)
                H_loc = lax.fori_loop(0, 4, loop_a, H_loc)

                # Repulsion (only if j>i)
                cut = self._cutoff(rij, Rc, w)
                Epair = A * jnp.exp(-rij / jnp.maximum(rrep, 1e-8)) * cut
                Erep_loc = Erep_loc + jnp.where(pair_active, Epair, 0.0)

                return (H_loc, Erep_loc)

            return lax.fori_loop(0, N, inner_j, (H_acc, Erep_acc))

        H, E_rep = lax.fori_loop(0, N, outer_i, (H, E_rep))

        # ---- Electrons & band energy (all-JAX)
        nelec = (
            (
                (Z == 1) * params.valence_H
                + (Z == 6) * params.valence_C
                + (Z == 7) * params.valence_N
                + (Z == 8) * params.valence_O
            )
            .sum()
            .astype(jnp.float64)
        )

        eps, _ = jnp.linalg.eigh(H)

        def occ(mu):
            return jax.nn.sigmoid((mu - eps) / jnp.maximum(params.kT, 1e-6))

        def N_of_mu(mu):
            return 2.0 * jnp.sum(occ(mu))

        mu_lo = eps[0] - 20.0
        mu_hi = eps[-1] + 20.0

        def bisect_body(_, state):
            lo, hi = state
            mid = 0.5 * (lo + hi)
            cond = N_of_mu(mid) > nelec
            lo2, hi2 = lax.cond(
                cond, lambda _: (lo, mid), lambda _: (mid, hi), operand=None
            )
            return (lo2, hi2)

        lo, hi = lax.fori_loop(0, 60, bisect_body, (mu_lo, mu_hi))
        mu = 0.5 * (lo + hi)
        f = occ(mu)
        E_elec = 2.0 * jnp.sum(f * eps)

        return E_elec + E_rep
