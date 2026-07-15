"""
FQ-vs-N scaling check at fixed Omega/J ratios.

Motivation (see conversation):
  - Undriven (Omega=0): H = J Z0 (x) Jz. Bath QFI ~ 4 Var(Jz) t^2 ~ N t^2  -> SQL floor.
  - Driven: adiabatic elimination gives H_eff ~ (J^2/Omega) Jz^2 (2-body generator),
    so FQ ~ (J/Omega)^2 N^2 t^2. Constant 1/Omega^2 penalty vs growing N^2 gain.
  - Predicted crossover: driven beats undriven for N >~ (Omega/J)^2.
    So sweep N at fixed, MODERATE Omega/J (2..8) instead of Omega at fixed small N.

Changes vs the old sweep script:
  - QFI is fully vectorized (no O(dim^2) Python loops).
  - No differential_evolution / CFI observable optimization -- QFI only.
  - Symmetric (collective spin) subspace: dim_bath = N+1, so N ~ 100 is cheap.
  - Coarse time grid; FQ is maximized over t per (Omega/J, N) point.
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    J: float = 1.0
    # Omega/J ratios to test. 0.0 = undriven SQL baseline.
    Omega_over_J_values: tuple = (0.0, 2.0, 3.0, 5.0, 8.0)
    # Bath sizes. dim_bath = N+1 in the symmetric subspace -- all cheap.
    N_values: tuple = (4, 8, 12, 20, 30, 50, 80)
    # Noise rates (central spin). gamma -> sz (dephasing), beta -> sx (bit flip).
    gamma: float = 0.0
    beta: float = 0.3
    # Time grid. Twisting rate chi ~ J^2/Omega is slow, so allow generous t_max.
    t_max: float = 20.0
    n_steps: int = 120
    # Central finite-difference step for d(rho)/dJ.
    dJ: float = 1e-3
    # "full"      : H = (Omega/2) sx0 + J sz0 Jz, Lindblad, bath QFI via ptrace.
    # "effective" : bath-only H_eff = +/- (J^2/(2 Omega)) Jz^2 branch check (noiseless,
    #               sanity check of the OAT mechanism without central-spin overhead).
    model: str = "full"


cfg = Config()

# ============================================================
# Operators / states (symmetric subspace)
# ============================================================


def build_system(N):
    """Central qubit (x) collective bath spin S = N/2. dim = 2*(N+1)."""
    S = N / 2.0
    jx, jy, jz = qt.jmat(S, "x"), qt.jmat(S, "y"), qt.jmat(S, "z")
    Ib = qt.qeye(int(2 * S + 1))
    Is = qt.qeye(2)
    ops = {
        "sx_s": qt.tensor(qt.sigmax(), Ib),
        "sy_s": qt.tensor(qt.sigmay(), Ib),
        "sz_s": qt.tensor(qt.sigmaz(), Ib),
        "Jz": qt.tensor(Is, jz),
        "Jx": qt.tensor(Is, jx),
        "jz_b": jz,  # bath-only
    }
    # Central spin |+x>, bath = spin coherent state along +x (all spins in |+>).
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    bath0 = qt.spin_coherent(S, np.pi / 2, 0.0)
    ops["psi0"] = qt.tensor(plus, bath0)
    ops["bath0"] = bath0
    return ops


# ============================================================
# Vectorized QFI
# ============================================================


def qfi_vectorized(rho, drho, tol=1e-12):
    """
    F_Q = 2 * sum_{m,n : p_m + p_n > tol} |<m|drho|n>|^2 / (p_m + p_n).
    rho, drho: dense numpy arrays. Replaces the old nested Python loops
    with eigh + broadcasting (BLAS-bound, ~100-1000x faster).
    """
    evals, evecs = np.linalg.eigh(rho)
    evals = np.clip(evals, 0.0, None)
    M = evecs.conj().T @ drho @ evecs
    denom = evals[:, None] + evals[None, :]
    mask = denom > tol
    return 2.0 * np.sum(np.abs(M[mask]) ** 2 / denom[mask])


# ============================================================
# Bath QFI trajectory
# ============================================================


def _evolve_bath_states(J, Omega, N, cfg, ops, tlist):
    """Return list of bath density matrices (numpy) at each t, for coupling J."""
    if cfg.model == "effective":
        # Bath-only one-axis twisting: H_eff = (J^2 / (2 Omega)) Jz^2, noiseless.
        chi = J**2 / (2.0 * Omega) if Omega > 0 else 0.0
        H = chi * ops["jz_b"] ** 2 if Omega > 0 else J * ops["jz_b"]
        res = qt.sesolve(H, ops["bath0"], tlist)
        return [qt.ket2dm(s).full() for s in res.states]

    # Full model, solved EXACTLY via block decomposition (no ODE integration).
    # The bath enters H only through Jz (diagonal) and the noise acts on the
    # central spin alone, so each bath matrix element (m, n) evolves under an
    # independent 4x4 constant-coefficient Liouvillian, solved by eigendecomp.
    # This matters: mesolve + finite-difference drho/dJ (dJ ~ 1e-3) amplifies
    # solver error ~500x and the QFI formula divides by small eigenvalues --
    # with default tolerances this inflated driven-case QFI enough to
    # manufacture spurious driven-vs-undriven crossovers at large N.
    return _bath_states_block(J, Omega, N, cfg, tlist)


def _bath_states_block(J, Omega, N, cfg, tlist):
    S = N / 2.0
    m_vals = np.arange(-S, S + 1)                      # Jz eigenvalues, spacing 1
    chi = qt.spin_coherent(S, np.pi / 2, 0).full().ravel()
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2, I4 = np.eye(2, dtype=complex), np.eye(4, dtype=complex)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    r0 = np.outer(plus, plus.conj()).reshape(4)        # central spin |+><+|
    L_diss = (cfg.beta * (np.kron(sx, sx.T) - I4)
              + cfg.gamma * (np.kron(sz, sz.T) - I4))
    d, nt = len(m_vals), len(tlist)
    bath = np.zeros((nt, d, d), dtype=complex)
    for i in range(d):
        Hi = 0.5 * Omega * sx + J * m_vals[i] * sz
        for j in range(i, d):
            Hj = 0.5 * Omega * sx + J * m_vals[j] * sz
            L = -1j * (np.kron(Hi, I2) - np.kron(I2, Hj.T)) + L_diss
            evals, V = np.linalg.eig(L)
            c0 = np.linalg.solve(V, r0) * (chi[i] * np.conj(chi[j]))
            rt = V @ (np.exp(np.outer(evals, tlist)) * c0[:, None])
            tr = rt[0] + rt[3]                         # Tr over central spin
            bath[:, i, j] = tr
            if j != i:
                bath[:, j, i] = np.conj(tr)
    return [bath[k] for k in range(nt)]


def cfg_psi0(ops):
    return ops["psi0"]


def compute_bath_qfi_trajectory(J, Omega, N, cfg, ops, tlist):
    """
    QFI(t) of the reduced bath state w.r.t. J, via central finite difference:
      drho/dJ ~ (rho(J+dJ) - rho(J-dJ)) / (2 dJ),  rho ~ average of the two.
    Two solver runs total per (Omega, N) point.
    """
    rp = _evolve_bath_states(J + cfg.dJ, Omega, N, cfg, ops, tlist)
    rm = _evolve_bath_states(J - cfg.dJ, Omega, N, cfg, ops, tlist)
    fq = np.empty(len(tlist))
    for k in range(len(tlist)):
        rho = 0.5 * (rp[k] + rm[k])
        drho = (rp[k] - rm[k]) / (2.0 * cfg.dJ)
        fq[k] = qfi_vectorized(rho, drho)
    return fq


# ============================================================
# Sweep: FQ_max vs N at fixed Omega/J
# ============================================================


def run_sweep(cfg):
    """
    Figure of merit: R = max_t F_Q(t)/t  (information rate).
    With a fixed total time budget T, a duration-t run can be repeated T/t
    times, so total info = (T/t) F(t) = T * [F(t)/t]. Maximizing raw F(t)
    would wrongly favor one long shot over many short repetitions.
    """
    tlist = np.linspace(0.0, cfg.t_max, cfg.n_steps)
    results = {}  # ratio -> (N_values, R_max = max_t FQ/t, t_star)
    for ratio in cfg.Omega_over_J_values:
        r_max, t_star = [], []
        for N in cfg.N_values:
            ops = build_system(N)
            Omega = ratio * cfg.J
            fq = compute_bath_qfi_trajectory(cfg.J, Omega, N, cfg, ops, tlist)
            rate = fq[1:] / tlist[1:]  # skip t=0
            k = int(np.argmax(rate)) + 1
            r_max.append(rate[k - 1])
            t_star.append(tlist[k])
            print(
                f"Omega/J={ratio:4.1f}  N={N:3d}  max FQ/t={rate[k - 1]:.4e}  t*={tlist[k]:.2f}"
            )
        results[ratio] = (
            np.array(cfg.N_values, float),
            np.array(r_max),
            np.array(t_star),
        )
    return results


def fit_power_law(N, FQ):
    """Fit FQ ~ c * N^p on the upper half of the N range; return p."""
    m = len(N) // 2
    lo = max(0, m - 1)
    good = FQ[lo:] > 0
    if good.sum() < 2:
        return np.nan
    p, _ = np.polyfit(np.log(N[lo:][good]), np.log(FQ[lo:][good]), 1)
    return p


# ============================================================
# Plotting
# ============================================================


def plot_results(results, cfg, fname="fq_vs_N_scaling.png"):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    base = results.get(0.0)
    for ratio, (N, FQ, ts) in sorted(results.items()):
        p = fit_power_law(N, FQ)
        lbl = (
            "undriven (SQL floor)" if ratio == 0.0 else f"$\\Omega/J={ratio:g}$"
        ) + f",  fit $N^{{{p:.2f}}}$"
        ax1.loglog(N, FQ, "o-", label=lbl)
    ax1.set_xlabel("N")
    ax1.set_ylabel(r"$\max_t\, F_Q(t)/t$")
    ax1.set_title(
        f"Bath QFI rate vs N  (model={cfg.model}, "
        f"$\\gamma$={cfg.gamma}, $\\beta$={cfg.beta})"
    )
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    if base is not None:
        Nb, FQb, _ = base
        for ratio, (N, FQ, _) in sorted(results.items()):
            if ratio == 0.0:
                continue
            ax2.semilogx(N, FQ / FQb, "o-", label=f"$\\Omega/J={ratio:g}$")
            Nc = ratio**2  # predicted crossover N ~ (Omega/J)^2
            if N.min() <= Nc <= N.max():
                ax2.axvline(Nc, ls=":", color=ax2.lines[-1].get_color(), alpha=0.5)
        ax2.axhline(1.0, color="k", lw=1)
        ax2.set_xlabel("N")
        ax2.set_ylabel(r"$(F_Q/t)^{driven}\,/\,(F_Q/t)^{undriven}$")
        ax2.set_title("Driven / undriven ratio (dotted: $N=(\\Omega/J)^2$)")
        ax2.legend(fontsize=8)
        ax2.grid(True, which="both", alpha=0.3)

    for ratio, (N, _, ts) in sorted(results.items()):
        lbl = "undriven (SQL floor)" if ratio == 0.0 else f"$\\Omega/J={ratio:g}$"
        ax3.loglog(N, ts, "o-", label=lbl)
    ax3.set_xlabel("N")
    ax3.set_ylabel(r"$t^*(N)$  (argmax of $F_Q(t)/t$)")
    ax3.set_title("Optimal integration time vs N")
    ax3.legend(fontsize=8)
    ax3.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = run_sweep(cfg)
    plot_results(results, cfg)

    # Summary of fitted exponents.
    print("\nFitted power laws (FQ/t ~ N^p, upper half of N range):")
    for ratio, (N, FQ, _) in sorted(results.items()):
        print(
            f"  Omega/J = {ratio:4.1f}:  p = {fit_power_law(N, FQ):.3f}"
            + (
                "   (expect ~1, SQL)"
                if ratio == 0.0
                else "   (expect ->2 if OAT mechanism survives noise)"
            )
        )
