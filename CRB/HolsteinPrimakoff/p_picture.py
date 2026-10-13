"""Bath QFI of the linearized central-spin model without back-action, through the conserved quadrature P, hbar = 1.

Rotating frame of the bath drive, linearized Holstein-Primakoff, back-action -2J a^dag a cos(2 omega t) Z dropped:
    H = Omega X + J Z [N cos(2 omega t) + sqrt(N) P sin(2 omega t)],   P = i(a^dag - a).
P is conserved, so each eigenvalue p carries the single-qubit problem H_p = Omega X + h_p(t) Z with
    h_p(t) = J sqrt(N^2 + N p^2) cos(2 omega t - theta_p),   tan(theta_p) = p / sqrt(N):
the same sweep for every p, shifted to the crossing time t_c(p) = (pi/2 + theta_p) / (2 omega) with sweep rate
v_p = 2 omega J sqrt(N^2 + N p^2).  The bath vacuum weights p by |phi(p)|^2 = N(0, 1).  On a p grid the bath is
psi_k(p) = sqrt(dp) phi(p) c_k(p), with c_k(p) the amplitude of central-spin state k, and rho_B = sum_k psi_k psi_k^dag.

Three levels of description, each giving the reduced-bath QFI for J at readout time t:
  * ensemble_bath_qfi -- every p solved numerically; identical to the Fock-basis model (fock_bath_qfi).
  * aim_bath_qfi      -- adiabatic impulse: c = D(zeta_a) S D(zeta_b) |+> with D(z) = exp(-i z Z), the diabatic
                         phases zeta_b, zeta_a before and after the p-shifted crossing, and the Landau-Zener matrix
                         S = [[sqrt(P), -sqrt(1-P) e^{-i phi_S}], [sqrt(1-P) e^{i phi_S}, sqrt(P)]].
  * leading_bath_qfi  -- the limit of well separated bath pointers, 4 P (1 - P) (N / omega)^2 with
                         P = exp(-pi Omega^2 / 2 J N omega); maximal at P = 1/2, i.e. the LZ 50:50 drive.
Before the crossing the two Z components pick up -+(J / 2 omega)(N + sqrt(N) p).  The p part shifts the bath's
X quadrature (the which-path record); the two paths into the same final Z state keep the relative phase
J N / omega, frozen at the crossing, and its J-derivative N / omega gives the (N / omega)^2.

Run from the repository root to compare the levels with the exact model at one readout time:
    python CRB/HolsteinPrimakoff/p_picture.py
    python CRB/HolsteinPrimakoff/p_picture.py --t 1.2 --N-values 15 30 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import loggamma

# Support both direct execution and `python -m CRB.HolsteinPrimakoff.p_picture`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CRB.crb_core import coherent_bath_state, evolve_bath_density_matrix_noiseless, qfi_from_rho_and_drho

Z_SIGNS = np.array([[1.0], [-1.0]])


def p_grid(p_max: float = 9.0, points: int = 721) -> tuple[np.ndarray, np.ndarray]:
    """Return the p grid and the discretized vacuum weights sqrt(dp) phi(p), phi(p)^2 = N(0, 1)."""
    p = np.linspace(-p_max, p_max, points)
    return p, np.sqrt(p[1] - p[0]) * (2 * np.pi) ** -0.25 * np.exp(-p**2 / 4)


def qfi_from_amplitude_pairs(plus: np.ndarray, minus: np.ndarray, dJ: float) -> float:
    """Bath QFI from (2, M) amplitudes at J +- dJ, computed on their joint support of dimension <= 4."""
    basis, _ = np.linalg.qr(np.vstack([plus, minus]).T)
    def rho(amplitudes: np.ndarray) -> np.ndarray:
        projected = basis.conj().T @ amplitudes.T
        return projected @ projected.conj().T
    rho_plus, rho_minus = rho(plus), rho(minus)
    return qfi_from_rho_and_drho(0.5 * (rho_plus + rho_minus), (rho_plus - rho_minus) / (2 * dJ), tol=1e-12)[0]


def sweep(p: np.ndarray, t: np.ndarray | float, N: int, J: float, omega: float) -> np.ndarray:
    """h_p(t) = J [N cos(2 omega t) + sqrt(N) p sin(2 omega t)]."""
    return J * (N * np.cos(2 * omega * t) + np.sqrt(N) * p * np.sin(2 * omega * t))


def ensemble_amplitudes(N: int, Omega: float, t: float, J: float, omega: float, p: np.ndarray) -> np.ndarray:
    """Solve H_p = Omega X + h_p(t) Z from |+> for every p at once; returns c_k(p) with shape (2, p.size)."""
    def rhs(s: float, y: np.ndarray) -> np.ndarray:
        c = y.reshape(2, p.size)
        return (-1j * (Omega * c[::-1] + sweep(p, s, N, J, omega) * Z_SIGNS * c)).ravel()
    initial = np.full((2, p.size), 2 ** -0.5, dtype=complex).ravel()
    solution = solve_ivp(rhs, (0.0, t), initial, t_eval=[t], rtol=1e-10, atol=1e-12, method="DOP853")
    return solution.y[:, -1].reshape(2, p.size)


def aim_amplitudes(N: int, Omega: float, t: float, J: float, omega: float, p: np.ndarray) -> np.ndarray:
    """Adiabatic-impulse c_k(p) = [D(zeta_a) S D(zeta_b) |+>]_k; p whose crossing is after t see no splitter."""
    theta = np.arctan(p / np.sqrt(N))
    crossing = (np.pi / 2 + theta) / (2 * omega)
    def phase(a, b):  # integral of h_p from a to b
        return (J / (2 * omega)) * (N * (np.sin(2 * omega * b) - np.sin(2 * omega * a))
                                    + np.sqrt(N) * p * (np.cos(2 * omega * a) - np.cos(2 * omega * b)))
    crossed = crossing < t
    zeta_b = np.where(crossed, phase(0.0, crossing), phase(0.0, t))
    zeta_a = np.where(crossed, phase(crossing, t), 0.0)
    rate = 2 * omega * J * np.sqrt(N**2 + N * p**2)
    P = np.where(crossed, np.exp(-np.pi * Omega**2 / rate), 1.0)
    delta = Omega**2 / (2 * rate)  # P = exp(-2 pi delta)
    stokes = np.pi / 4 + delta * (np.log(delta) - 1) + np.imag(loggamma(1 - 1j * delta))
    before = np.vstack([np.exp(-1j * zeta_b), np.exp(1j * zeta_b)]) / np.sqrt(2)
    after_splitter = np.vstack([np.sqrt(P) * before[0] - np.sqrt(1 - P) * np.exp(-1j * stokes) * before[1],
                                np.sqrt(1 - P) * np.exp(1j * stokes) * before[0] + np.sqrt(P) * before[1]])
    return np.vstack([np.exp(-1j * zeta_a), np.exp(1j * zeta_a)]) * after_splitter


def ensemble_bath_qfi(N: int, Omega: float, t: float, J: float = 1.0, omega: float = 1.0,
                      dJ: float = 1e-5) -> float:
    p, weight = p_grid()
    return qfi_from_amplitude_pairs(*(weight * ensemble_amplitudes(N, Omega, t, j, omega, p)
                                      for j in (J + dJ, J - dJ)), dJ)


def aim_bath_qfi(N: int, Omega: float, t: float, J: float = 1.0, omega: float = 1.0, dJ: float = 1e-5) -> float:
    p, weight = p_grid()
    return qfi_from_amplitude_pairs(*(weight * aim_amplitudes(N, Omega, t, j, omega, p)
                                      for j in (J + dJ, J - dJ)), dJ)


def leading_bath_qfi(N: int, Omega: float, J: float = 1.0, omega: float = 1.0) -> float:
    P = np.exp(-np.pi * Omega**2 / (2 * J * N * omega))
    return 4 * P * (1 - P) * (N / omega) ** 2


def fock_bath_qfi(N: int, Omega: float, t: float, J: float = 1.0, omega: float = 1.0, dJ: float = 1e-5,
                  n_max: int | None = None) -> float:
    """The same no-back-action model in the boson number basis (cutoff 4N + 40 by default)."""
    dim = (4 * N + 40 if n_max is None else n_max) + 1
    a = np.diag(np.sqrt(np.arange(1, dim)), 1).astype(complex)
    coupling = np.sqrt(N) * 1j * (a.conj().T - a)
    def amplitudes(j: float) -> np.ndarray:
        def rhs(s: float, y: np.ndarray) -> np.ndarray:
            A = y.reshape(2, dim)
            field = np.cos(2 * omega * s) * N * A + np.sin(2 * omega * s) * A @ coupling.T
            return (-1j * (Omega * A[::-1] + j * Z_SIGNS * field)).ravel()
        initial = np.zeros((2, dim), dtype=complex)
        initial[:, 0] = 2 ** -0.5
        solution = solve_ivp(rhs, (0.0, t), initial.ravel(), t_eval=[t], rtol=1e-10, atol=1e-12, method="DOP853")
        A = solution.y[:, -1].reshape(2, dim)
        if np.sum(np.abs(A[:, -5:]) ** 2) > 1e-8:
            raise RuntimeError(f"boson cutoff {dim - 1} too small for N={N}, t={t}")
        return A
    return qfi_from_amplitude_pairs(amplitudes(J + dJ), amplitudes(J - dJ), dJ)


def exact_bath_qfi(N: int, Omega: float, t: float, J: float = 1.0, omega: float = 1.0, dJ: float = 1e-5) -> float:
    """Full model with back-action and spin nonlinearity, for reference."""
    bath = coherent_bath_state(N, 0.0)
    plus, minus = (evolve_bath_density_matrix_noiseless(Omega, j, t, N, bath, omega=omega) for j in (J + dJ, J - dJ))
    return qfi_from_rho_and_drho(0.5 * (plus + minus), (plus - minus) / (2 * dJ), tol=1e-10)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--N-values", type=int, nargs="+", default=[15, 30, 60, 100])
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--omega", "--w", dest="omega", type=float, default=1.0)
    parser.add_argument("--c-points", type=int, default=23, help="Omega grid points over [0.3, 1.4] sqrt(J N omega).")
    args = parser.parse_args()
    models = {
        "ensemble": lambda N, Om: ensemble_bath_qfi(N, Om, args.t, args.J, args.omega),
        "AIM": lambda N, Om: aim_bath_qfi(N, Om, args.t, args.J, args.omega),
        "leading": lambda N, Om: leading_bath_qfi(N, Om, args.J, args.omega),
        "exact": lambda N, Om: exact_bath_qfi(N, Om, args.t, args.J, args.omega),
    }
    print(f"t={args.t:g}, J={args.J:g}, omega={args.omega:g}: Omega_opt / F_B,max (N/omega)^-2 on the grid")
    for N in args.N_values:
        grid = np.linspace(0.3, 1.4, args.c_points) * np.sqrt(args.J * N * args.omega)
        cells = []
        for name, model in models.items():
            values = np.array([model(N, Omega) for Omega in grid])
            best = int(np.argmax(values))
            cells.append(f"{name} {grid[best]:.2f} / {values[best] * (args.omega / N) ** 2:.3f}")
        print(f"N={N:4d}:  " + "  |  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
