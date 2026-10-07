"""Vacuum momentum evolution for the HP model with the number term omitted.

H(t) = Omega X + J Z [N cos(2 omega t) + sqrt(N) P sin(2 omega t)],
P = i(a^dagger - a), hbar = 1.  The vacuum momentum distribution is a
unit-variance normal distribution. Central-spin observables are its weighted
average of the conditional qubit observables; bath coherences are unnecessary
for these central-spin observables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import solve_ivp

from CRB.crb_core import central_spin_state
from CRB.HolsteinPrimakoff.hp_dynamics import PAULI


@dataclass(frozen=True, slots=True)
class MomentumConfig:
    """Frequencies are inverse reference-time units; theta and phi are radians."""

    N: int = 15
    Omega: float = 2.2
    J: float = 1.0
    omega: float = 1.0
    tmax: float = 2.0
    points: int = 2001
    p_nodes: int = 64
    theta: float = np.pi / 2.0
    phi: float = 0.0
    rtol: float = 1e-9
    atol: float = 1e-11
    method: str = "DOP853"

    def __post_init__(self) -> None:
        if not isinstance(self.N, int) or self.N < 1:
            raise ValueError("N must be a positive integer.")
        for name in ("Omega", "J", "omega", "theta", "phi"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if self.omega < 0:
            raise ValueError("omega must be non-negative.")
        for name in ("tmax", "rtol", "atol"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("points", "p_nodes"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 2:
                raise ValueError(f"{name} must be an integer of at least 2.")
        if self.method not in {"DOP853", "RK45", "RK23"}:
            raise ValueError("method must be DOP853, RK45, or RK23.")


@dataclass(frozen=True, slots=True)
class MomentumTrajectory:
    time: np.ndarray
    bloch: np.ndarray
    states: np.ndarray  # (time, momentum node, central-spin index)
    momenta: np.ndarray
    weights: np.ndarray
    norm_error: float


def simulate_momentum(cfg: MomentumConfig) -> MomentumTrajectory:
    """Evolve the initial central state times boson vacuum and trace the bath."""
    nodes, weights = hermgauss(cfg.p_nodes)
    momenta = np.sqrt(2.0) * nodes
    weights = weights / np.sqrt(np.pi)
    time = np.linspace(0.0, cfg.tmax, cfg.points)
    initial = np.broadcast_to(
        central_spin_state(cfg.theta, cfg.phi).full().ravel(), (cfg.p_nodes, 2)
    ).copy()
    z_signs = np.array([1.0, -1.0])

    def rhs(t: float, flat: np.ndarray) -> np.ndarray:
        states = flat.reshape(cfg.p_nodes, 2)
        angle = 2.0 * cfg.omega * t
        field = cfg.J * (
            cfg.N * np.cos(angle) + np.sqrt(cfg.N) * momenta * np.sin(angle)
        )
        return (-1j * (cfg.Omega * states[:, ::-1] + field[:, None] * z_signs * states)).ravel()

    solution = solve_ivp(
        rhs, (0.0, cfg.tmax), initial.ravel(), t_eval=time,
        method=cfg.method, rtol=cfg.rtol, atol=cfg.atol,
    )
    if not solution.success:
        raise RuntimeError(f"Momentum evolution failed: {solution.message}")
    states = solution.y.T.reshape(cfg.points, cfg.p_nodes, 2)
    norms_squared = np.sum(np.abs(states) ** 2, axis=2)
    norm_error = float(np.max(np.abs(norms_squared - 1.0)))
    states = states / np.sqrt(norms_squared[:, :, None])
    bloch = np.einsum("q,tqi,aij,tqj->ta", weights, states.conj(), PAULI, states).real
    return MomentumTrajectory(time, bloch, states, momenta, weights, norm_error)
