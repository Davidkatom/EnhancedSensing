"""Single-qubit dynamics for H(t) = Omega X + N J cos(omega t) Z, with hbar = 1.

X, Y, Z are Pauli matrices (no factor of 1/2). Time and frequencies use a
fixed reference unit. Lowercase omega is the modulation angular frequency,
with default 1; uppercase Omega is the transverse coupling. N is a fixed
dimensionless scale factor and J is the longitudinal coupling. The initial
states are |0> and |+>.

By default the QFI estimates J, holding Omega, omega, N, and time fixed, with J = 1.
Change ``qfi`` to "Omega", "N", or "t" to estimate another parameter.
For parameter estimation, integrate the sensitivity equation
    d_t(d_theta psi) = -i H d_theta psi - i (d_theta H) psi.
For time estimation, d_t psi = -i H(t) psi directly.

The right panels also show the classical Fisher information of separate
projective X, Y, and Z measurements. For a Pauli observable A this is
    F_C(A) = (d_theta <A>)^2 / (1 - <A>^2)
when both outcomes have nonzero probability. The shared projective-FI
helper sums (d_theta p)^2/p over outcomes with p > cfi_tol; zero-probability
outcomes are omitted, including at exactly deterministic measurements.

Run from the repository root:
    python CRB/LandauZener/driven_qubit.py
    python CRB/LandauZener/driven_qubit.py --Omega 2 --omega 0.5 --t-pi 8 --no-show

The 2x2 figure and its complete numerical .plot.json record are saved
together by CRB.smart_save under PhD/Graphs/driven_qubit/spin_and_fisher_vs_time.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp

# Support both direct execution and `python -m CRB.LandauZener.driven_qubit`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CRB.crb_core import (
    central_spin_state,
    observable_projective_fisher,
    qfi_from_rho_and_drho,
)
from CRB.smart_save import PlotRecord, save_plot


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Physics, sampling, solver, and figure settings for one run.

    Omega, J, and omega use a common inverse-time unit; N is dimensionless.
    Evolution starts at zero and ends at ``t_pi * pi``. ``size`` is in inches.
    """

    N: float = 15.0
    Omega: float = 0.6*np.sqrt(N)
    omega: float = 1.0
    J: float = 1.0
    qfi: str = "J"
    t_pi: float = 1.0
    points: int = 2001
    rtol: float = 1e-9
    atol: float = 1e-11
    method: str = "DOP853"
    qfi_tol: float = 1e-10
    cfi_tol: float = 1e-12
    size: tuple[float, float] = (12.0, 7.0)
    dpi: int = 160
    show: bool = True

    def __post_init__(self) -> None:
        if not np.all(np.isfinite([self.N, self.Omega, self.J])):
            raise ValueError("N, Omega, and J must be finite.")
        if not np.isfinite(self.omega) or self.omega < 0:
            raise ValueError("omega must be finite and non-negative (0 gives a static drive).")
        if self.qfi not in {"J", "Omega", "N", "t"}:
            raise ValueError("qfi must be 'J', 'Omega', 'N', or 't'.")
        for name in ("t_pi", "rtol", "atol", "qfi_tol", "cfi_tol"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not isinstance(self.points, int) or self.points < 2:
            raise ValueError("points must be an integer of at least 2.")
        if self.method not in {"DOP853", "RK45", "RK23"}:
            raise ValueError("method must be DOP853, RK45, or RK23.")
        if self.qfi_tol >= 1:
            raise ValueError("qfi_tol must be smaller than 1.")
        if self.cfi_tol >= 0.5:
            raise ValueError("cfi_tol must be smaller than 0.5.")
        if len(self.size) != 2 or not all(np.isfinite(v) and v > 0 for v in self.size):
            raise ValueError("size must contain two finite positive dimensions.")
        if not isinstance(self.dpi, int) or self.dpi <= 0:
            raise ValueError("dpi must be a positive integer.")


@dataclass(frozen=True, slots=True)
class Trajectory:
    time: np.ndarray
    states: np.ndarray
    tangents: np.ndarray
    bloch: np.ndarray
    qfi: np.ndarray
    cfi: np.ndarray  # Shape (points, 3), ordered X, Y, Z.
    norm_error: float


def simulate(cfg: SimulationConfig, theta: float) -> Trajectory:
    """Evolve central_spin_state(theta) and its parameter derivative.

    ``theta=0`` gives |0>; ``theta=pi/2`` gives |+>. Initial states do not
    depend on the estimated parameter, so their parameter tangents start at 0.
    """
    pauli = np.array([qt.sigmax().full(), qt.sigmay().full(), qt.sigmaz().full()])
    x, _, z = pauli
    initial = central_spin_state(theta).full().ravel()
    time = np.linspace(0.0, cfg.t_pi * np.pi, cfg.points)

    def rhs(t: float, augmented: np.ndarray) -> np.ndarray:
        state, tangent = augmented.reshape(2, 2)
        modulation = np.cos(cfg.omega * t)
        h = cfg.Omega * x + cfg.N * cfg.J * modulation * z
        if cfg.qfi == "J":
            dh = cfg.N * modulation * z
        elif cfg.qfi == "Omega":
            dh = x
        elif cfg.qfi == "N":
            dh = cfg.J * modulation * z
        else:
            dh = np.zeros_like(h)
        return np.concatenate((-1j * h @ state, -1j * (h @ tangent + dh @ state)))

    solution = solve_ivp(
        rhs,
        (time[0], time[-1]),
        np.concatenate((initial, np.zeros(2, dtype=complex))),
        t_eval=time,
        method=cfg.method,
        rtol=cfg.rtol,
        atol=cfg.atol,
    )
    if not solution.success:
        raise RuntimeError(f"Schrodinger integration failed: {solution.message}")

    raw_states = solution.y[:2].T
    raw_tangents = solution.y[2:].T
    norms = np.linalg.norm(raw_states, axis=1)
    norm_error = float(np.max(np.abs(norms**2 - 1.0)))
    states = raw_states / norms[:, None]
    # Differentiate the numerical normalization as well as the raw state.
    norm_derivative = np.einsum("ti,ti->t", states.conj(), raw_tangents).real
    tangents = (raw_tangents - states * norm_derivative[:, None]) / norms[:, None]
    if cfg.qfi == "t":
        h = cfg.Omega * x + cfg.N * cfg.J * np.cos(cfg.omega * time)[:, None, None] * z
        tangents = -1j * np.einsum("tij,tj->ti", h, states)

    bloch = np.einsum("ti,aij,tj->ta", states.conj(), pauli, states).real
    qfi = np.empty(cfg.points)
    cfi = np.empty((cfg.points, 3))
    for index, (state, tangent) in enumerate(zip(states, tangents)):
        rho = np.outer(state, state.conj())
        drho = np.outer(tangent, state.conj()) + np.outer(state, tangent.conj())
        qfi[index], _ = qfi_from_rho_and_drho(rho, drho, tol=cfg.qfi_tol)
        for component, observable in enumerate(pauli):
            cfi[index, component] = observable_projective_fisher(
                rho, drho, observable, tol=cfg.cfi_tol
            )
    return Trajectory(time, states, tangents, bloch, qfi, cfi, norm_error)


def figure_filename(cfg: SimulationConfig) -> str:
    """Record every configuration field in a stable, readable filename."""
    def tag(value: object) -> str:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            return str(int(value))
        if isinstance(value, float):
            return str(int(value)) if value.is_integer() else repr(value)
        if isinstance(value, tuple):
            return "x".join(tag(item) for item in value)
        return re.sub(r'[<>:"/\\|?*\s]+', "-", str(value))

    parameters = "__".join(f"{field.name}={tag(getattr(cfg, field.name))}" for field in fields(cfg))
    return f"spin-fisher__{parameters}.png"


def plot_trajectories(cfg: SimulationConfig, trajectories: tuple[Trajectory, Trajectory]) -> PlotRecord:
    """Plot |0> above |+>, with Bloch components left and QFI/CFI right."""
    figure, axes = plt.subplots(2, 2, figsize=cfg.size, sharex=True, layout="constrained")
    parameter = {"J": "J", "Omega": r"\Omega", "N": "N", "t": "t"}[cfg.qfi]
    guide_times = np.arange(int(np.floor(2 * cfg.t_pi)) + 1) * (np.pi / 2)
    for row, (label, result) in enumerate(zip(("0", "+"), trajectories)):
        spin_axis, qfi_axis = axes[row]
        for component, name in enumerate(("X", "Y", "Z")):
            spin_axis.plot(result.time, result.bloch[:, component], label=rf"$\langle {name}\rangle$", lw=1.3)
        spin_axis.set(title=rf"Initial state $|{label}\rangle$ — Pauli expectations", ylabel="Expectation value", ylim=(-1.08, 1.08))
        spin_axis.legend(loc="lower left", ncol=3)
        qfi_axis.plot(result.time, result.qfi, color="tab:purple", lw=2.1, label="QFI", zorder=3)
        for component, name in enumerate(("X", "Y", "Z")):
            qfi_axis.plot(result.time, result.cfi[:, component], color=f"C{component}",
                          lw=1.2, alpha=0.85, label=rf"CFI $\langle {name}\rangle$")
        qfi_axis.set(title=rf"Initial state $|{label}\rangle$ — QFI and CFI for ${parameter}$",
                     ylabel=rf"Fisher information for ${parameter}$", ylim=(0, None))
        qfi_axis.legend(loc="upper left", ncol=4, fontsize=9)
        for axis in axes[row]:
            axis.grid(True, axis="y", alpha=0.25)
            for guide_time in guide_times:
                axis.axvline(guide_time, color="0.4", linestyle="--", linewidth=0.9, alpha=0.65, zorder=0)
            axis.set_xlim(result.time[0], result.time[-1])
    for axis in axes[-1]:
        axis.set_xlabel(r"Time $t$")
    figure.suptitle(rf"$H(t)=\Omega X + NJ\cos(\omega t)Z$, $\Omega={cfg.Omega:g}$, $\omega={cfg.omega:g}$, $N={cfg.N:g}$, $J={cfg.J:g}$, $\hbar=1$")

    # Follow plot_driven_qfi_map: pass every numerical curve explicitly to
    # one explorer-mode save, which creates the paired PNG and JSON record.
    data = {}
    for label, result in zip(("|0>", "|+>"), trajectories):
        for component, name in enumerate(("X", "Y", "Z")):
            data[f"{label}: <{name}>"] = (result.time, result.bloch[:, component])
        data[f"{label}: QFI({cfg.qfi})"] = (result.time, result.qfi)
        for component, name in enumerate(("X", "Y", "Z")):
            data[f"{label}: CFI({cfg.qfi}) <{name}>"] = (result.time, result.cfi[:, component])

    try:
        record = save_plot(
            figure,
            system="driven_qubit",
            plot_type="spin_and_fisher_vs_time",
            params=asdict(cfg),
            data=data,
            name=Path(figure_filename(cfg)).stem,
            xlabel=r"Time $t$",
            ylabel="Pauli expectation value / Fisher information",
            xunit="dimensionless",
            notes=(
                "2x2 preview: |0> on the top row, |+> on the bottom; "
                "Pauli expectations on the left, QFI and CFI on the right. "
                "H = Omega X + N J cos(omega t) Z; Pauli matrices; hbar = 1. "
                "CFI sums (d_parameter p)^2/p over outcomes with p > cfi_tol."
            ),
            metadata={
                "vertical_guide_spacing": "pi/2",
                "maximum_norm_errors": [result.norm_error for result in trajectories],
            },
            script_path=__file__,
            format="png",
            dpi=cfg.dpi,
            bbox_inches="tight",
        )
        if cfg.show:
            plt.show()
        return record
    finally:
        plt.close(figure)


def main() -> None:
    defaults = SimulationConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--N", type=float, default=defaults.N)
    parser.add_argument("--Omega", type=float, default=defaults.Omega, help="Transverse coupling.")
    parser.add_argument("--omega", "--w", dest="omega", type=float, default=defaults.omega,
                        help="Modulation angular frequency in cos(omega*t), default: 1.")
    parser.add_argument("--J", type=float, default=defaults.J)
    parser.add_argument("--qfi", choices=("J", "Omega", "N", "t"), default=defaults.qfi)
    parser.add_argument("--t-pi", type=float, default=defaults.t_pi,
                        help=f"Final time in multiples of pi (default: {defaults.t_pi:g}).")
    parser.add_argument("--points", type=int, default=defaults.points)
    parser.add_argument("--rtol", type=float, default=defaults.rtol)
    parser.add_argument("--atol", type=float, default=defaults.atol)
    parser.add_argument("--method", choices=("DOP853", "RK45", "RK23"), default=defaults.method)
    parser.add_argument("--qfi-tol", type=float, default=defaults.qfi_tol)
    parser.add_argument("--cfi-tol", type=float, default=defaults.cfi_tol)
    parser.add_argument("--size", nargs=2, type=float, default=defaults.size, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int, default=defaults.dpi)
    parser.add_argument("--no-show", action="store_false", dest="show", default=defaults.show)
    arguments = vars(parser.parse_args())
    arguments["size"] = tuple(arguments["size"])
    cfg = SimulationConfig(**arguments)
    trajectories = (simulate(cfg, 0.0), simulate(cfg, np.pi / 2.0))
    for label, result in zip(("0", "+"), trajectories):
        print(f"|{label}>: max norm error = {result.norm_error:.3e}; max QFI({cfg.qfi}) = {result.qfi.max():.6g}")
    record = plot_trajectories(cfg, trajectories)
    print(f"Saved numerical data: {record.json_path}")
    print(f"Saved figure: {record.image_path}")


if __name__ == "__main__":
    main()
