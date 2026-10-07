"""Holstein-Primakoff dynamics of the driven central spin and collective bath, hbar = 1.

The full model of CRB.crb_core (Pauli convention, no factors of 1/2) is
    H = Omega X + J Z S_z + omega S_x,   S_a = sum_i sigma_a^(i).
In the frame of the bath drive, U_B = exp(-i omega S_x t), it becomes exactly
    H_I(t) = Omega X + J Z [S_z cos(2 omega t) + S_y sin(2 omega t)].
The initial state is |+>|S_z = N>: the central spin in |+> and the bath fully
polarized, |0>^N = |N/2, N/2>.  Holstein-Primakoff about that state,
    S_z = N - 2 a^dag a,   S_y = -i [sqrt(N - a^dag a) a - a^dag sqrt(N - a^dag a)],
maps it to the boson vacuum |n = 0>, with n the number of flipped bath spins.
Linearizing S_y ~ sqrt(N) P, with P = i(a^dag - a) and <0|P^2|0> = 1, gives
    H_HP(t) = Omega X + J Z [(N - 2 a^dag a) cos(2 omega t) + sqrt(N) P sin(2 omega t)].

Both models have the form Omega X + J Z B(t) and are integrated with their
J-sensitivity, d_t(d_J psi) = -i H d_J psi - i Z B(t) psi:
  * "HP":    the linearized boson, truncated at n_max (default 4N + 40); the run
             fails if the top five levels ever hold more than tail_tol;
  * "exact": the same rotating frame with the exact S_z and S_y in the Dicke
             basis n = 0..N, i.e. unlinearized Holstein-Primakoff.
Central-spin expectations, purities, and QFIs are frame-independent; <a^dag a>
and the number distribution p_n are rotating-frame quantities.

Panels: central-spin Pauli expectations; QFI for J of the bath, the central
spin, and the global state; the longitudinal field h_z = J B(t) on the central
spin, mean +- standard deviation, beside the Landau-Zener toy field
J N cos(2 omega t); the bath excitation fraction <a^dag a>/N and purity; and the
number distributions p_n(t) of both models.  Dashed guides mark the mean-field
crossings cos(2 omega t) = 0.

Run from the repository root:
    python CRB/HolsteinPrimakoff/hp_dynamics.py
    python CRB/HolsteinPrimakoff/hp_dynamics.py --N 30 --Omega 2.75 --t-pi 0.75 --no-show

The figure and its complete numerical .plot.json record are saved together by
CRB.smart_save under PhD/Graphs/holstein_primakoff/dynamics_vs_time.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp

# Support both direct execution and `python -m CRB.HolsteinPrimakoff.hp_dynamics`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CRB.crb_core import central_spin_state, qfi_from_rho_and_drho
from CRB.smart_save import PlotRecord, save_plot

MODELS = ("HP", "exact")
MODEL_LABELS = {"HP": "HP (linearized)", "exact": "exact"}
MODEL_STYLES = {"HP": {"ls": "-", "lw": 1.6}, "exact": {"ls": "--", "lw": 1.3}}
PAULI = np.array([qt.sigmax().full(), qt.sigmay().full(), qt.sigmaz().full()])
Z_SIGNS = np.array([[1.0], [-1.0]])  # Z on the central index of a (2, d) amplitude array.
TAIL_LEVELS = 5


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Physics, sampling, solver, and figure settings for one run.

    Omega, J, and omega use a common inverse-time unit; N is the number of bath
    spins. Evolution starts at zero and ends at ``t_pi * pi``. ``n_max`` is the
    HP boson cutoff (None gives 4N + 40). ``size`` is in inches.
    """

    N: int = 15
    Omega: float = 2.0
    omega: float = 1.0
    J: float = 1.0
    initial: str = "+"
    t_pi: float = 1.0
    points: int = 2001
    n_max: int | None = None
    tail_tol: float = 1e-8
    rtol: float = 1e-9
    atol: float = 1e-11
    method: str = "DOP853"
    qfi_tol: float = 1e-10
    fock_floor: float = 1e-4
    size: tuple[float, float] = (12.0, 11.0)
    dpi: int = 160
    show: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.N, int) or self.N < 1:
            raise ValueError("N must be a positive integer.")
        if not np.all(np.isfinite([self.Omega, self.J])):
            raise ValueError("Omega and J must be finite.")
        if not np.isfinite(self.omega) or self.omega < 0:
            raise ValueError("omega must be finite and non-negative (0 gives a static bath).")
        if self.initial not in {"0", "+"}:
            raise ValueError("initial must be '0' or '+'.")
        for name in ("t_pi", "tail_tol", "rtol", "atol", "qfi_tol", "fock_floor"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not isinstance(self.points, int) or self.points < 2:
            raise ValueError("points must be an integer of at least 2.")
        if self.n_max is not None and (not isinstance(self.n_max, int) or self.n_max <= TAIL_LEVELS):
            raise ValueError(f"n_max must be None or an integer above {TAIL_LEVELS}.")
        if self.method not in {"DOP853", "RK45", "RK23"}:
            raise ValueError("method must be DOP853, RK45, or RK23.")
        if self.qfi_tol >= 1 or self.fock_floor >= 1:
            raise ValueError("qfi_tol and fock_floor must be smaller than 1.")
        if len(self.size) != 2 or not all(np.isfinite(v) and v > 0 for v in self.size):
            raise ValueError("size must contain two finite positive dimensions.")
        if not isinstance(self.dpi, int) or self.dpi <= 0:
            raise ValueError("dpi must be a positive integer.")

    @property
    def boson_cutoff(self) -> int:
        return 4 * self.N + 40 if self.n_max is None else self.n_max

    @property
    def central_theta(self) -> float:
        return 0.0 if self.initial == "0" else np.pi / 2.0


@dataclass(frozen=True, slots=True)
class BathOperators:
    """Rotating-frame bath coupling B(t) = B_c cos(2 omega t) + B_s sin(2 omega t), n basis."""

    B_c: np.ndarray  # Diagonal of S_z = N - 2n.
    B_s: np.ndarray  # S_y, exact or linearized.


def bath_operators(cfg: SimulationConfig, model: str) -> BathOperators:
    """Return S_z and S_y in the basis of n flipped spins, n = 0 being S_z = N."""
    if model == "exact":
        spin = cfg.N / 2.0  # qt.jmat orders m = N/2, ..., -N/2, i.e. n = 0, ..., N.
        return BathOperators(2.0 * np.real(np.diag(qt.jmat(spin, "z").full())),
                             2.0 * qt.jmat(spin, "y").full())
    if model == "HP":
        n = np.arange(cfg.boson_cutoff + 1)
        a = np.diag(np.sqrt(n[1:]), 1).astype(complex)
        return BathOperators(cfg.N - 2.0 * n, np.sqrt(cfg.N) * 1j * (a.conj().T - a))
    raise ValueError(f"model must be one of {MODELS}.")


@dataclass(frozen=True, slots=True)
class Trajectory:
    model: str
    time: np.ndarray
    states: np.ndarray  # Shape (points, 2, d): central index, then bath level n.
    tangents: np.ndarray  # d_J of states, same shape.
    bloch: np.ndarray  # Shape (points, 3), ordered X, Y, Z.
    qfi_bath: np.ndarray
    qfi_central: np.ndarray
    qfi_global: np.ndarray
    purity: np.ndarray  # Tr rho_c^2 = Tr rho_B^2.
    excitation: np.ndarray  # <a^dag a>/N in the rotating frame.
    field_mean: np.ndarray  # <h_z> with h_z = J B(t).
    field_std: np.ndarray
    populations: np.ndarray  # Shape (points, d): p_n.
    norm_error: float
    tail_population: float  # Largest population in the top TAIL_LEVELS levels; a cutoff check for HP only.


def simulate(cfg: SimulationConfig, model: str) -> Trajectory:
    """Evolve central_spin_state(theta) |n = 0> and its J-derivative in the rotating frame."""
    ops = bath_operators(cfg, model)
    dim = ops.B_c.size
    time = np.linspace(0.0, cfg.t_pi * np.pi, cfg.points)
    initial = np.zeros((2, dim), dtype=complex)
    initial[:, 0] = central_spin_state(cfg.central_theta).full().ravel()

    def zb(amplitudes: np.ndarray, t: float) -> np.ndarray:
        """Apply Z B(t) to a (2, d) amplitude array."""
        phase = 2.0 * cfg.omega * t
        coupled = np.cos(phase) * ops.B_c * amplitudes + np.sin(phase) * amplitudes @ ops.B_s.T
        return Z_SIGNS * coupled

    def rhs(t: float, augmented: np.ndarray) -> np.ndarray:
        state, tangent = augmented.reshape(2, 2, dim)
        zb_state = zb(state, t)
        h_state = cfg.Omega * state[::-1] + cfg.J * zb_state
        h_tangent = cfg.Omega * tangent[::-1] + cfg.J * zb(tangent, t)
        return np.concatenate(((-1j * h_state).ravel(), (-1j * (h_tangent + zb_state)).ravel()))

    solution = solve_ivp(
        rhs,
        (time[0], time[-1]),
        np.concatenate((initial.ravel(), np.zeros(2 * dim, dtype=complex))),
        t_eval=time,
        method=cfg.method,
        rtol=cfg.rtol,
        atol=cfg.atol,
    )
    if not solution.success:
        raise RuntimeError(f"Schrodinger integration failed ({model}): {solution.message}")

    raw_states = solution.y[:2 * dim].T
    raw_tangents = solution.y[2 * dim:].T
    norms = np.linalg.norm(raw_states, axis=1)
    norm_error = float(np.max(np.abs(norms**2 - 1.0)))
    flat_states = raw_states / norms[:, None]
    # Differentiate the numerical normalization as well as the raw state.
    norm_derivative = np.einsum("ti,ti->t", flat_states.conj(), raw_tangents).real
    flat_tangents = (raw_tangents - flat_states * norm_derivative[:, None]) / norms[:, None]
    states = flat_states.reshape(-1, 2, dim)
    tangents = flat_tangents.reshape(-1, 2, dim)

    populations = np.sum(np.abs(states)**2, axis=1)
    tail_population = float(np.max(np.sum(populations[:, -TAIL_LEVELS:], axis=1)))
    if model == "HP" and tail_population > cfg.tail_tol:
        raise RuntimeError(
            f"HP truncation at n_max={cfg.boson_cutoff} holds population {tail_population:.2e} "
            f"in its top {TAIL_LEVELS} levels (tail_tol={cfg.tail_tol:g}); increase --n-max."
        )

    rho_c = np.einsum("tik,tjk->tij", states, states.conj())
    drho_c = np.einsum("tik,tjk->tij", tangents, states.conj())
    drho_c = drho_c + drho_c.conj().transpose(0, 2, 1)
    bloch = np.einsum("tij,aji->ta", rho_c, PAULI).real
    purity = np.einsum("tij,tji->t", rho_c, rho_c).real
    overlap = np.einsum("tik,tik->t", states.conj(), tangents)
    qfi_global = 4.0 * (np.sum(np.abs(tangents)**2, axis=(1, 2)) - np.abs(overlap)**2)
    qfi_central = np.empty(cfg.points)
    qfi_bath = np.empty(cfg.points)
    # Bath matrices are (d, d), so build them one time at a time.
    for index, (state, tangent) in enumerate(zip(states, tangents)):
        qfi_central[index], _ = qfi_from_rho_and_drho(rho_c[index], drho_c[index], tol=cfg.qfi_tol)
        drho_b = tangent.T @ state.conj()
        qfi_bath[index], _ = qfi_from_rho_and_drho(state.T @ state.conj(), drho_b + drho_b.conj().T,
                                                   tol=cfg.qfi_tol)
    excitation = populations @ np.arange(dim) / cfg.N

    phase = 2.0 * cfg.omega * time
    b_states = (np.cos(phase)[:, None, None] * ops.B_c * states
                + np.sin(phase)[:, None, None] * states @ ops.B_s.T)
    b_mean = np.einsum("tik,tik->t", states.conj(), b_states).real
    b_square = np.sum(np.abs(b_states)**2, axis=(1, 2))
    field_mean = cfg.J * b_mean
    field_std = abs(cfg.J) * np.sqrt(np.clip(b_square - b_mean**2, 0.0, None))

    return Trajectory(model, time, states, tangents, bloch, qfi_bath, qfi_central, qfi_global,
                      purity, excitation, field_mean, field_std, populations, norm_error, tail_population)


def crossing_times(cfg: SimulationConfig) -> np.ndarray:
    """Zeros of cos(2 omega t) in [0, t_pi * pi], where the mean-field detuning vanishes."""
    if cfg.omega == 0:
        return np.empty(0)
    count = int(np.floor(4.0 * cfg.omega * cfg.t_pi - 1.0) // 2) + 1
    return (2 * np.arange(max(count, 0)) + 1) * np.pi / (4.0 * cfg.omega)


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
        return re.sub(r'[<>:"/\\|?*\s]+', "-", str(value).replace("+", "plus"))

    parameters = "__".join(f"{field.name}={tag(getattr(cfg, field.name))}" for field in fields(cfg))
    return f"hp-dynamics__{parameters}.png"


def model_legend(axis: plt.Axes, colored: list[tuple[str, str]], **kwargs: object) -> None:
    """Legend with one entry per colored quantity and one per model line style."""
    handles = [Line2D([], [], color=color, lw=1.6, label=label) for label, color in colored]
    handles += [Line2D([], [], color="0.3", label=MODEL_LABELS[m], **MODEL_STYLES[m]) for m in MODELS]
    axis.legend(handles=handles, **kwargs)


def plot_trajectories(cfg: SimulationConfig, trajectories: dict[str, Trajectory]) -> PlotRecord:
    """Plot the central spin, QFIs, field, bath excitation, and number distributions."""
    figure, axes = plt.subplot_mosaic(
        [["spin", "qfi"], ["field", "bath"], ["fock_HP", "fock_exact"]],
        figsize=cfg.size, sharex=True, layout="constrained",
    )
    time = trajectories["HP"].time
    guides = crossing_times(cfg)
    state_label = rf"|{cfg.initial}\rangle|S_z=N\rangle"

    spin_axis = axes["spin"]
    for model, result in trajectories.items():
        for component in range(3):
            spin_axis.plot(time, result.bloch[:, component], color=f"C{component}", **MODEL_STYLES[model])
    spin_axis.set(title=rf"Central spin — Pauli expectations, ${state_label}$",
                  ylabel="Expectation value", ylim=(-1.08, 1.45))
    spin_axis.set_yticks(np.linspace(-1, 1, 5))  # headroom above for the legend
    model_legend(spin_axis, [(rf"$\langle {name}\rangle$", f"C{i}") for i, name in enumerate("XYZ")],
                 loc="upper center", ncol=5, fontsize=8)

    qfi_axis = axes["qfi"]
    qfi_curves = (("qfi_bath", r"$F_Q[\rho_B]$", "tab:purple"),
                  ("qfi_central", r"$F_Q[\rho_c]$", "tab:orange"),
                  ("qfi_global", r"$F_Q[|\psi\rangle]$", "0.15"))
    for model, result in trajectories.items():
        for attribute, _, color in qfi_curves:
            qfi_axis.plot(time, getattr(result, attribute), color=color, **MODEL_STYLES[model])
    top = max(result.qfi_global.max() for result in trajectories.values())
    qfi_axis.set(title=r"QFI for $J$ — bath, central spin, and global state",
                 ylabel=r"Fisher information for $J$", ylim=(0, 1.3 * top))
    model_legend(qfi_axis, [(label, color) for _, label, color in qfi_curves], loc="upper left", fontsize=9)

    # h_z = J B(t): its mean crosses zero at the guides, where its spread J sqrt<S_y^2> peaks.
    field_axis = axes["field"]
    toy_field = cfg.J * cfg.N * np.cos(2.0 * cfg.omega * time)
    for model, color in zip(MODELS, ("C0", "C3")):
        result = trajectories[model]
        field_axis.fill_between(time, result.field_mean - result.field_std, result.field_mean + result.field_std,
                                color=color, alpha=0.18, lw=0)
        field_axis.plot(time, result.field_mean, color=color,
                        label=rf"{MODEL_LABELS[model]}: $\langle h_z\rangle\pm\Delta h_z$", **MODEL_STYLES[model])
    field_axis.plot(time, toy_field, color="k", ls=":", lw=1.4, label=r"LZ toy $JN\cos 2\omega t$")
    for sign in (1, -1):
        field_axis.axhline(sign * cfg.Omega, color="tab:green", ls="-.", lw=1.0, alpha=0.8,
                           label=r"$\pm\Omega$" if sign > 0 else None)
    field_axis.set(title=r"Field on the central spin, $h_z=J[S_z\cos 2\omega t+S_y\sin 2\omega t]$",
                   ylabel="Field")
    field_axis.legend(loc="lower left", fontsize=8)

    bath_axis = axes["bath"]
    for model, result in trajectories.items():
        bath_axis.plot(time, result.excitation, color="C3", **MODEL_STYLES[model])
        bath_axis.plot(time, result.purity, color="C2", **MODEL_STYLES[model])
    # HP excitation above 1 (n > N on average) is unphysical; keep it in view as the breakdown signal.
    top = max(1.08, 1.05 * max(result.excitation.max() for result in trajectories.values()))
    bath_axis.set(title=r"Bath — excitation $\langle a^\dagger a\rangle/N$ (rotating frame) and purity",
                  ylabel="Fraction", ylim=(0, top))
    model_legend(bath_axis, [(r"$\langle a^\dagger a\rangle/N$", "C3"), (r"$\mathrm{Tr}\,\rho_B^2$", "C2")],
                 loc="upper left", ncol=2, fontsize=9)

    # Show the exact range n <= N plus any HP population above the floor.
    occupied = np.flatnonzero(trajectories["HP"].populations.max(axis=0) > cfg.fock_floor)
    n_show = int(max(cfg.N, occupied.max() if occupied.size else 0))
    norm = LogNorm(vmin=cfg.fock_floor, vmax=1.0)
    cmap = plt.get_cmap("magma").with_extremes(bad="0.85")
    for model in MODELS:
        axis = axes[f"fock_{model}"]
        grid = np.full((cfg.points, n_show + 1), np.nan)
        available = min(n_show + 1, trajectories[model].populations.shape[1])
        grid[:, :available] = np.clip(trajectories[model].populations[:, :available], cfg.fock_floor, 1.0)
        image = axis.imshow(grid.T, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap,
                            norm=norm, extent=(time[0], time[-1], -0.5, n_show + 0.5))
        axis.axhline(cfg.N + 0.5, color="w", ls=":", lw=1.0)
        axis.set(title=rf"{MODEL_LABELS[model]} — bath number distribution $p_n(t)$",
                 ylabel=r"Flipped bath spins $n$")
    figure.colorbar(image, ax=[axes["fock_HP"], axes["fock_exact"]], label=r"$p_n$", shrink=0.9)

    for name, axis in axes.items():
        axis.set_xlabel(r"Time $t$")
        axis.tick_params(labelbottom=True)  # sharex hides these on the upper rows
        on_map = name.startswith("fock")
        if not on_map:
            axis.grid(True, axis="y", alpha=0.25)
        for guide_time in guides:
            axis.axvline(guide_time, color="w" if on_map else "0.4", linestyle="--",
                         linewidth=0.9, alpha=0.65, zorder=0 if not on_map else 2)
        axis.set_xlim(time[0], time[-1])
    figure.suptitle(
        rf"$H_I(t)=\Omega X+JZ[S_z\cos 2\omega t+S_y\sin 2\omega t]$; HP: $S_z=N-2a^\dagger a$, "
        rf"$S_y\simeq\sqrt{{N}}P$; $\Omega={cfg.Omega:g}$, $\omega={cfg.omega:g}$, $N={cfg.N}$, "
        rf"$J={cfg.J:g}$, $n_{{\max}}={cfg.boson_cutoff}$"
    )

    data = {}
    for model, result in trajectories.items():
        for component, name in enumerate("XYZ"):
            data[f"{model}: <{name}>"] = (time, result.bloch[:, component])
        data[f"{model}: QFI(J) bath"] = (time, result.qfi_bath)
        data[f"{model}: QFI(J) central"] = (time, result.qfi_central)
        data[f"{model}: QFI(J) global"] = (time, result.qfi_global)
        data[f"{model}: <h_z>"] = (time, result.field_mean)
        data[f"{model}: std h_z"] = (time, result.field_std)
        data[f"{model}: <a^dag a>/N"] = (time, result.excitation)
        data[f"{model}: purity"] = (time, result.purity)
    data["LZ toy: J N cos(2 omega t)"] = (time, toy_field)
    for model, result in trajectories.items():
        for level in range(min(n_show + 1, result.populations.shape[1])):
            data[f"{model}: p_n={level}"] = (time, result.populations[:, level])

    try:
        record = save_plot(
            figure,
            system="holstein_primakoff",
            plot_type="dynamics_vs_time",
            params=asdict(cfg),
            data=data,
            name=Path(figure_filename(cfg)).stem,
            xlabel=r"Time $t$",
            ylabel="Expectation value / Fisher information / field / population",
            xunit="dimensionless",
            notes=(
                "Rotating frame of the bath drive: H_I = Omega X + J Z [S_z cos(2 omega t) + S_y sin(2 omega t)], "
                "Pauli convention S_a = sum sigma_a, initial |initial>|S_z = N> (HP vacuum). "
                "HP: S_z = N - 2 a^dag a, S_y ~ sqrt(N) i(a^dag - a), truncated at n_max. "
                "exact: Dicke basis n = 0..N. Solid HP, dashed exact. "
                "Rows: central Pauli expectations | QFI(J) of bath, central spin, global state; "
                "h_z = J B(t) mean +- std with the LZ toy field J N cos(2 omega t) | <a^dag a>/N and purity; "
                "p_n(t) for HP | exact. Guides at cos(2 omega t) = 0."
            ),
            metadata={
                "boson_cutoff": cfg.boson_cutoff,
                "crossing_times": guides.tolist(),
                "maximum_norm_errors": {m: r.norm_error for m, r in trajectories.items()},
                "HP_maximum_tail_population": trajectories["HP"].tail_population,
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
    parser.add_argument("--N", type=int, default=defaults.N, help="Number of bath spins.")
    parser.add_argument("--Omega", type=float, default=defaults.Omega, help="Central-spin drive.")
    parser.add_argument("--omega", "--w", dest="omega", type=float, default=defaults.omega,
                        help=f"Bath drive in omega S_x (default: {defaults.omega:g}).")
    parser.add_argument("--J", type=float, default=defaults.J)
    parser.add_argument("--initial", choices=("0", "+"), default=defaults.initial,
                        help="Central-spin initial state; the bath starts at S_z = N.")
    parser.add_argument("--t-pi", type=float, default=defaults.t_pi,
                        help=f"Final time in multiples of pi (default: {defaults.t_pi:g}).")
    parser.add_argument("--points", type=int, default=defaults.points)
    parser.add_argument("--n-max", type=int, default=defaults.n_max, help="HP boson cutoff (default: 4N + 40).")
    parser.add_argument("--tail-tol", type=float, default=defaults.tail_tol)
    parser.add_argument("--rtol", type=float, default=defaults.rtol)
    parser.add_argument("--atol", type=float, default=defaults.atol)
    parser.add_argument("--method", choices=("DOP853", "RK45", "RK23"), default=defaults.method)
    parser.add_argument("--qfi-tol", type=float, default=defaults.qfi_tol)
    parser.add_argument("--fock-floor", type=float, default=defaults.fock_floor,
                        help="Lowest population on the logarithmic p_n color scale.")
    parser.add_argument("--size", nargs=2, type=float, default=defaults.size, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int, default=defaults.dpi)
    parser.add_argument("--no-show", action="store_false", dest="show", default=defaults.show)
    arguments = vars(parser.parse_args())
    arguments["size"] = tuple(arguments["size"])
    cfg = SimulationConfig(**arguments)
    trajectories = {model: simulate(cfg, model) for model in MODELS}
    for model, result in trajectories.items():
        peak = int(np.argmax(result.qfi_bath))
        print(f"{model:>5}: max norm error = {result.norm_error:.3e}; max bath QFI(J) = "
              f"{result.qfi_bath[peak]:.6g} at t = {result.time[peak]:.4g}; final bath QFI(J) = "
              f"{result.qfi_bath[-1]:.6g}")
    print(f"   HP: n_max = {cfg.boson_cutoff}; top-{TAIL_LEVELS} tail population = "
          f"{trajectories['HP'].tail_population:.2e}")
    record = plot_trajectories(cfg, trajectories)
    print(f"Saved numerical data: {record.json_path}")
    print(f"Saved figure: {record.image_path}")


if __name__ == "__main__":
    main()
