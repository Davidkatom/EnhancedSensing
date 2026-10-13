"""Check one- and two-passage cat states of the linearized HP Hamiltonian.

H = Omega X_c + J Z_c [N cos(2 omega t) + sqrt(N) P sin(2 omega t)],
P = i(a^dagger-a), [a+a^dagger,P] = 2i, hbar = 1, initial |+>_c |vac>.
Evolution ends at pi/omega and also checks the half-period pi/(2 omega).
Pass --readout 1 to use t=1 for the first snapshot instead of the half-period.
The exact simulation reuses linearized_momentum.simulate_momentum: P is
conserved and vacuum momentum is N(0,1), so no Fock cutoff is needed.

For positive J, omega and Omega, the p=0 crossing slope is 2 J N omega:
  delta = Omega^2/(4 J N omega), P_LZ = exp(-2 pi delta),
  phi_S = pi/4 + Im logGamma(1-i delta) + delta (log(delta)-1).
Use delta log(delta)=0 at delta=0. This is the unshifted Stokes phase.
  Theta = integral_0^[pi/(2 omega)] sqrt(Omega^2+(J N cos(2 omega t))^2) dt,
  chi = Theta + phi_S, d = J sqrt(N)/omega, |v_d> = exp(-i d P)|vac>.
With endpoint eigenbasis rotations neglected and momentum-dependent phases
replaced by their constant/linear parts, the ideal half-period state is
  [|0>(sqrt(P_LZ)|v_d>-sqrt(1-P_LZ)e^(i chi)|vac>)
   +|1>(sqrt(1-P_LZ)e^(-i chi)|vac>+sqrt(P_LZ)|v_-d>)]/sqrt(2).
At the full period there are TWO crossings. Define
  A=P_LZ+(1-P_LZ)e^(-2i chi), B=-2i sqrt(P_LZ(1-P_LZ)) sin(chi).
The corresponding ideal two-passage state is
  [|0>(A|vac>+B|v_-d>)+|1>(B|v_d>+A*|vac>)]/sqrt(2).
Neither ideal formula is exact: p-dependent crossing times and accumulated
quadratic phases can deform the packets. The momentum-resolved adiabatic-
impulse approximation (AIA) below retains those effects and endpoint rotations.
It uses eigenvectors R_y(theta), theta=atan2(Omega,h_p), ordered (E_+,E_-):
  N_down = [[sqrt(1-P)e^(-i phi_S),sqrt(P)],
            [-sqrt(P),sqrt(1-P)e^(i phi_S)]], N_up=N_down.T.
These signs are also checked against a direct finite-time linear LZ sweep.
Reference: https://www.nature.com/articles/s41598-023-31084-y (Eq. 27).

Run from the repository root:
  python CRB/HolsteinPrimakoff/linearized_cat_state.py
  python CRB/HolsteinPrimakoff/linearized_cat_state.py --N 60 --Omega 4.4
Figures/records use CRB.smart_save, with complete parameter-named local copies
and the joint-state NPZ/summary JSON under graphs/linearized_cat_state/.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
import re
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipe, ellipeinc, loggamma, xlogy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CRB.HolsteinPrimakoff.linearized_momentum import MomentumConfig, MomentumTrajectory, simulate_momentum
from CRB.HolsteinPrimakoff.hp_dynamics import PAULI
from CRB.smart_save import save_plot


@dataclass(frozen=True, slots=True)
class CatConfig:
    """Frequencies in inverse reference-time units; figure dimensions in inches.

    nodes/check are vacuum Gaussian quadratures; span is the extra coherent-
    amplitude range around +/-d. samples controls the real-axis Husimi slice.
    Initial state is fixed; readout=0 selects the half-period snapshot.
    """

    N: int = 15
    Omega: float = 2.2
    J: float = 1.0
    omega: float = 1.0
    readout: float = 0.0  # Reference-time units; 0 selects pi/(2 omega).
    pmax: float = 9.0  # Momentum tail cutoff for boundary-split readout quadrature.
    points: int = 801
    nodes: int = 96
    check: int = 160
    rtol: float = 1e-10
    atol: float = 1e-12
    method: str = "DOP853"
    tol: float = 1e-7
    span: float = 3.0
    samples: int = 401
    width: float = 12.0
    height: float = 8.0
    dpi: int = 150

    def __post_init__(self) -> None:
        if isinstance(self.N, bool) or not isinstance(self.N, int) or self.N < 1:
            raise ValueError("N must be a positive integer.")
        if not np.isfinite(self.Omega) or self.Omega < 0:
            raise ValueError("Omega must be finite and non-negative.")
        for name in ("J", "omega", "pmax", "rtol", "atol", "tol", "span", "width", "height"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("points", "nodes", "check", "samples", "dpi"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 2:
                raise ValueError(f"{name} must be an integer of at least 2.")
        if self.points < 3 or self.points % 2 != 1:
            raise ValueError("points must be odd and at least 3, so the half-period is sampled exactly.")
        if not self.nodes < self.check <= 256:
            raise ValueError("Require nodes < check <= 256; increase both to check convergence.")
        if self.method not in {"DOP853", "RK45", "RK23"}:
            raise ValueError("method must be DOP853, RK45, or RK23.")
        if not np.isfinite(self.readout) or (self.readout != 0 and not np.pi/(4*self.omega) < self.readout < 3*np.pi/(4*self.omega)):
            raise ValueError("readout must be 0 (half-period) or lie between the two mean-field crossings.")

    def momentum_config(self) -> MomentumConfig:
        return MomentumConfig(N=self.N, Omega=self.Omega, J=self.J, omega=self.omega,
                              tmax=np.pi / self.omega, points=self.points, p_nodes=self.nodes,
                              theta=np.pi / 2, phi=0., rtol=self.rtol, atol=self.atol, method=self.method)


def stokes_phase(delta: np.ndarray | float) -> np.ndarray:
    """Unshifted Stokes phase, continuous branch; delta=0 gives pi/4 exactly."""
    delta = np.asarray(delta, dtype=float)
    if np.any(~np.isfinite(delta)) or np.any(delta < 0):
        raise ValueError("delta must be finite and non-negative.")
    # loggamma avoids the principal-argument jumps of angle(gamma(...)).
    return np.pi / 4 + np.imag(loggamma(1 - 1j * delta)) + xlogy(delta, delta) - delta


def lz_parameters(cfg: CatConfig, p: np.ndarray | float = 0.) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = np.sqrt(cfg.N**2 + cfg.N * np.asarray(p)**2)
    delta = cfg.Omega**2 / (4 * cfg.J * cfg.omega * radius)
    return delta, np.exp(-2 * np.pi * delta), stokes_phase(delta)


def ideal_parameters(cfg: CatConfig) -> dict[str, float]:
    delta, probability, phi = lz_parameters(cfg)
    energy_max = np.hypot(cfg.Omega, cfg.J * cfg.N)
    theta = energy_max / cfg.omega * ellipe((cfg.J * cfg.N / energy_max)**2)
    return dict(delta=float(delta), P_LZ=float(probability), phi_S=float(phi), Theta=float(theta),
                chi=float(theta + phi), d=float(cfg.J * np.sqrt(cfg.N) / cfg.omega))


def ideal_cat(cfg: CatConfig, p: np.ndarray, passage: str, *, proposed: bool = False) -> np.ndarray:
    """Conditional momentum amplitudes; multiply by the vacuum wavefunction.

    proposed=True evaluates the user's one-passage expression with Theta-phi_S
    at either readout. The corrected ideal expressions use Theta+phi_S.
    """
    pars = ideal_parameters(cfg)
    probability, d = pars["P_LZ"], pars["d"]
    chi = pars["Theta"] - pars["phi_S"] if proposed else pars["chi"]
    a, b = np.sqrt(probability), np.sqrt(1 - probability)
    pointer = np.exp(-1j * d * p)
    if passage == "half" or proposed:
        return np.column_stack((a * pointer - b * np.exp(1j * chi),
                                b * np.exp(-1j * chi) + a * pointer.conj())) / np.sqrt(2)
    if passage != "full":
        raise ValueError("passage must be 'half' or 'full'.")
    A = probability + (1 - probability) * np.exp(-2j * chi)
    B = -2j * a * b * np.sin(chi)
    return np.column_stack((A + B * pointer.conj(), B * pointer + A.conjugate())) / np.sqrt(2)


def coherent_cat_at(cfg: CatConfig, p: np.ndarray, t: float, *, proposed: bool = False) -> np.ndarray:
    """Single-passage cat with time-dependent centres and subsequent Z phase.

    The no-flip packet has centre s=d(1-cos(2 omega t))/2. The flip packet
    has centre f=s-d; it reaches the vacuum centre only at the half-period.
    Endpoint mixing and momentum-dependent quadratic phases are omitted.
    """
    if not np.pi/(4*cfg.omega) < t < 3*np.pi/(4*cfg.omega):
        raise ValueError("Single-passage cat requires t between the mean-field crossings.")
    pars = ideal_parameters(cfg)
    a, b = np.sqrt(pars["P_LZ"]), np.sqrt(1-pars["P_LZ"])
    d = pars["d"]
    stay = d/2*(1-np.cos(2*cfg.omega*t))
    flip = stay-d
    chi = pars["Theta"]-pars["phi_S"] if proposed else pars["chi"]
    emax = np.hypot(cfg.Omega,cfg.J*cfg.N)
    m = (cfg.J*cfg.N/emax)**2
    eta = emax/(2*cfg.omega)*(ellipeinc(np.pi,m)-ellipeinc(2*cfg.omega*t,m))
    stay_pointer, flip_pointer = np.exp(-1j*stay*p), np.exp(-1j*flip*p)
    return np.column_stack((np.exp(-1j*eta)*(a*stay_pointer-b*np.exp(1j*chi)*flip_pointer),
                            np.exp(1j*eta)*(b*np.exp(-1j*chi)*flip_pointer.conj()+a*stay_pointer.conj())))/np.sqrt(2)


def readout_quadrature(cfg: CatConfig, t: float, nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the vacuum integral where a momentum component crosses at time t.

    The impulse approximation has a discontinuity at p=-sqrt(N) cot(2 omega t).
    Integrate the two smooth pieces separately with Gaussian Legendre rules.
    """
    angle = 2*cfg.omega*t
    boundaries = [-cfg.pmax,cfg.pmax]
    if abs(np.sin(angle)) > 1e-14:
        cutoff = -np.sqrt(cfg.N)*np.cos(angle)/np.sin(angle)
        if -cfg.pmax < cutoff < cfg.pmax:
            boundaries.insert(1,float(cutoff))
    x,w = np.polynomial.legendre.leggauss(nodes)
    momenta, weights = [],[]
    for left,right in zip(boundaries[:-1],boundaries[1:]):
        p = (left+right)/2+(right-left)/2*x
        momenta.append(p)
        weights.append((right-left)/2*w*np.exp(-p*p/2)/np.sqrt(2*np.pi))
    return np.concatenate(momenta),np.concatenate(weights)


def adiabatic_impulse(cfg: CatConfig, time: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Both p-shifted crossings, exact adiabatic phases, and endpoint rotations.

    For each p, h_p=J R_p cos(2 omega t-theta_p). E_max ellipeinc(u,m)
    is the energy primitive; integrating it avoids crossing-grid errors.
    """
    shift = np.arctan(p / np.sqrt(cfg.N))
    radius = np.sqrt(cfg.N**2 + cfg.N * p**2)
    emax = np.hypot(cfg.Omega, cfg.J * radius)
    m = (cfg.J * radius / emax)**2
    scale = emax / (2 * cfg.omega)
    start = ellipeinc(-shift, m)
    u = 2 * cfg.omega * time[:, None] - shift
    phase = scale * (ellipeinc(u, m) - start)
    first = scale * (ellipe(m) - start)
    second = first + 2 * scale * ellipe(m)
    tc1 = (np.pi / 2 + shift) / (2 * cfg.omega)
    tc2 = (3 * np.pi / 2 + shift) / (2 * cfg.omega)
    _, probability, phi = lz_parameters(cfg, p)
    a, b = np.sqrt(probability), np.sqrt(1 - probability)
    diagonal = b * np.exp(-1j * phi)

    theta0 = np.arctan2(cfg.Omega, cfg.J * cfg.N)
    c0, s0 = np.cos(theta0 / 2), np.sin(theta0 / 2)
    initial = np.broadcast_to(np.array([c0 + s0, c0 - s0]) / np.sqrt(2), (p.size, 2))

    def propagate(state: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        return state * np.exp(zeta[..., None] * np.array([-1j, 1j]))

    before = propagate(initial, first)
    after1 = np.column_stack((diagonal * before[:, 0] + a * before[:, 1],
                              -a * before[:, 0] + diagonal.conj() * before[:, 1]))
    middle = propagate(after1, second - first)
    after2 = np.column_stack((diagonal * middle[:, 0] - a * middle[:, 1],
                              a * middle[:, 0] + diagonal.conj() * middle[:, 1]))
    amplitudes = propagate(initial, phase)
    amplitudes = np.where((time[:, None] >= tc1)[..., None], propagate(after1, phase - first), amplitudes)
    amplitudes = np.where((time[:, None] >= tc2)[..., None], propagate(after2, phase - second), amplitudes)
    h = cfg.J * (cfg.N * np.cos(2 * cfg.omega * time[:, None])
                 + np.sqrt(cfg.N) * p * np.sin(2 * cfg.omega * time[:, None]))
    angle = np.arctan2(cfg.Omega, h) / 2
    c, s = np.cos(angle), np.sin(angle)
    return np.stack((c * amplitudes[..., 0] - s * amplitudes[..., 1],
                     s * amplitudes[..., 0] + c * amplitudes[..., 1]), axis=-1)


def fidelity(reference: np.ndarray, state: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Full joint-state fidelity, including bath phases and normalization."""
    overlap = np.einsum("q,...qi,...qi->...", weights, reference.conj(), state)
    nr = np.einsum("q,...qi,...qi->...", weights, reference.conj(), reference).real
    ns = np.einsum("q,...qi,...qi->...", weights, state.conj(), state).real
    return np.abs(overlap)**2 / (nr * ns)


def husimi_slice(state: np.ndarray, p: np.ndarray, weights: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Q_s(alpha)=|<v_alpha|psi_s>|^2/pi for real alpha, without postselection normalization."""
    return np.abs((np.exp(1j * alpha[:, None] * p) * weights) @ state)**2 / np.pi


def diagnostics(cfg: CatConfig, result: MomentumTrajectory, aia: np.ndarray, alpha: np.ndarray) -> tuple[dict, dict]:
    summary = ideal_parameters(cfg)
    summary.update(config=asdict(cfg), norm_error=result.norm_error,
                   crossing_times=[float(np.pi / (4 * cfg.omega)), float(3 * np.pi / (4 * cfg.omega))],
                   endpoint_mixing=float(cfg.Omega / (cfg.J * cfg.N)))
    snapshots = {}
    first_name = "readout" if cfg.readout else "half"
    for name, index in ((first_name, cfg.points // 2), ("full", -1)):
        momenta,weights = result.momenta,result.weights
        if name == "readout":
            momenta,weights = readout_quadrature(cfg,cfg.readout,result.momenta.size)
            readout_result = simulate_momentum(replace(cfg.momentum_config(),tmax=cfg.readout,points=3),quadrature=(momenta,weights))
            exact = readout_result.states[-1]
            approx = adiabatic_impulse(cfg,np.array([cfg.readout]),momenta)[0]
            ideal = coherent_cat_at(cfg,momenta,cfg.readout)
            proposed = coherent_cat_at(cfg,momenta,cfg.readout,proposed=True)
            readout_time = cfg.readout
            summary['norm_error'] = max(summary['norm_error'],readout_result.norm_error)
        else:
            exact = result.states[index]
            approx = aia[index]
            ideal = ideal_cat(cfg, result.momenta, name)
            proposed = ideal_cat(cfg, result.momenta, name, proposed=True)
            readout_time = float(result.time[index])
        snapshots[name] = dict(exact=husimi_slice(exact, momenta, weights, alpha),
                               aia=husimi_slice(approx, momenta, weights, alpha),
                               ideal=husimi_slice(ideal, momenta, weights, alpha))
        summary[name] = dict(time=float(readout_time),
                            aia_fidelity=float(fidelity(approx, exact, weights)),
                            ideal_fidelity=float(fidelity(ideal, exact, weights)),
                            proposed_fidelity=float(fidelity(proposed, exact, weights)))
    return summary, snapshots


def parameter_stem(cfg: CatConfig) -> str:
    def tag(value: object) -> str:
        if isinstance(value, float):
            value = str(int(value)) if value.is_integer() else repr(value)
        return re.sub(r'[<>:"/\\|?*\s]+', "-", re.sub(r"e([+-])0+(\d+)", r"e\1\2", str(value)))
    return "cat__" + "__".join(f"{f.name}={tag(getattr(cfg, f.name))}" for f in fields(cfg))


def save_results(cfg: CatConfig, result: MomentumTrajectory, aia: np.ndarray,
                 alpha: np.ndarray, summary: dict, snapshots: dict) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(cfg.width, cfg.height), layout="constrained")
    scaled_time = result.time * cfg.omega / np.pi
    rho_aia = np.einsum("q,tqi,tqj->tij", result.weights, aia, aia.conj())
    bloch_aia = np.einsum("tij,aji->ta", rho_aia, PAULI).real
    for i, label in enumerate("XYZ"):
        axes[0, 0].plot(scaled_time, result.bloch[:, i], color=f"C{i}", label=f"Exact <{label}>")
        axes[0, 0].plot(scaled_time, bloch_aia[:, i], color=f"C{i}", ls="--", alpha=.65)
    axes[0, 0].set(title="Central spin (dashed: momentum AIA)", ylabel="Pauli expectation", ylim=(-1.08, 1.08))
    axes[0, 0].legend(ncol=3, fontsize=9)
    joint_fidelity = fidelity(aia, result.states, result.weights)
    first_name = "readout" if cfg.readout else "half"
    first_scaled_time = cfg.readout*cfg.omega/np.pi if cfg.readout else .5
    axes[0, 1].plot(scaled_time, joint_fidelity, color="C4", label="Momentum AIA")
    axes[0, 1].scatter([first_scaled_time, 1], [summary[x]["ideal_fidelity"] for x in (first_name, "full")],
                       marker="x", s=65, color="black", label="Ideal cat at readout", zorder=4)
    axes[0, 1].set(title="Joint-state fidelity with numerical evolution", ylabel="Fidelity", ylim=(-.02, 1.04), xlim=(-.01,1.01))
    axes[0, 1].legend(fontsize=9, loc="lower left")
    for ax in axes[0]:
        ax.set(xlabel=r"$\omega t/\pi$", xlim=(-.01, 1.01))
        for crossing in (.25, .75):
            ax.axvline(crossing, color=".5", ls=":", lw=1)
        ax.axvline(first_scaled_time, color=".65", ls="--", lw=.8)
    data = {f"Exact <{label}>": (result.time, result.bloch[:, i]) for i, label in enumerate("XYZ")}
    data.update({f"AIA <{label}>": (result.time, bloch_aia[:, i]) for i, label in enumerate("XYZ")})
    data["AIA joint fidelity"] = (result.time, joint_fidelity)
    for ax, name in zip(axes[1], (first_name, "full")):
        for model, style in (("exact", "-"), ("aia", "--"), ("ideal", ":")):
            for spin in range(2):
                values = snapshots[name][model][:, spin]
                ax.plot(alpha, values, color=f"C{spin}", ls=style, lw=1.8 if model == "exact" else 1.2,
                        label=f"{model.upper()} |{spin}>")
                data[f"{name} {model} Q_{spin}(alpha real)"] = (alpha, values)
        if name == "readout":
            stay = summary["d"]/2*(1-np.cos(2*cfg.omega*cfg.readout))
            centres = (stay,stay-summary["d"],-stay,summary["d"]-stay)
        else:
            centres = (-summary["d"],0.,summary["d"])
        for center in centres:
            ax.axvline(center, color=".8", lw=.7, zorder=0)
        label = f"t={cfg.readout:g}" if name == "readout" else f"{name.title()} period"
        ax.set(xlabel=r"Real coherent amplitude $\alpha$", ylabel=r"Joint Husimi slice $Q_s(\alpha)$",
               xlim=(alpha[0], alpha[-1]), ylim=(0, None),
               title=f"{label}: ideal F={summary[name]['ideal_fidelity']:.4f}, AIA F={summary[name]['aia_fidelity']:.4f}")
        ax.legend(ncol=3, fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.suptitle("Linearized HP cat-state check: initial vacuum and central |+>\n"
                 f"N={cfg.N}, Omega={cfg.Omega:g}, J={cfg.J:g}, omega={cfg.omega:g}; "
                 f"P_LZ={summary['P_LZ']:.5f}, Stokes phase={summary['phi_S']:.5f} rad", fontsize=12)
    stem = parameter_stem(cfg)
    try:
        # Legacy smart_save preserves the entire parameter-rich image name;
        # explorer mode truncates names, so use it for the numerical record only.
        image_path = save_plot(fig, filename=stem + ".png", metadata=summary, dpi=cfg.dpi)
        record = save_plot(fig, system=Path(__file__).stem, plot_type="cat_check",
                           params=asdict(cfg), data=data, name="cat_state", script_path=__file__, save_image=False,
                           xlabel="Time or real coherent amplitude (see series)", ylabel="Spin, fidelity, or Husimi Q (see series)",
                           notes="P=i(a^dagger-a); vacuum variance=1; Pauli convention. Both crossings retained. "
                           "AIA impulses need not agree within a crossing. Husimi slices are not normalized 1D densities. "
                           "phi_S is unshifted, chi=Theta+phi_S. Reference: https://www.nature.com/articles/s41598-023-31084-y",
                           metadata=summary, dpi=cfg.dpi)
    finally:
        plt.close(fig)
    destination = REPO_ROOT / "graphs" / Path(__file__).stem
    destination.mkdir(parents=True, exist_ok=True)
    for source, extension in ((image_path, ".png"), (record.json_path, ".plot.json")):
        if source is not None:
            shutil.copy2(source, destination / (stem + extension))
    (destination / (stem + ".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(destination / (stem + ".npz"), time=result.time, momenta=result.momenta,
                        weights=result.weights, states=result.states, aia_states=aia, alpha=alpha)
    return destination / (stem + ".png")


def main(argv: list[str] | None = None) -> None:
    defaults = CatConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(defaults):
        value = getattr(defaults, f.name)
        parser.add_argument("--" + f.name, type=type(value), default=value)
    cfg = CatConfig(**vars(parser.parse_args(argv)))
    result = simulate_momentum(cfg.momentum_config())
    aia = adiabatic_impulse(cfg, result.time, result.momenta)
    d = ideal_parameters(cfg)["d"]
    alpha = np.linspace(-d - cfg.span, d + cfg.span, cfg.samples)
    summary, snapshots = diagnostics(cfg, result, aia, alpha)
    checked = simulate_momentum(replace(cfg.momentum_config(), p_nodes=cfg.check))
    checked_aia = adiabatic_impulse(cfg, checked.time, checked.momenta)
    checked_summary, checked_snapshots = diagnostics(cfg, checked, checked_aia, alpha)
    errors = [float(np.max(np.abs(result.bloch - checked.bloch)))]
    first_name = "readout" if cfg.readout else "half"
    for name in (first_name, "full"):
        errors.extend(abs(summary[name][k] - checked_summary[name][k]) for k in ("aia_fidelity", "ideal_fidelity", "proposed_fidelity"))
        errors.extend(float(np.max(np.abs(snapshots[name][k] - checked_snapshots[name][k]))) for k in ("exact", "aia", "ideal"))
    summary["quadrature_error"] = max(errors)
    summary["checked_norm_error"] = checked.norm_error
    if max(errors) > cfg.tol:
        raise RuntimeError(f"Quadrature difference {max(errors):.3e} exceeds tol={cfg.tol:g}; increase --nodes and --check.")
    if max(summary['norm_error'],checked_summary['norm_error']) > cfg.tol:
        raise RuntimeError("Numerical norm error exceeds tol; tighten --rtol and --atol.")
    summary["full_ideal_flip_probability"] = float(4 * summary["P_LZ"] * (1 - summary["P_LZ"]) * np.sin(summary["chi"])**2)
    for key in ("delta", "P_LZ", "phi_S", "Theta", "chi", "d", "quadrature_error", "norm_error"):
        print(f"{key} = {summary[key]:.10g}")
    for name in (first_name, "full"):
        print(f"{name}: {summary[name]}")
    image = save_results(cfg, result, aia, alpha, summary, snapshots)
    print(f"LOCAL_IMAGE={image}")


if __name__ == "__main__":
    main()
