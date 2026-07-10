import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from dataclasses import dataclass


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    # Bath size in the collective Dicke basis.
    # Hilbert dimension = 2 * (N + 1), so N can be much larger
    # than in the full individual-spin simulation.
    N: int = 10

    # Parameter to estimate
    J0: float = 1.0
    dJ: float = 1e-3

    # Central-spin decoherence channels.
    # beta: x-dephasing / bit-flip-like central noise
    # gamma: z-dephasing central noise
    beta: float = 0.0
    gamma: float = 0.0

    # Total wall-clock time
    T_max: float = 20.0

    # Floquet scan
    # pulse_angle phi means R_x(phi)=exp(-i phi sigma_x / 2)
    n_pulse_angles: int = 17
    pulse_angle_min: float = 0.0
    pulse_angle_max: float = 2.0 * np.pi

    n_taus: int = 25
    tau_min: float = 0.02
    tau_max: float = 1.5

    # Pulse-time accounting
    # If include_pulse_time = True, pulse duration is counted as real time.
    # A pulse R_x(phi) takes time phi / Omega_max.
    include_pulse_time: bool = True
    Omega_max: float = 50.0

    # If pulses are finite-time, decoherence acts during pulses too.
    # If pulses are instantaneous, pulse time is zero and no decoherence occurs during pulse.
    # For fair metrology, include_pulse_time=True is safer.

    # QFI numerics
    qfi_tol: float = 1e-12
    eps: float = 1e-15

    # Plot/output
    output_figure: str = "floquet_vs_baseline_bath_qfi_rate.png"


# ============================================================
# Operators and states
# ============================================================

def build_collective_system(cfg: Config):
    """
    Central spin + symmetric collective bath spin.

    Bath:
        S = N/2
        S_z = sum_i sigma_z_i = 2 J_z

    Full Hilbert space:
        central spin dimension 2
        bath dimension N+1
    """
    S = cfg.N / 2.0
    dim_bath = int(2 * S + 1)

    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()
    si = qt.qeye(2)

    Jx = 2.0 * qt.jmat(S, "x")
    Jy = 2.0 * qt.jmat(S, "y")
    Jz = 2.0 * qt.jmat(S, "z")
    Ib = qt.qeye(dim_bath)

    sx_c = qt.tensor(sx, Ib)
    sy_c = qt.tensor(sy, Ib)
    sz_c = qt.tensor(sz, Ib)

    Sx_b = qt.tensor(si, Jx)
    Sy_b = qt.tensor(si, Jy)
    Sz_b = qt.tensor(si, Jz)

    I_full = qt.tensor(si, Ib)

    # Initial state: central |+x>, bath coherent along +x
    plus_c = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    plus_b = qt.spin_coherent(S, np.pi / 2.0, 0.0)
    psi0 = qt.tensor(plus_c, plus_b)
    rho0 = psi0.proj()

    return {
        "S": S,
        "dim_bath": dim_bath,
        "Ib": Ib,
        "I_full": I_full,
        "sx_c": sx_c,
        "sy_c": sy_c,
        "sz_c": sz_c,
        "Sx_b": Sx_b,
        "Sy_b": Sy_b,
        "Sz_b": Sz_b,
        "rho0": rho0,
    }


def build_hz(J: float, ops: dict) -> qt.Qobj:
    """
    Signal Hamiltonian:

        H_z = J sigma_z^c S_z

    where S_z = sum_i sigma_z_i.
    """
    return J * ops["sz_c"] * ops["Sz_b"]


def build_hx(Omega: float, ops: dict) -> qt.Qobj:
    """
    Central-spin x drive:

        H_x = (Omega / 2) sigma_x

    so that evolution for time t gives a physical rotation angle Omega*t.
    """
    return 0.5 * Omega * ops["sx_c"] + ops["sz_c"] * ops["Sz_b"]


def build_c_ops(cfg: Config, ops: dict):
    """
    Unobserved central-spin decoherence.
    """
    c_ops = []

    if cfg.beta > 0:
        c_ops.append(np.sqrt(cfg.beta) * ops["sx_c"])

    if cfg.gamma > 0:
        c_ops.append(np.sqrt(cfg.gamma) * ops["sz_c"])

    return c_ops


# ============================================================
# Liouville evolution helpers
# ============================================================

def sanitize_rho(rho: qt.Qobj, eps: float = 1e-15) -> qt.Qobj:
    """
    Hermitize and renormalize a density matrix.
    """
    rho = 0.5 * (rho + rho.dag())
    tr = np.real(rho.tr())

    if abs(tr) < eps:
        raise RuntimeError("Trace vanished during evolution.")

    return rho / tr


def superprop_from_hamiltonian(H: qt.Qobj, dt: float, c_ops: list) -> qt.Qobj:
    """
    Return exp(L dt), where L is the Lindblad Liouvillian.
    """
    if dt == 0:
        dim = H.shape[0]
        I = qt.qeye(dim)
        I.dims = H.dims
        return qt.sprepost(I, I)

    L = qt.liouvillian(H, c_ops)
    return (L * dt).expm()


def unitary_superprop(U: qt.Qobj) -> qt.Qobj:
    """
    Superoperator rho -> U rho U^\dagger.
    """
    return qt.sprepost(U, U.dag())


def apply_superprop(rho: qt.Qobj, Sprop: qt.Qobj, cfg: Config) -> qt.Qobj:
    """
    Apply a superoperator to a density matrix.
    """
    vec = qt.operator_to_vector(rho)
    vec_new = Sprop * vec
    rho_new = qt.vector_to_operator(vec_new)
    rho_new.dims = rho.dims
    return sanitize_rho(rho_new, cfg.eps)


def central_rotation_superprop(angle: float, cfg: Config, ops: dict):
    """
    Build the superoperator for R_x(angle)=exp(-i angle sigma_x / 2).

    If include_pulse_time is True:
        simulate a finite pulse with H = Omega_max sigma_x / 2
        for duration angle/Omega_max, and include decoherence during the pulse.

    If include_pulse_time is False:
        treat the pulse as instantaneous.
    """
    if abs(angle) < 1e-15:
        return unitary_superprop(ops["I_full"]), 0.0

    if not cfg.include_pulse_time:
        U = (-1j * 0.5 * angle * ops["sx_c"]).expm()
        return unitary_superprop(U), 0.0

    sign = 1.0 if angle >= 0 else -1.0
    duration = abs(angle) / cfg.Omega_max

    Hx = build_hx(sign * cfg.Omega_max, ops)
    c_ops = build_c_ops(cfg, ops)

    return superprop_from_hamiltonian(Hx, duration, c_ops), duration


def free_evolution_superprop(J: float, duration: float, cfg: Config, ops: dict):
    """
    Free evolution under H_z = J sigma_z S_z for a given duration.
    """
    H = build_hz(J, ops)
    c_ops = build_c_ops(cfg, ops)
    return superprop_from_hamiltonian(H, duration, c_ops)


# ============================================================
# QFI
# ============================================================

def qfi_from_rho_and_drho(rho: np.ndarray, drho: np.ndarray, tol: float = 1e-12) -> float:
    r"""
    Mixed-state QFI:

        F_Q = 2 sum_{m,n} |<m| d rho |n>|^2 / (lambda_m + lambda_n)

    where rho = sum_n lambda_n |n><n|.
    """
    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)

    evals, evecs = np.linalg.eigh(rho)
    evals = np.real(evals)

    evals[np.abs(evals) < tol] = 0.0

    qfi = 0.0
    dim = len(evals)

    for m in range(dim):
        vm = evecs[:, m]
        for n in range(dim):
            denom = evals[m] + evals[n]
            if denom > tol:
                vn = evecs[:, n]
                elem = np.vdot(vm, drho @ vn)
                qfi += 2.0 * (abs(elem) ** 2) / denom

    return float(np.real(qfi))


def bath_qfi_from_plus_minus(rho_plus: qt.Qobj, rho_minus: qt.Qobj, cfg: Config) -> float:
    """
    Bath-only QFI from finite difference in J.

    The central spin is traced out.
    """
    rho_b_plus = rho_plus.ptrace(1).full()
    rho_b_minus = rho_minus.ptrace(1).full()

    rho = 0.5 * (rho_b_plus + rho_b_minus)
    drho = (rho_b_plus - rho_b_minus) / (2.0 * cfg.dJ)

    return qfi_from_rho_and_drho(rho, drho, tol=cfg.qfi_tol)


# ============================================================
# Baseline and Floquet simulation
# ============================================================

def simulate_baseline_curve(cfg: Config, ops: dict, period_time: float, n_periods: int):
    """
    Baseline protocol:

        evolve under H = J sigma_z S_z for the same wall-clock times.

    Returns:
        times, qfi, qfi_rate
    """
    S_plus = free_evolution_superprop(cfg.J0 + cfg.dJ, period_time, cfg, ops)
    S_minus = free_evolution_superprop(cfg.J0 - cfg.dJ, period_time, cfg, ops)

    rho_plus = ops["rho0"].copy()
    rho_minus = ops["rho0"].copy()

    times = np.zeros(n_periods)
    qfi = np.zeros(n_periods)

    for n in range(n_periods):
        rho_plus = apply_superprop(rho_plus, S_plus, cfg)
        rho_minus = apply_superprop(rho_minus, S_minus, cfg)

        T = (n + 1) * period_time
        times[n] = T

        qfi[n] = bath_qfi_from_plus_minus(rho_plus, rho_minus, cfg)

    qfi_rate = qfi / (times + cfg.eps)

    return times, qfi, qfi_rate


def build_floquet_period_superprop(J: float, pulse_angle: float, tau: float, cfg: Config, ops: dict):
    r"""
    One symmetric Floquet period:

        U_F = R_x(phi/2) exp(-i J sigma_z S_z tau) R_x(phi/2)

    where pulse_angle = phi.

    In superoperator language:

        S_F = S_xhalf S_z S_xhalf

    because the rightmost operation acts first.
    """
    half_angle = 0.5 * pulse_angle

    S_xhalf, half_pulse_time = central_rotation_superprop(half_angle, cfg, ops)
    S_z = free_evolution_superprop(J, tau, cfg, ops)

    S_period = S_xhalf * S_z * S_xhalf

    total_period_time = tau + 2.0 * half_pulse_time

    return S_period, total_period_time


def simulate_floquet_curve(cfg: Config, ops: dict, pulse_angle: float, tau: float):
    """
    Simulate the Floquet protocol for J0 +/- dJ.

    Returns:
        times, qfi, qfi_rate, period_time
    """
    S_plus, period_time = build_floquet_period_superprop(
        cfg.J0 + cfg.dJ, pulse_angle, tau, cfg, ops
    )
    S_minus, period_time_check = build_floquet_period_superprop(
        cfg.J0 - cfg.dJ, pulse_angle, tau, cfg, ops
    )

    if abs(period_time - period_time_check) > 1e-12:
        raise RuntimeError("Inconsistent period times.")

    n_periods = int(np.floor(cfg.T_max / period_time))

    if n_periods < 1:
        return np.array([]), np.array([]), np.array([]), period_time

    rho_plus = ops["rho0"].copy()
    rho_minus = ops["rho0"].copy()

    times = np.zeros(n_periods)
    qfi = np.zeros(n_periods)

    for n in range(n_periods):
        rho_plus = apply_superprop(rho_plus, S_plus, cfg)
        rho_minus = apply_superprop(rho_minus, S_minus, cfg)

        T = (n + 1) * period_time
        times[n] = T

        qfi[n] = bath_qfi_from_plus_minus(rho_plus, rho_minus, cfg)

    qfi_rate = qfi / (times + cfg.eps)

    return times, qfi, qfi_rate, period_time


# ============================================================
# Scan and plotting
# ============================================================

def run_scan():
    cfg = Config()
    ops = build_collective_system(cfg)

    pulse_angles = np.linspace(
        cfg.pulse_angle_min,
        cfg.pulse_angle_max,
        cfg.n_pulse_angles,
    )

    taus = np.linspace(cfg.tau_min, cfg.tau_max, cfg.n_taus)

    best_floquet_rate = np.zeros((len(pulse_angles), len(taus)))
    best_baseline_rate = np.zeros((len(pulse_angles), len(taus)))
    improvement_ratio = np.zeros((len(pulse_angles), len(taus)))

    best = {
        "ratio": -np.inf,
        "floquet_rate": -np.inf,
        "baseline_rate": -np.inf,
        "pulse_angle": None,
        "tau": None,
        "period_time": None,
    }

    print("Starting Floquet scan...")
    print(f"N = {cfg.N}")
    print(f"beta = {cfg.beta}, gamma = {cfg.gamma}")
    print(f"include_pulse_time = {cfg.include_pulse_time}")
    print()

    total_cases = len(pulse_angles) * len(taus)
    case = 0

    for i, phi in enumerate(pulse_angles):
        for j, tau in enumerate(taus):
            case += 1

            if case % 20 == 0 or case == 1:
                print(f"Processed {case}/{total_cases}")

            t_f, q_f, rate_f, period_time = simulate_floquet_curve(
                cfg, ops, pulse_angle=phi, tau=tau
            )

            if len(t_f) == 0:
                best_floquet_rate[i, j] = np.nan
                best_baseline_rate[i, j] = np.nan
                improvement_ratio[i, j] = np.nan
                continue

            n_periods = len(t_f)

            t_b, q_b, rate_b = simulate_baseline_curve(
                cfg, ops, period_time=period_time, n_periods=n_periods
            )

            max_f = np.max(rate_f)
            max_b = np.max(rate_b)

            best_floquet_rate[i, j] = max_f
            best_baseline_rate[i, j] = max_b
            improvement_ratio[i, j] = max_f / (max_b + cfg.eps)

            if improvement_ratio[i, j] > best["ratio"]:
                best.update({
                    "ratio": improvement_ratio[i, j],
                    "floquet_rate": max_f,
                    "baseline_rate": max_b,
                    "pulse_angle": phi,
                    "tau": tau,
                    "period_time": period_time,
                })

    print("\nScan complete.")
    print("Best protocol found:")
    print(f"  pulse angle phi       = {best['pulse_angle']:.6f}")
    print(f"  tau                   = {best['tau']:.6f}")
    print(f"  period time           = {best['period_time']:.6f}")
    print(f"  max Floquet F_Q/T     = {best['floquet_rate']:.6e}")
    print(f"  max baseline F_Q/T    = {best['baseline_rate']:.6e}")
    print(f"  improvement ratio     = {best['ratio']:.6f}")

    # Recompute best curves for plotting
    t_f, q_f, rate_f, period_time = simulate_floquet_curve(
        cfg,
        ops,
        pulse_angle=best["pulse_angle"],
        tau=best["tau"],
    )

    t_b, q_b, rate_b = simulate_baseline_curve(
        cfg,
        ops,
        period_time=period_time,
        n_periods=len(t_f),
    )

    sens_f = np.sqrt(t_f / (q_f + cfg.eps))
    sens_b = np.sqrt(t_b / (q_b + cfg.eps))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Heatmap of improvement ratio
    im = axes[0].pcolormesh(
        taus,
        pulse_angles,
        improvement_ratio,
        shading="auto",
    )
    axes[0].set_xlabel(r"free-evolution time $\tau$")
    axes[0].set_ylabel(r"pulse angle $\phi$ in $R_x(\phi)$")
    axes[0].set_title(r"Best ratio: Floquet $(F_Q/T)$ / baseline $(F_Q/T)$")
    fig.colorbar(im, ax=axes[0], label="improvement ratio")

    axes[0].scatter(
        [best["tau"]],
        [best["pulse_angle"]],
        marker="x",
        s=80,
        linewidths=2,
        label="best",
    )
    axes[0].legend()

    # QFI rate curve
    axes[1].plot(
        t_b,
        rate_b,
        linestyle="--",
        linewidth=2,
        label="baseline",
    )
    axes[1].plot(
        t_f,
        rate_f,
        linestyle="-",
        linewidth=2,
        label="best Floquet",
    )
    axes[1].set_xlabel("total wall-clock time T")
    axes[1].set_ylabel(r"bath-only QFI rate $F_Q^{(B)}(T)/T$")
    axes[1].set_title("Time-normalized QFI")
    axes[1].grid(True, linestyle=":")
    axes[1].legend()

    # Sensitivity curve
    axes[2].plot(
        t_b,
        sens_b,
        linestyle="--",
        linewidth=2,
        label="baseline",
    )
    axes[2].plot(
        t_f,
        sens_f,
        linestyle="-",
        linewidth=2,
        label="best Floquet",
    )
    axes[2].set_xlabel("total wall-clock time T")
    axes[2].set_ylabel(r"time-normalized sensitivity $\sqrt{T/F_Q^{(B)}}$")
    axes[2].set_title("Sensitivity, lower is better")
    axes[2].grid(True, linestyle=":")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(cfg.output_figure, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"\nSaved figure to: {cfg.output_figure}")

    return {
        "cfg": cfg,
        "pulse_angles": pulse_angles,
        "taus": taus,
        "best_floquet_rate": best_floquet_rate,
        "best_baseline_rate": best_baseline_rate,
        "improvement_ratio": improvement_ratio,
        "best": best,
        "best_curves": {
            "t_f": t_f,
            "q_f": q_f,
            "rate_f": rate_f,
            "sens_f": sens_f,
            "t_b": t_b,
            "q_b": q_b,
            "rate_b": rate_b,
            "sens_b": sens_b,
        },
    }


if __name__ == "__main__":
    results = run_scan()