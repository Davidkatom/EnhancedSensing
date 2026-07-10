import itertools
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    # Keep N small: this code uses the full 2^(N+1) Hilbert space.
    # N=4 is safe. N=5 may be slower. N=6 can become expensive.
    N: int = 4

    # Parameter to estimate
    J0: float = 1.0
    dJ: float = 1e-3

    # Central-spin decoherence channels:
    # beta: x-dephasing / bit-flip-like noise on central spin
    # gamma: z-dephasing on central spin
    beta: float = 0.3
    gamma: float = 0.0

    # Total maximum experiment time
    T_max: float = 10.0
    n_time_points: int = 80

    # Monte Carlo trajectories for sequential weak-measurement protocols
    n_trajectories: int = 400

    # Weak measurement strengths
    epsilons: tuple = (0.1, 0.3, 0.5, 0.8, 1.0)

    # Measurement rates.
    # k = number of random bath-spin measurements per unit time.
    # k=0 is the baseline: no intermediate measurements, measure all bath spins at end.
    k_rates: tuple = (0.0, 0.5, 1.0, 2.0)

    # Numerical safety
    prob_floor: float = 1e-14

    # Random seed
    seed: int = 1234

    # Output
    output_figure: str = "random_weak_bath_measurement_fi_rate.png"


# ============================================================
# Basic operators
# ============================================================

def embed_single_qubit(op: qt.Qobj, target: int, total_qubits: int) -> qt.Qobj:
    """
    Embed a single-qubit operator into the full Hilbert space.

    target = 0 is central spin.
    target = 1,...,N are bath spins.
    """
    I = qt.qeye(2)
    ops = [I for _ in range(total_qubits)]
    ops[target] = op
    return qt.tensor(ops)


def build_system(cfg: Config):
    """
    Full Hilbert space:
        central spin + N bath spins
        dimension = 2^(N+1)
    """
    total_qubits = cfg.N + 1

    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()
    I = qt.qeye(2)

    sx_c = embed_single_qubit(sx, 0, total_qubits)
    sy_c = embed_single_qubit(sy, 0, total_qubits)
    sz_c = embed_single_qubit(sz, 0, total_qubits)

    sx_bath = []
    sy_bath = []
    sz_bath = []

    for i in range(cfg.N):
        site = i + 1
        sx_bath.append(embed_single_qubit(sx, site, total_qubits))
        sy_bath.append(embed_single_qubit(sy, site, total_qubits))
        sz_bath.append(embed_single_qubit(sz, site, total_qubits))

    Sz_bath = 0 * sz_bath[0]
    for op in sz_bath:
        Sz_bath = Sz_bath + op

    # Initial state: central |+x>, all bath spins |+x>
    plus_x = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    psi0 = qt.tensor([plus_x for _ in range(total_qubits)])
    rho0 = psi0.proj()

    return {
        "total_qubits": total_qubits,
        "sx": sx,
        "sy": sy,
        "sz": sz,
        "I": I,
        "sx_c": sx_c,
        "sy_c": sy_c,
        "sz_c": sz_c,
        "sx_bath": sx_bath,
        "sy_bath": sy_bath,
        "sz_bath": sz_bath,
        "Sz_bath": Sz_bath,
        "rho0": rho0,
    }


def build_hamiltonian(J: float, ops: dict) -> qt.Qobj:
    """
    Central-spin sensing Hamiltonian:

        H = J sigma_z^c sum_i sigma_z^i

    No transverse field is included here.
    This file is testing measurement-based readout, not x-drive control.
    """
    return J * ops["sz_c"] * ops["Sz_bath"]


def build_c_ops(cfg: Config, ops: dict):
    """
    Unobserved decoherence channels on the central spin.
    These are not measurement channels.
    """
    c_ops = []

    if cfg.beta > 0:
        c_ops.append(np.sqrt(cfg.beta) * ops["sx_c"])

    if cfg.gamma > 0:
        c_ops.append(np.sqrt(cfg.gamma) * ops["sz_c"])

    return c_ops


# ============================================================
# Evolution helpers
# ============================================================

def sanitize_rho(rho: qt.Qobj, prob_floor: float = 1e-14) -> qt.Qobj:
    """
    Remove small numerical non-Hermiticity and normalize trace.
    """
    rho = 0.5 * (rho + rho.dag())
    tr = np.real(rho.tr())

    if abs(tr) < prob_floor:
        raise RuntimeError("Density matrix trace vanished.")

    return rho / tr


def build_liouvillian_propagator(J: float, dt: float, cfg: Config, ops: dict) -> qt.Qobj:
    """
    Build exp(L dt), where L is the Lindblad Liouvillian.

    This is used for fast repeated evolution between measurement events.
    """
    H = build_hamiltonian(J, ops)
    c_ops = build_c_ops(cfg, ops)
    L = qt.liouvillian(H, c_ops)
    return (L * dt).expm()


def evolve_with_superoperator(rho: qt.Qobj, U_super: qt.Qobj, cfg: Config) -> qt.Qobj:
    """
    Apply a Liouville-space propagator to a density matrix.
    """
    vec = qt.operator_to_vector(rho)
    vec_new = U_super * vec
    rho_new = qt.vector_to_operator(vec_new)
    return sanitize_rho(rho_new, cfg.prob_floor)


# ============================================================
# Weak measurement of one bath spin in x basis
# ============================================================

def weak_x_kraus_single_spin(epsilon: float):
    """
    Weak measurement of sigma_x.

    epsilon = 0: no information, no disturbance.
    epsilon = 1: projective measurement in x basis.

    Outcomes are labeled +1 and -1.
    """
    if epsilon < 0 or epsilon > 1:
        raise ValueError("epsilon must satisfy 0 <= epsilon <= 1")

    I = qt.qeye(2)
    sx = qt.sigmax()

    P_plus = (I + sx) / 2.0
    P_minus = (I - sx) / 2.0

    M_plus = (
        np.sqrt((1.0 + epsilon) / 2.0) * P_plus
        + np.sqrt((1.0 - epsilon) / 2.0) * P_minus
    )

    M_minus = (
        np.sqrt((1.0 - epsilon) / 2.0) * P_plus
        + np.sqrt((1.0 + epsilon) / 2.0) * P_minus
    )

    return M_plus, M_minus


def build_weak_x_measurement_ops(cfg: Config, epsilon: float):
    """
    Build full-system Kraus operators for weakly measuring sigma_x
    on each bath spin.

    Returns:
        measurement_ops[i][+1] and measurement_ops[i][-1]
    """
    total_qubits = cfg.N + 1

    M_plus_single, M_minus_single = weak_x_kraus_single_spin(epsilon)

    measurement_ops = []

    for i in range(cfg.N):
        site = i + 1  # bath spin i
        M_plus_full = embed_single_qubit(M_plus_single, site, total_qubits)
        M_minus_full = embed_single_qubit(M_minus_single, site, total_qubits)

        measurement_ops.append({
            +1: M_plus_full,
            -1: M_minus_full,
        })

    return measurement_ops


def probability_of_outcome(rho: qt.Qobj, M: qt.Qobj) -> float:
    """
    p = Tr(M rho M^\dagger)
    """
    tmp = M * rho * M.dag()
    return float(np.real(tmp.tr()))


def apply_measurement_update(rho: qt.Qobj, M: qt.Qobj, cfg: Config):
    """
    Apply the measurement update rho -> M rho M^\dagger / p.
    """
    tmp = M * rho * M.dag()
    p = float(np.real(tmp.tr()))
    p_safe = max(p, cfg.prob_floor)

    rho_new = tmp / p_safe
    rho_new = sanitize_rho(rho_new, cfg.prob_floor)

    return rho_new, p_safe


# ============================================================
# Baseline: final strong measurement of all bath spins in x
# ============================================================

def build_all_bath_x_projectors(cfg: Config):
    """
    Projective measurement of every bath spin in the x basis.

    Central spin is not measured, so each projector is:

        I_central tensor |x_1 ... x_N><x_1 ... x_N|
    """
    plus_x = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    minus_x = (qt.basis(2, 0) - qt.basis(2, 1)).unit()

    I_c = qt.qeye(2)

    projectors = []

    for bits in itertools.product([+1, -1], repeat=cfg.N):
        bath_kets = []
        for b in bits:
            bath_kets.append(plus_x if b == +1 else minus_x)

        bath_state = qt.tensor(bath_kets)
        P_bath = bath_state.proj()

        P_full = qt.tensor(I_c, P_bath)
        projectors.append(P_full)

    return projectors


def probabilities_from_projectors(rho: qt.Qobj, projectors: list, prob_floor: float):
    """
    Compute probabilities for a projective measurement.
    """
    probs = np.array([float(np.real((P * rho).tr())) for P in projectors])
    probs = np.maximum(probs, 0.0)

    total = np.sum(probs)
    if total > prob_floor:
        probs = probs / total

    return probs


def run_mesolve_states(J: float, times: np.ndarray, cfg: Config, ops: dict):
    """
    Evolve from t=0 and return rho(t) at requested times.
    """
    H = build_hamiltonian(J, ops)
    c_ops = build_c_ops(cfg, ops)

    times_with_zero = np.concatenate([[0.0], times])
    result = qt.mesolve(H, ops["rho0"], times_with_zero, c_ops=c_ops, e_ops=[])

    return result.states[1:]

def classical_fi_hellinger(pp, pm, dJ):
    """
    Numerically stable finite-difference classical Fisher information.

    Uses:
        F ≈ sum_m [sqrt(p_m(J+dJ)) - sqrt(p_m(J-dJ))]^2 / dJ^2

    This is equivalent to F = sum (dp^2 / p), but is much more stable
    near zero-probability outcomes.
    """
    pp = np.asarray(pp, dtype=float)
    pm = np.asarray(pm, dtype=float)

    pp = np.maximum(pp, 0.0)
    pm = np.maximum(pm, 0.0)

    pp = pp / np.sum(pp)
    pm = pm / np.sum(pm)

    return np.sum((np.sqrt(pp) - np.sqrt(pm)) ** 2) / (dJ ** 2)

def compute_baseline_final_x_cfi_rate(cfg: Config, ops: dict, times: np.ndarray):
    """
    Stable classical Fisher information for:

        evolve to T,
        then strongly measure all bath spins in x basis.

    Uses Hellinger finite-difference form instead of dp^2 / p.
    """
    projectors = build_all_bath_x_projectors(cfg)

    states_p = run_mesolve_states(cfg.J0 + cfg.dJ, times, cfg, ops)
    states_m = run_mesolve_states(cfg.J0 - cfg.dJ, times, cfg, ops)

    cfi = np.zeros_like(times)

    for idx, (rhop, rhom) in enumerate(zip(states_p, states_m)):
        pp = probabilities_from_projectors(rhop, projectors, cfg.prob_floor)
        pm = probabilities_from_projectors(rhom, projectors, cfg.prob_floor)

        cfi[idx] = classical_fi_hellinger(pp, pm, cfg.dJ)

    cfi_rate = cfi / times

    return cfi, cfi_rate


# ============================================================
# Sequential random weak measurements
# ============================================================

def estimate_record_fi_rate_for_protocol(
    cfg: Config,
    ops: dict,
    epsilon: float,
    k_rate: float,
    times: np.ndarray,
    seed: int,
):
    """
    Estimate the classical Fisher information in the sequential
    weak-measurement record.

    Protocol:
        every tau = 1/k_rate,
        choose random bath spin i,
        weakly measure sigma_x_i with strength epsilon,
        store outcome +1/-1.

    FI estimator:
        sample records at J0,
        compute score via finite-difference log likelihood:

            score = [log P(record | J0+dJ) - log P(record | J0-dJ)] / (2 dJ)

        F ~= E[score^2].

    This uses only the measurement record, not a final all-bath readout.
    """
    if k_rate <= 0:
        raise ValueError("k_rate must be positive for sequential measurement protocols.")

    rng = np.random.default_rng(seed)

    tau = 1.0 / k_rate
    event_times = np.arange(tau, cfg.T_max + 1e-12, tau)

    if len(event_times) == 0:
        return np.zeros_like(times), np.zeros_like(times)

    # Precompute one-step propagators for this tau
    U0 = build_liouvillian_propagator(cfg.J0, tau, cfg, ops)
    Up = build_liouvillian_propagator(cfg.J0 + cfg.dJ, tau, cfg, ops)
    Um = build_liouvillian_propagator(cfg.J0 - cfg.dJ, tau, cfg, ops)

    # Weak measurement operators
    meas_ops = build_weak_x_measurement_ops(cfg, epsilon)

    score_sum = np.zeros_like(times)
    score_sq_sum = np.zeros_like(times)

    for traj in range(cfg.n_trajectories):
        # True state used to generate records
        rho_true = ops["rho0"].copy()

        # Filter states used to evaluate likelihood under J+dJ and J-dJ
        rho_p = ops["rho0"].copy()
        rho_m = ops["rho0"].copy()

        logL_p = 0.0
        logL_m = 0.0

        scores_at_events = np.zeros(len(event_times))

        for ev_idx, _t_event in enumerate(event_times):
            # Evolve all three states to the next measurement event
            rho_true = evolve_with_superoperator(rho_true, U0, cfg)
            rho_p = evolve_with_superoperator(rho_p, Up, cfg)
            rho_m = evolve_with_superoperator(rho_m, Um, cfg)

            # Randomly choose a bath spin to measure
            bath_spin = rng.integers(0, cfg.N)

            M_plus = meas_ops[bath_spin][+1]
            M_minus = meas_ops[bath_spin][-1]

            # Sample outcome from the true state
            p_true_plus = probability_of_outcome(rho_true, M_plus)
            p_true_minus = probability_of_outcome(rho_true, M_minus)

            p_sum = p_true_plus + p_true_minus
            if p_sum < cfg.prob_floor:
                p_true_plus = 0.5
            else:
                p_true_plus = p_true_plus / p_sum

            outcome = +1 if rng.random() < p_true_plus else -1
            M_out = meas_ops[bath_spin][outcome]

            # Update true state
            rho_true, _ = apply_measurement_update(rho_true, M_out, cfg)

            # Likelihood/update under J+dJ
            rho_p, p_out_p = apply_measurement_update(rho_p, M_out, cfg)
            logL_p += np.log(max(p_out_p, cfg.prob_floor))

            # Likelihood/update under J-dJ
            rho_m, p_out_m = apply_measurement_update(rho_m, M_out, cfg)
            logL_m += np.log(max(p_out_m, cfg.prob_floor))

            # Finite-difference score for the record up to this event
            score = (logL_p - logL_m) / (2.0 * cfg.dJ)
            scores_at_events[ev_idx] = score

        # Convert event scores into scores at requested plot times.
        # Between measurement events, the observed record has not changed,
        # so the record Fisher information is piecewise constant.
        event_index_for_time = np.searchsorted(event_times, times, side="right") - 1

        scores_at_times = np.zeros_like(times)
        valid = event_index_for_time >= 0
        scores_at_times[valid] = scores_at_events[event_index_for_time[valid]]

        score_sum += scores_at_times
        score_sq_sum += scores_at_times ** 2

    mean_score = score_sum / cfg.n_trajectories
    mean_score_sq = score_sq_sum / cfg.n_trajectories

    # For exact scores, E[score] = 0 and FI = E[score^2].
    # With finite samples and finite dJ, centering reduces Monte Carlo bias.
    cfi_record = mean_score_sq - mean_score ** 2
    cfi_record = np.maximum(cfi_record, 0.0)

    cfi_rate = cfi_record / times

    return cfi_record, cfi_rate


# ============================================================
# Plotting
# ============================================================

def make_protocol_list(cfg: Config):
    """
    Make exactly 16 panels:

        1 baseline:
            k = 0, epsilon = 1, final all-x measurement

        15 sequential protocols:
            5 epsilons x 3 nonzero k rates
    """
    nonzero_k = [k for k in cfg.k_rates if k > 0]

    protocols = [
        {
            "kind": "baseline",
            "epsilon": 1.0,
            "k_rate": 0.0,
            "title": r"Baseline: final all-$x$ readout",
        }
    ]

    for k_rate in nonzero_k:
        for eps in cfg.epsilons:
            protocols.append({
                "kind": "sequential",
                "epsilon": eps,
                "k_rate": k_rate,
                "title": fr"$\epsilon={eps}$, $k={k_rate}$",
            })

    if len(protocols) != 16:
        print(f"Warning: protocol count is {len(protocols)}, not 16.")
        print("For 16 panels, use 5 epsilons and 4 k values including k=0.")

    return protocols


def run_all_and_plot():
    cfg = Config()
    ops = build_system(cfg)

    # Times are total experiment times T.
    # Start above zero because we plot F(T)/T.
    times = np.linspace(0.05, cfg.T_max, cfg.n_time_points)

    print("Computing baseline: final strong all-bath x measurement...")
    baseline_cfi, baseline_cfi_rate = compute_baseline_final_x_cfi_rate(cfg, ops, times)

    protocols = make_protocol_list(cfg)

    results = []

    for p_idx, protocol in enumerate(protocols):
        if protocol["kind"] == "baseline":
            results.append({
                **protocol,
                "cfi": baseline_cfi,
                "cfi_rate": baseline_cfi_rate,
            })
            continue

        eps = protocol["epsilon"]
        k_rate = protocol["k_rate"]

        print(f"Computing sequential protocol: epsilon={eps}, k={k_rate}")

        cfi, cfi_rate = estimate_record_fi_rate_for_protocol(
            cfg=cfg,
            ops=ops,
            epsilon=eps,
            k_rate=k_rate,
            times=times,
            seed=cfg.seed + 1000 * p_idx,
        )

        results.append({
            **protocol,
            "cfi": cfi,
            "cfi_rate": cfi_rate,
        })

    # Plot 16 panels
    fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=True, sharey=False)
    axes = axes.flatten()

    for ax, result in zip(axes, results):
        if result["kind"] == "baseline":
            ax.plot(
                times,
                result["cfi_rate"],
                linewidth=2.5,
                label="baseline",
            )
        else:
            # Overlay baseline in every panel for comparison
            ax.plot(
                times,
                baseline_cfi_rate,
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="baseline",
            )

            ax.plot(
                times,
                result["cfi_rate"],
                linewidth=2.0,
                label="weak record",
            )

        ax.set_title(result["title"])
        ax.set_xlabel("total experiment time $T$")
        ax.set_ylabel(r"time-normalized FI $F_J(T)/T$")
        ax.grid(True, linestyle=":")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(cfg.output_figure, dpi=200, bbox_inches="tight")
    plt.show()

    # Print summary
    print("\nSummary: best time-normalized Fisher information rate")
    print("Higher F/T is better.\n")

    for result in results:
        idx = int(np.argmax(result["cfi_rate"]))
        best_rate = result["cfi_rate"][idx]
        best_time = times[idx]

        print(
            f"{result['title']:<45s} "
            f"max F/T = {best_rate:.6e} at T = {best_time:.3f}"
        )

    print(f"\nSaved figure to: {cfg.output_figure}")


if __name__ == "__main__":
    run_all_and_plot()