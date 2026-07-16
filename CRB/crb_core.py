"""Shared numerical utilities for collective-bath CRB analyses.

The exact solver in this module uses the symmetric bath subspace and the
Hamiltonian convention

    H = Omega_0 * sigma_x + J * sigma_z * S_z,

where ``S_z = 2 * jmat(N / 2, "z")``.  Keeping that convention explicit is
important when comparing these helpers with scripts that use a factor of one
half in either the drive or collective-spin operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import qutip as qt


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration fields shared by the collective-bath analyses."""

    N: int = 8
    J_nominal: float = 1.0
    dJ: float = 1e-3

    t_min: float = 0.01
    t_max: float = 60.0
    n_steps: int = 300

    gamma: float = 0.0
    beta: float = 0.3

    qfi_tol: float = 1e-12
    qcrb_eps: float = 1e-15
    t_overhead: float = 5.0


# ---------------------------------------------------------------------------
# Operator and state construction
# ---------------------------------------------------------------------------


def build_spin_operators(N: int) -> dict[str, object]:
    """Build central-spin and collective-bath operators.

    The bath is represented in its spin-``N / 2`` symmetric subspace, whose
    dimension is ``N + 1``.
    """
    if N < 0:
        raise ValueError("N must be non-negative")

    S_spin = N / 2.0
    dim_bath = N + 1
    Jz = 2.0 * qt.jmat(S_spin, "z")
    I_bath = qt.qeye(dim_bath)

    sx = qt.sigmax()
    sz = qt.sigmaz()
    si = qt.qeye(2)

    return {
        "S_spin": S_spin,
        "dim_bath": dim_bath,
        "Jz": Jz,
        "I_bath": I_bath,
        "sx": sx,
        "sz": sz,
        "si": si,
        "sx_s": qt.tensor(sx, I_bath),
        "sz_s": qt.tensor(sz, I_bath),
        "Sz_op": qt.tensor(si, Jz),
    }


def build_initial_state(S_spin: float) -> qt.Qobj:
    """Return the central ``|+x>`` and bath ``+x`` coherent product state."""
    plus_state_central = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    plus_state_bath = qt.spin_coherent(S_spin, np.pi / 2.0, 0.0)
    return qt.tensor(plus_state_central, plus_state_bath)


def build_hamiltonian(Omega_0: float, J: float, N: int) -> qt.Qobj:
    """Return ``Omega_0 sigma_x + J sigma_z S_z`` in the symmetric subspace."""
    operators = build_spin_operators(N)
    return (
        Omega_0 * operators["sx_s"]
        + J * operators["sz_s"] * operators["Sz_op"]
    )


def build_bath_operators(N: int) -> dict[str, np.ndarray]:
    """Return dense collective bath operators using the solver's scaling."""
    if N < 0:
        raise ValueError("N must be non-negative")

    spin = N / 2.0
    return {
        "I": np.eye(N + 1, dtype=complex),
        "Jx": 2.0 * qt.jmat(spin, "x").full(),
        "Jy": 2.0 * qt.jmat(spin, "y").full(),
        "Jz": 2.0 * qt.jmat(spin, "z").full(),
    }


# ---------------------------------------------------------------------------
# Exact block-decomposed bath evolution
# ---------------------------------------------------------------------------


def get_bath_density_matrices(
    Omega_0: float,
    J: float,
    tlist: Sequence[float] | np.ndarray,
    N: int = 10,
    gamma: float = 1.0,
    beta: float = 1.0,
) -> list[np.ndarray]:
    """Evolve and return ``rho_B(t) = Tr_central[rho(t)]``.

    Each bath coherence closes on a four-dimensional central-spin block.  The
    resulting independent constant-coefficient Liouvillians are solved by
    eigendecomposition, avoiding construction of the full superoperator.
    """
    if N < 0:
        raise ValueError("N must be non-negative")

    times = np.asarray(tlist, dtype=float)
    if times.ndim != 1:
        raise ValueError("tlist must be one-dimensional")

    S_spin = N / 2.0
    dim_bath = N + 1

    # Ordering matches qt.jmat and qt.spin_coherent.
    s_vals = 2.0 * np.real(np.diag(qt.jmat(S_spin, "z").full()))
    chi = qt.spin_coherent(S_spin, np.pi / 2.0, 0.0).full().ravel()

    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    identity_2 = np.eye(2, dtype=complex)
    identity_4 = np.eye(4, dtype=complex)

    # Row-major vectorization: vec(A R B) = (A kron B.T) vec(R).
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    central_rho_0 = np.outer(plus, plus.conj()).reshape(4)
    dissipator = (
        beta * (np.kron(sx, sx.T) - identity_4)
        + gamma * (np.kron(sz, sz.T) - identity_4)
    )

    bath = np.zeros((len(times), dim_bath, dim_bath), dtype=complex)
    for i in range(dim_bath):
        H_i = Omega_0 * sx + J * s_vals[i] * sz
        for j in range(i, dim_bath):
            H_j = Omega_0 * sx + J * s_vals[j] * sz
            liouvillian = (
                -1j
                * (
                    np.kron(H_i, identity_2)
                    - np.kron(identity_2, H_j.T)
                )
                + dissipator
            )

            eigenvalues, eigenvectors = np.linalg.eig(liouvillian)
            coefficients = np.linalg.solve(eigenvectors, central_rho_0)
            coefficients *= chi[i] * np.conj(chi[j])
            evolved = eigenvectors @ (
                np.exp(np.outer(eigenvalues, times)) * coefficients[:, None]
            )

            reduced_element = evolved[0] + evolved[3]
            bath[:, i, j] = reduced_element
            if j != i:
                bath[:, j, i] = np.conj(reduced_element)

    return list(bath)


# ---------------------------------------------------------------------------
# Quantum Fisher information
# ---------------------------------------------------------------------------


def qfi_from_rho_and_drho(
    rho: np.ndarray,
    drho: np.ndarray,
    tol: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """Return the mixed-state QFI and symmetric logarithmic derivative."""
    rho = np.asarray(rho, dtype=complex)
    drho = np.asarray(drho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix")
    if drho.shape != rho.shape:
        raise ValueError("drho must have the same shape as rho")

    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.real(eigenvalues)
    eigenvalues[np.abs(eigenvalues) < tol] = 0.0

    derivative_eigenbasis = eigenvectors.conj().T @ drho @ eigenvectors
    denominator = eigenvalues[:, None] + eigenvalues[None, :]
    inverse_weight = np.zeros_like(denominator)
    valid = denominator > tol
    inverse_weight[valid] = 2.0 / denominator[valid]

    qfi = np.sum(np.abs(derivative_eigenbasis) ** 2 * inverse_weight)
    sld_eigenbasis = derivative_eigenbasis * inverse_weight
    sld = eigenvectors @ sld_eigenbasis @ eigenvectors.conj().T
    sld = 0.5 * (sld + sld.conj().T)
    return float(np.real(qfi)), sld


def qfi_vectorized(
    rho: np.ndarray,
    drho: np.ndarray,
    tol: float = 1e-12,
) -> float:
    """Return only the scalar QFI, without retaining the SLD."""
    rho = np.asarray(rho, dtype=complex)
    drho = np.asarray(drho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix")
    if drho.shape != rho.shape:
        raise ValueError("drho must have the same shape as rho")

    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    # Preserve qfi_N_scaling.py's treatment of numerical noise by clipping
    # all negative eigenvalues in the scalar-only implementation.
    eigenvalues = np.clip(np.real(eigenvalues), 0.0, None)

    derivative_eigenbasis = eigenvectors.conj().T @ drho @ eigenvectors
    denominator = eigenvalues[:, None] + eigenvalues[None, :]
    valid = denominator > tol
    return float(
        np.real(
            2.0
            * np.sum(
                np.abs(derivative_eigenbasis[valid]) ** 2 / denominator[valid]
            )
        )
    )


def compute_bath_qfi_trajectory(
    bath_rhos_plus: Sequence[np.ndarray],
    bath_rhos_minus: Sequence[np.ndarray],
    dJ: float,
    tol: float = 1e-12,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Compute a finite-difference bath QFI trajectory and its intermediates."""
    if len(bath_rhos_plus) != len(bath_rhos_minus):
        raise ValueError("plus and minus trajectories must have the same length")
    if dJ == 0:
        raise ValueError("dJ must be non-zero")

    qfi_t = np.zeros(len(bath_rhos_plus))
    sld_t: list[np.ndarray] = []
    rho_t: list[np.ndarray] = []
    drho_t: list[np.ndarray] = []
    for index, (rho_plus, rho_minus) in enumerate(
        zip(bath_rhos_plus, bath_rhos_minus)
    ):
        rho = 0.5 * (rho_plus + rho_minus)
        drho = (rho_plus - rho_minus) / (2.0 * dJ)
        qfi_t[index], sld = qfi_from_rho_and_drho(rho, drho, tol=tol)
        sld_t.append(sld)
        rho_t.append(rho)
        drho_t.append(drho)

    return qfi_t, sld_t, rho_t, drho_t


# ---------------------------------------------------------------------------
# QCRB utilities
# ---------------------------------------------------------------------------


def compute_qcrb_matrices(
    qfi_matrix: np.ndarray,
    tlist: Sequence[float] | np.ndarray,
    t_overhead: float,
    qcrb_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized/unnormalized QCRBs and normalized optimum indices."""
    qfi = np.asarray(qfi_matrix, dtype=float)
    times = np.asarray(tlist, dtype=float)
    if qfi.ndim != 2:
        raise ValueError("qfi_matrix must be two-dimensional")
    if times.ndim != 1 or qfi.shape[1] != len(times):
        raise ValueError("the second qfi_matrix axis must match tlist")

    normalized = np.sqrt((times + t_overhead)[None, :] / (qfi + qcrb_eps))
    unnormalized = 1.0 / np.sqrt(qfi + qcrb_eps)
    optimal_time_indices = np.argmin(normalized, axis=1)
    return normalized, unnormalized, optimal_time_indices


# ---------------------------------------------------------------------------
# Fisher-metric projection onto operator bases
# ---------------------------------------------------------------------------


def frobenius_orthonormal_span(
    operators: Sequence[np.ndarray],
    tol: float = 1e-12,
) -> list[np.ndarray]:
    """Return a stable Hermitian basis for the same real operator span."""
    basis: list[np.ndarray] = []
    for operator in operators:
        candidate = np.asarray(operator, dtype=complex)
        candidate = 0.5 * (candidate + candidate.conj().T)
        for previous in basis:
            candidate -= np.real(np.vdot(previous, candidate)) * previous
        norm = np.linalg.norm(candidate, ord="fro")
        if norm > tol:
            basis.append(candidate / norm)
    return basis


def fisher_metric_projection(
    rho: np.ndarray,
    drho: np.ndarray,
    operators: Sequence[np.ndarray],
    ridge: float = 1e-10,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Project the SLD onto an operator span using Fisher normal equations."""
    stable_basis = frobenius_orthonormal_span(operators)
    count = len(stable_basis)
    if count == 0:
        return 0.0, np.empty(0), np.empty((0, 0)), np.empty(0)

    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)
    metric = np.empty((count, count), dtype=float)
    derivative = np.empty(count, dtype=float)
    rho_times_operators = [rho @ operator for operator in stable_basis]

    for i, operator_i in enumerate(stable_basis):
        derivative[i] = float(np.real(np.trace(operator_i @ drho)))
        for j in range(i, count):
            value = float(
                np.real(np.trace(rho_times_operators[i] @ stable_basis[j]))
            )
            metric[i, j] = value
            metric[j, i] = value

    metric = 0.5 * (metric + metric.T)
    scale = max(
        float(np.max(np.abs(np.diag(metric)))),
        np.finfo(float).tiny,
    )
    regularized_metric = metric + ridge * scale * np.eye(count)
    try:
        coefficients = np.linalg.solve(regularized_metric, derivative)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(
            regularized_metric, derivative, rcond=None
        )[0]

    projected_qfi = max(float(np.dot(coefficients, derivative)), 0.0)
    return projected_qfi, coefficients, metric, derivative


def build_configured_operator_bases(
    operators: dict[str, np.ndarray],
) -> dict[str, list[np.ndarray]]:
    """Build the standard linear and second-moment readout bases."""
    identity = operators["I"]
    Jx = operators["Jx"]
    Jy = operators["Jy"]
    linear = [identity, Jx, Jy]
    return {
        "Linear": linear,
        "Second moments": linear
        + [Jx @ Jx, Jy @ Jy, Jx @ Jy + Jy @ Jx],
    }


# ---------------------------------------------------------------------------
# Scaling fits
# ---------------------------------------------------------------------------


def fit_power_law(N: np.ndarray, FQ: np.ndarray) -> float:
    """Fit ``FQ ~ N**p`` on the upper half of the supplied sizes."""
    sizes = np.asarray(N, dtype=float)
    values = np.asarray(FQ, dtype=float)
    if sizes.ndim != 1 or values.shape != sizes.shape:
        raise ValueError("N and FQ must be one-dimensional arrays of equal length")

    midpoint = len(sizes) // 2
    start = max(0, midpoint - 1)
    valid = (sizes[start:] > 0.0) & (values[start:] > 0.0)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    exponent, _ = np.polyfit(
        np.log(sizes[start:][valid]),
        np.log(values[start:][valid]),
        1,
    )
    return float(exponent)


__all__ = [
    "SimulationConfig",
    "build_bath_operators",
    "build_configured_operator_bases",
    "build_hamiltonian",
    "build_initial_state",
    "build_spin_operators",
    "compute_bath_qfi_trajectory",
    "compute_qcrb_matrices",
    "fisher_metric_projection",
    "fit_power_law",
    "frobenius_orthonormal_span",
    "get_bath_density_matrices",
    "qfi_from_rho_and_drho",
    "qfi_vectorized",
]
