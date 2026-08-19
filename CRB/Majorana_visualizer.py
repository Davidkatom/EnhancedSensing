"""Animate the driven central-spin model with a Majorana bath constellation.

This is a companion to :mod:`state_visualizer`.  The physics -- Hamiltonian,
initial state, and evolution -- is identical; only the *right* sphere changes.
Where ``state_visualizer`` paints the reduced bath state with its Husimi-Q
quasiprobability, this script draws the bath's **Majorana stellar
representation** (the "constellation").

The model is the driven central-spin Hamiltonian used throughout this project
(see :func:`CRB.crb_core.build_hamiltonian`):

    H = Omega * sigma_x  +  J * sigma_z * S_z  +  omega * S_x ,

with central-spin Pauli operators ``X = sigma_x``, ``Z = sigma_z`` and
collective-bath operators ``S_i = 2 * jmat(N / 2, i)``.  The initial state is
``|psi(0)> = |0>_central (x) |+>^N_bath``.

The Majorana representation and mixed states
-------------------------------------------
A *pure* spin-``S`` state ``|chi> = sum_m c_m |S, m>`` is represented by the
``2S`` roots of its Majorana polynomial

    P(z) = sum_{k=0}^{2S} (-1)^k sqrt(C(2S, k)) c_{S-k} z^{2S-k},

each root ``z = tan(theta/2) e^{i phi}`` giving one "star" at ``(theta, phi)``
on the sphere (a root at infinity sits at the south pole).  The constellation is
rotationally covariant: a spin-coherent state has all ``2S`` stars piled at a
single point, and the more "spread out" the stars, the more non-classical the
state.

The reduced bath state ``rho_b(t) = Tr_central |psi(t)><psi(t)|`` is generally
**mixed** (the central spin entangles with it -- this is exactly the dephasing
the model exploits, and without dissipation ``rho_b`` has rank <= 2).  Since the
stellar representation is defined for pure states, we diagonalize

    rho_b(t) = sum_i p_i |chi_i(t)><chi_i(t)|

and draw the constellation of each leading eigenstate ``|chi_i>``, encoding its
weight ``p_i`` in the stars' opacity and size.  When ``rho_b`` is pure this is a
single constellation; as the bath mixes, a second (fainter) constellation fades
in -- the right-sphere analogue of the left Bloch vector shrinking inside the
sphere.  ``--majorana-n-components`` sets how many eigenstates are drawn.

The animation shows, at each interrogation time ``t``:

    * left  -- the Bloch sphere of the reduced central spin ``rho_c(t)``; its
               Bloch vector shrinks inside the sphere as the state decoheres.
    * right -- the Majorana constellation(s) of the reduced collective bath
               ``rho_b(t)``: gold stars for the dominant eigenstate, blue for
               sub-dominant ones (opacity/size proportional to weight ``p_i``).

Encoding and readout overlays
-----------------------------
The right sphere carries two optional overlays that turn it into a picture of the
whole ``J``-metrology story -- how the coupling is written into the bath, and how
an ideal detector reads it back:

    * ``--show-dJ-arrows`` draws a teal arrow ``d(star)/dJ`` on each drawn star,
      the direction it drifts as ``J`` increases (estimated by re-evolving the
      bath at ``J +/- dJ_fd_step`` and tracking/matching the perturbed stars).
      Long arrows flag the stars -- hence the state directions -- most sensitive
      to ``J``: this is *how ``J`` is encoded* geometrically.
    * ``--show-sld-constellation`` draws the Majorana constellations (magenta
      diamonds) of the leading eigenvectors of the reduced-bath symmetric
      logarithmic derivative ``L_b``.  A projective measurement in the SLD
      eigenbasis saturates the bath QFI ``F_Q = Tr(rho_b L_b^2)``, with
      per-outcome share ``lambda_k^2 <l_k|rho_b|l_k>``; the top outcomes are the
      states the optimal detector projects onto -- *what measurement extracts
      that information*.  ``F_Q`` itself is reported alongside the purity.

Both overlays reuse ``crb_core.qfi_from_rho_and_drho`` for the QFI/SLD and are
exported into the star-data JSON, so the interactive viewer shows them too.

Every parameter is exposed through :class:`SimulationConfig` (and mirrored on
the command line), matching the other plot scripts in this folder.  Optional
Markovian dephasing/relaxation on the central spin is available through
``gamma`` and ``beta``; leaving both at zero evolves the pure state unitarily.

Sensing / decoding echo protocol
--------------------------------
By default the state evolves under one Hamiltonian for the whole run.  Setting
``--sense-time`` and/or ``--decode-time`` above zero reinterprets that run as
the *preparation* phase of an echo protocol and appends ``--num-of-cycles``
repetitions of a sensing segment followed by a decoding segment, each
continuing from the previous segment's final state (the state is never reset).
Sensing reuses the preparation Hamiltonian
``H_sense = Omega sigma_x + J sigma_z S_z + omega S_x``; decoding runs the
fixed-estimate decoder of the echo scripts,

    H_dec(J0) = -Omega sigma_x + J0 sigma_z S_z - omega S_x,

which flips the single-spin and bath drives while keeping the interaction sign
and substituting the estimate ``J0 = --J-estimate`` for the true ``J``.  Leaving
both ``--sense-time`` and ``--decode-time`` at zero appends nothing and
reproduces the default single-Hamiltonian behaviour.

Instead of (or in addition to) rendering a GIF, the star trajectory can be
exported as a compact JSON file with ``--save-star-data`` (the default output).
Dragging that file onto the companion ``majorana_viewer.html`` opens an
interactive WebGL view: orbit the constellation with the mouse, zoom, and scrub
through interrogation time -- no re-rendering required.

Examples
--------
Export the star data for the interactive viewer with the default parameters::

    python Majorana_visualizer.py

Stronger coupling and more bath spins, still exporting data for the viewer::

    python Majorana_visualizer.py --J 2 --N 16

Prepare for the default t=5, then run three sense/decode echo cycles with a
perfect decoder (``J0 = J``)::

    python Majorana_visualizer.py --sense-time 0.5 --decode-time 0.5 \
        --num-of-cycles 3 --J-estimate 1

Fall back to a rendered GIF (the previous default behaviour)::

    python Majorana_visualizer.py --no-save-star-data --save-animation \
        --animation-format gif
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from matplotlib import animation
from scipy.optimize import linear_sum_assignment

try:
    from CRB.crb_core import (
        _metadata_value,
        build_hamiltonian,
        build_spin_operators,
        central_spin_state,
        coherent_bath_state,
        qfi_from_rho_and_drho,
        save_plot,
    )
except ModuleNotFoundError:  # Allow: python CRB/Majorana_visualizer.py
    from crb_core import (
        _metadata_value,
        build_hamiltonian,
        build_spin_operators,
        central_spin_state,
        coherent_bath_state,
        qfi_from_rho_and_drho,
        save_plot,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationConfig:
    """Every physics, sampling, and rendering knob for the visualizer."""

    # --- Hamiltonian: H = Omega*sigma_x + J*sigma_z*S_z + omega*S_x ---
    N: int = 15
    Omega: float = 4.2  # central-spin drive, coefficient of X = sigma_x
    omega: float = 2.0  # collective-bath drive, coefficient of S_x
    J: float = 1.0 # central-bath coupling, coefficient of Z * S_z

    # --- Optional Markovian dissipation on the central spin ---
    # gamma weights a sqrt(gamma) * sigma_z dephasing collapse operator and
    # beta a sqrt(beta) * sigma_x relaxation one, matching crb_core's
    # convention.  Both zero -> unitary Schroedinger evolution.
    gamma: float = 0.0
    beta: float = 0.0

    # --- Initial product state |psi0> = R(theta_c, phi_c) (x) coherent_bath ---
    # Defaults reproduce |0>_central (x) |+>^N_bath.
    central_theta_rad: float = 0.0  # 0 -> |0>
    central_phi_rad: float = 0.0
    bath_theta_rad: float = np.pi / 2.0  # pi/2, phi 0 -> |+>^N (the +x state)
    bath_phi_rad: float = 0.0

    # --- Interrogation-time grid (one animation frame per sample) ---
    t_min: float = 0.0
    t_max: float = 5.0
    n_steps: int = 500

    # --- Sensing / decoding echo protocol (appended after preparation) ---
    # The t_min..t_max evolution above is the *preparation* phase, run under the
    # sensing Hamiltonian H_sense = Omega*sigma_x + J*sigma_z*S_z + omega*S_x.
    # When sense_time and/or decode_time are positive, num_of_cycles repetitions
    # of a sensing segment (H_sense, duration sense_time) then a decoding segment
    # (H_dec(J0), duration decode_time) are appended, each continuing from the
    # previous segment's final state.  Matching the echo scripts, the decoder
    # flips the single-spin and bath drives, keeps the interaction sign, and
    # swaps the estimate J0 for the true J:
    #     H_dec(J0) = -Omega*sigma_x + J0*sigma_z*S_z - omega*S_x.
    # Both sense_time and decode_time at 0 append nothing and reproduce the
    # default single-Hamiltonian animation.
    sense_time: float = 0.0
    n_sense_steps: int = 100  # frames per sensing segment (used when sense_time>0)
    decode_time: float = 0.0
    n_decode_steps: int = 100  # frames per decoding segment (used when decode_time>0)
    num_of_cycles: int = 1
    J_estimate: float = 1.0  # J0 used by the decoder H_dec (defaults to J=1)

    # --- Majorana constellation of the reduced bath ---
    # rho_b(t) is generally mixed; we draw the constellation of its
    # majorana_n_components leading eigenstates, opacity/size ~ eigenvalue.
    # 1 -> only the dominant eigenstate; 2 -> also show the fading second
    # component (the full picture in the unitary, rank-<=2 case).
    majorana_n_components: int = 2
    # Eigenstates with weight below this are not drawn (hides ~0-weight noise).
    majorana_weight_threshold: float = 5e-3
    majorana_star_size: float = 150.0  # base scatter size of a full-weight star
    majorana_star_marker: str = "*"
    majorana_star_color: str = "#f5c518"  # dominant eigenstate stars (gold)
    majorana_subdominant_color: str = "#2f8fd6"  # sub-dominant stars (blue)

    # --- dJ star-motion arrows (how the parameter J is encoded, geometrically) ---
    # Each drawn star carries an arrow along d(star)/dJ, the direction it drifts
    # on the sphere as the coupling J increases.  The field is estimated by a
    # symmetric finite difference: the bath is re-evolved at J +/- dJ_fd_step, the
    # corresponding eigenstate is tracked by maximum overlap, and its stars are
    # matched to the base constellation by nearest neighbour.  Long arrows mark
    # the stars (and hence the measurement directions) most sensitive to J.
    show_dJ_arrows: bool = True
    dJ_fd_step: float = 1e-3  # +/- step in J for the central finite difference
    # Arrows are drawn for the leading dJ_arrow_n_components eigenstate
    # constellations (capped at majorana_n_components; 1 -> dominant only).
    dJ_arrow_n_components: int = 1
    dJ_arrow_color: str = "#39e0c8"  # teal, distinct from the gold/blue stars
    # Target on-screen arrow length (in unit-sphere radii) for the 90th-percentile
    # star speed; the per-frame vectors are exported raw and the viewer/renderer
    # rescale by this so arrows stay legible without hiding relative magnitudes.
    dJ_arrow_target_len: float = 0.35

    # --- SLD eigenvector constellations (what measurement extracts the info) ---
    # The symmetric logarithmic derivative L_b of the reduced bath state (for
    # estimating J) is the optimal observable: a projective measurement in its
    # eigenbasis saturates the bath QFI, with per-outcome contribution
    # c_k = lambda_k^2 <l_k|rho_b|l_k>.  We draw the Majorana constellation of the
    # sld_n_components outcomes of largest c_k -- the states the ideal detector
    # projects onto -- so the right sphere shows both the encoding (arrows) and
    # its optimal readout (these constellations).
    show_sld_constellation: bool = True
    sld_n_components: int = 2
    sld_star_marker: str = "D"  # diamond, to read apart from the * bath stars
    sld_star_size: float = 90.0
    sld_star_color: str = "#e05fd6"  # magenta SLD-eigenvector stars
    # Outcomes contributing below this fraction of the bath QFI are not drawn.
    sld_contribution_threshold: float = 1e-2
    qfi_tol: float = 1e-12  # eigenvalue floor passed to qfi_from_rho_and_drho

    # Sphere backdrop appearance.
    sphere_color: str = "#9fb3c8"
    sphere_alpha: float = 0.10
    sphere_mesh_u: int = 60  # azimuthal resolution of the backdrop sphere
    sphere_mesh_v: int = 30  # polar resolution of the backdrop sphere

    # --- Central-spin Bloch trajectory trail (traces the Bloch-vector tip) ---
    show_trajectory: bool = True
    trajectory_length: int = 0  # 0 -> keep the full history
    trajectory_color: str = "#d62728"
    trajectory_alpha: float = 0.55

    # --- Figure / animation appearance ---
    figure_width_in: float = 13.0
    figure_height_in: float = 6.5
    figure_dpi: int = 120
    interval_ms: int = 40  # delay between frames when shown
    elevation_deg: float = 30.0
    azimuth_deg: float = -60.0

    # --- Outputs ---
    show_animation: bool = False
    # Export the per-frame star trajectory as JSON (for the drag-and-drop
    # majorana_viewer.html WebGL app) instead of, or alongside, a rendered GIF.
    # This is the default output: it is far smaller and lets you orbit the
    # constellation and scrub time interactively rather than watching a fixed GIF.
    save_star_data: bool = True
    star_data_filename: str = ""  # empty -> auto-generated from parameters
    save_animation: bool = False
    animation_format: str = "gif"  # "gif" (Pillow) or "mp4" (ffmpeg)
    animation_filename: str = ""  # empty -> auto-generated from parameters
    save_poster_frame: bool = False  # also save one frame via crb_core.save_plot


def validate_config(cfg: SimulationConfig) -> None:
    """Reject configurations that cannot produce a meaningful animation."""
    if cfg.N < 1:
        raise ValueError("N must be a positive integer")
    finite_values = (
        cfg.Omega,
        cfg.omega,
        cfg.J,
        cfg.gamma,
        cfg.beta,
        cfg.central_theta_rad,
        cfg.central_phi_rad,
        cfg.bath_theta_rad,
        cfg.bath_phi_rad,
        cfg.t_min,
        cfg.t_max,
        cfg.majorana_weight_threshold,
        cfg.majorana_star_size,
        cfg.dJ_fd_step,
        cfg.dJ_arrow_target_len,
        cfg.sld_star_size,
        cfg.sld_contribution_threshold,
        cfg.qfi_tol,
        cfg.sphere_alpha,
        cfg.figure_width_in,
        cfg.figure_height_in,
        cfg.elevation_deg,
        cfg.azimuth_deg,
    )
    if not all(np.isfinite(value) for value in finite_values):
        raise ValueError("all floating-point configuration values must be finite")
    if cfg.gamma < 0.0 or cfg.beta < 0.0:
        raise ValueError("gamma and beta must be non-negative")
    if cfg.t_min < 0.0:
        raise ValueError("t_min must be non-negative")
    if cfg.t_max <= cfg.t_min:
        raise ValueError("t_max must be greater than t_min")
    if cfg.n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if cfg.majorana_n_components < 1:
        raise ValueError("majorana_n_components must be at least 1")
    if not 0.0 <= cfg.majorana_weight_threshold < 1.0:
        raise ValueError("majorana_weight_threshold must be in [0, 1)")
    if cfg.majorana_star_size <= 0.0:
        raise ValueError("majorana_star_size must be positive")
    if cfg.dJ_fd_step <= 0.0:
        raise ValueError("dJ_fd_step must be positive")
    if cfg.dJ_arrow_n_components < 1:
        raise ValueError("dJ_arrow_n_components must be at least 1")
    if cfg.dJ_arrow_target_len <= 0.0:
        raise ValueError("dJ_arrow_target_len must be positive")
    if cfg.sld_n_components < 1:
        raise ValueError("sld_n_components must be at least 1")
    if cfg.sld_star_size <= 0.0:
        raise ValueError("sld_star_size must be positive")
    if not 0.0 <= cfg.sld_contribution_threshold < 1.0:
        raise ValueError("sld_contribution_threshold must be in [0, 1)")
    if cfg.qfi_tol <= 0.0:
        raise ValueError("qfi_tol must be positive")
    if not 0.0 <= cfg.sphere_alpha <= 1.0:
        raise ValueError("sphere_alpha must be in [0, 1]")
    if cfg.sphere_mesh_u < 4 or cfg.sphere_mesh_v < 3:
        raise ValueError("sphere_mesh_u/_v must be at least 4/3")
    if cfg.trajectory_length < 0:
        raise ValueError("trajectory_length must be non-negative")
    if cfg.figure_width_in <= 0.0 or cfg.figure_height_in <= 0.0:
        raise ValueError("figure dimensions must be positive")
    if cfg.figure_dpi <= 0:
        raise ValueError("figure_dpi must be positive")
    if cfg.interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    if cfg.animation_format.lower() not in {"gif", "mp4"}:
        raise ValueError("animation_format must be 'gif' or 'mp4'")


def time_grid(cfg: SimulationConfig) -> np.ndarray:
    """Return the uniformly sampled interrogation-time grid (one frame each)."""
    return np.linspace(cfg.t_min, cfg.t_max, cfg.n_steps)


# ---------------------------------------------------------------------------
# State preparation and evolution
# ---------------------------------------------------------------------------


def build_initial_state(cfg: SimulationConfig) -> qt.Qobj:
    """Return the joint central (x) bath product ket for ``cfg``.

    With the defaults this is ``|0>_central (x) |+>^N_bath``.
    """
    central_ket = central_spin_state(
        theta=cfg.central_theta_rad,
        phi=cfg.central_phi_rad,
    )
    bath_vector = coherent_bath_state(
        cfg.N,
        theta=cfg.bath_theta_rad,
        phi=cfg.bath_phi_rad,
    )
    bath_ket = qt.Qobj(bath_vector, dims=[[cfg.N + 1], [1]])
    return qt.tensor(central_ket, bath_ket)


def collapse_operators(cfg: SimulationConfig) -> list[qt.Qobj]:
    """Return the central-spin collapse operators implied by gamma and beta."""
    operators = build_spin_operators(cfg.N)
    c_ops: list[qt.Qobj] = []
    if cfg.beta > 0.0:
        c_ops.append(np.sqrt(cfg.beta) * operators["sx_s"])
    if cfg.gamma > 0.0:
        c_ops.append(np.sqrt(cfg.gamma) * operators["sz_s"])
    return c_ops


def evolve_states(
    cfg: SimulationConfig,
    times: np.ndarray,
    J: float | None = None,
) -> list[qt.Qobj]:
    """Return the joint state at every sampled time.

    Uses a Schroedinger solve when there is no dissipation and a master-equation
    solve otherwise; the returned objects are kets or density matrices whose
    ``ptrace`` gives the two subsystem states the animation needs.

    ``J`` overrides ``cfg.J`` for the interaction strength, leaving every other
    knob untouched; this drives the ``J +/- dJ`` re-evolutions used to estimate
    the dJ star-motion arrows and the SLD by symmetric finite difference.
    """
    hamiltonian = build_hamiltonian(
        Omega_0=cfg.Omega,
        J=cfg.J if J is None else J,
        N=cfg.N,
        omega=cfg.omega,
    )
    initial_state = build_initial_state(cfg)
    c_ops = collapse_operators(cfg)
    options = {"store_states": True}
    if c_ops:
        result = qt.mesolve(
            hamiltonian, initial_state, times, c_ops=c_ops, options=options
        )
    else:
        result = qt.sesolve(hamiltonian, initial_state, times, options=options)
    return result.states


# ---------------------------------------------------------------------------
# Majorana stellar representation
# ---------------------------------------------------------------------------


def majorana_stars(vector: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Return the ``(2S, 3)`` unit-sphere coordinates of the Majorana stars.

    ``vector`` holds the amplitudes of a pure spin-``S`` state in the
    ``|S, m>`` basis ordered ``m = +S, ..., -S`` (QuTiP's spin ordering), so it
    has length ``2S + 1``.  The stars are the ``2S`` roots of the Majorana
    polynomial mapped to the sphere by ``z = tan(theta/2) e^{i phi}``; roots at
    infinity (a leading polynomial coefficient that vanishes) sit at the south
    pole.

    Global phase and normalization are irrelevant, so eigenvectors of a density
    matrix can be passed directly.  (For a spin-coherent state all ``2S`` roots
    coincide; ``numpy.roots`` then spreads that high-multiplicity root very
    slightly -- a harmless cosmetic artifact for a visualizer.)
    """
    vector = np.asarray(vector, dtype=complex).ravel()
    two_s = vector.size - 1
    if two_s <= 0:
        return np.empty((0, 3), dtype=float)

    # Coefficients in descending powers of z (numpy.roots convention):
    #   coeff of z^{2S-k} = (-1)^k sqrt(C(2S, k)) vector[k],   k = 0 .. 2S.
    coefficients = np.array(
        [
            ((-1.0) ** k) * math.sqrt(math.comb(two_s, k)) * vector[k]
            for k in range(vector.size)
        ],
        dtype=complex,
    )

    # Leading (near-)zero coefficients are roots at infinity -> south pole.
    scale = float(np.max(np.abs(coefficients)))
    threshold = tol * scale if scale > 0.0 else 0.0
    n_at_infinity = 0
    while n_at_infinity < two_s and abs(coefficients[n_at_infinity]) <= threshold:
        n_at_infinity += 1
    trimmed = coefficients[n_at_infinity:]
    finite_roots = np.roots(trimmed) if trimmed.size >= 2 else np.empty(0, complex)

    stars = np.empty((two_s, 3), dtype=float)
    for index, z in enumerate(finite_roots):
        theta = 2.0 * np.arctan(abs(z))
        phi = float(np.angle(z))
        stars[index] = (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        )
    stars[finite_roots.size :] = (0.0, 0.0, -1.0)  # roots at infinity
    return stars


def _reduced_bath_matrix(state: qt.Qobj) -> np.ndarray:
    """Return the Hermitized reduced-bath density matrix as a NumPy array."""
    rho = np.asarray(state.ptrace(1).full(), dtype=complex)
    return 0.5 * (rho + rho.conj().T)


def _match_stars(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return ``perm`` so ``candidate[perm]`` aligns row-by-row with ``reference``.

    The Majorana stars are the unordered roots of a polynomial, so a perturbed
    constellation comes back in an arbitrary order.  Pairing each reference star
    with its nearest perturbed star (minimum total squared chord distance, via the
    Hungarian assignment) recovers the correspondence needed to difference star
    positions into a per-star velocity.
    """
    n = reference.shape[0]
    if n == 0:
        return np.empty(0, dtype=int)
    difference = reference[:, None, :] - candidate[None, :, :]
    cost = np.einsum("ijk,ijk->ij", difference, difference)
    _, columns = linear_sum_assignment(cost)
    return columns


def _tracked_eigenvector(
    base_vector: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Return the column of ``eigenvectors`` with the largest overlap onto ``base``.

    Following an eigenstate across the ``J +/- dJ`` re-evolutions by maximum
    overlap (rather than by eigenvalue rank) keeps the finite difference on *the
    same* physical branch even where two bath eigenvalues nearly cross.
    """
    overlaps = np.abs(base_vector.conj() @ eigenvectors)
    return eigenvectors[:, int(np.argmax(overlaps))]


# ---------------------------------------------------------------------------
# Reduced-state trajectories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryData:
    """Precomputed per-frame quantities that drive the animation."""

    bloch_vectors: np.ndarray  # (n_frames, 3): <sigma_x>, <sigma_y>, <sigma_z>
    central_purity: np.ndarray  # (n_frames,): Tr(rho_c^2)
    bath_purity: np.ndarray  # (n_frames,): Tr(rho_b^2)
    weights: np.ndarray  # (n_frames, K): leading eigenvalues p_i of rho_b
    stars: np.ndarray  # (n_frames, K, 2S, 3): Majorana stars per eigenstate
    n_stars: int  # 2S = N
    # --- dJ star-motion arrows (empty last axis when not computed) ---
    # star_velocities[f, c] holds d(star)/dJ for eigenstate c's 2S stars; only
    # the leading dJ_arrow_n_components components are populated, the rest zero.
    star_velocities: np.ndarray  # (n_frames, K, 2S, 3)
    arrow_display_scale: float  # suggested multiplier: raw dstar/dJ -> screen len
    # --- SLD (optimal-measurement) eigenvector constellations ---
    bath_qfi: np.ndarray  # (n_frames,): reduced-bath QFI for J (0 if not computed)
    sld_stars: np.ndarray  # (n_frames, Ksld, 2S, 3): stars of leading SLD outcomes
    sld_eigenvalues: np.ndarray  # (n_frames, Ksld): SLD eigenvalues lambda_k
    sld_contributions: np.ndarray  # (n_frames, Ksld): lambda_k^2 p_k / QFI in [0,1]
    has_dJ: bool  # whether the J +/- dJ re-evolutions were supplied


def _bath_eigenstates(rho_bath: np.ndarray, n_components: int):
    """Return the leading ``n_components`` (weight, eigenvector) pairs of ``rho_b``."""
    eigenvalues, eigenvectors = np.linalg.eigh(rho_bath)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    return eigenvalues, eigenvectors, order


def _sld_outcome_constellations(
    rho_avg: np.ndarray,
    sld: np.ndarray,
    qfi: float,
    sld_k: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the stars, eigenvalues, and QFI shares of the top SLD outcomes.

    A projective measurement in the SLD eigenbasis ``L = sum_k lambda_k
    |l_k><l_k|`` saturates the QFI, contributing ``c_k = lambda_k^2 <l_k|rho|l_k>``
    per outcome (so ``sum_k c_k = QFI``).  Ranking by ``c_k`` picks the outcomes
    the ideal detector actually reads out, whose constellations show *what* the
    optimal measurement projects onto.
    """
    lam, vecs = np.linalg.eigh(sld)
    probabilities = np.real(np.einsum("ki,kl,li->i", vecs.conj(), rho_avg, vecs))
    contributions = lam**2 * np.clip(probabilities, 0.0, None)
    ranked = np.argsort(contributions)[::-1][:sld_k]
    n_stars = rho_avg.shape[0] - 1
    stars = np.empty((sld_k, n_stars, 3), dtype=float)
    eigenvalues = np.empty(sld_k, dtype=float)
    shares = np.empty(sld_k, dtype=float)
    normaliser = qfi if qfi > tol else 0.0
    for slot, k in enumerate(ranked):
        stars[slot] = majorana_stars(vecs[:, k])
        eigenvalues[slot] = float(lam[k])
        shares[slot] = float(contributions[k] / normaliser) if normaliser else 0.0
    return stars, eigenvalues, np.clip(shares, 0.0, 1.0)


def compute_trajectories(
    cfg: SimulationConfig,
    states: list[qt.Qobj],
    states_plus: list[qt.Qobj] | None = None,
    states_minus: list[qt.Qobj] | None = None,
) -> TrajectoryData:
    """Reduce each joint state into a Bloch vector and bath constellations.

    When the ``J +/- dJ`` re-evolutions ``states_plus``/``states_minus`` are
    supplied, each frame additionally gets (a) the ``d(star)/dJ`` velocity of the
    leading eigenstate constellations -- how the coupling is geometrically encoded
    -- and (b) the reduced-bath QFI together with the Majorana constellations of
    the dominant SLD eigenvectors -- the optimal measurement that reads it out.
    """
    pauli = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    n_frames = len(states)
    n_stars = cfg.N  # 2S
    n_components = min(cfg.majorana_n_components, cfg.N + 1)

    has_dJ = states_plus is not None and states_minus is not None
    compute_arrows = has_dJ and cfg.show_dJ_arrows
    compute_sld = has_dJ and cfg.show_sld_constellation
    arrow_k = min(cfg.dJ_arrow_n_components, n_components) if compute_arrows else 0
    sld_k = min(cfg.sld_n_components, cfg.N + 1) if compute_sld else 0
    two_dJ = 2.0 * cfg.dJ_fd_step

    bloch_vectors = np.empty((n_frames, 3), dtype=float)
    central_purity = np.empty(n_frames, dtype=float)
    bath_purity = np.empty(n_frames, dtype=float)
    weights = np.zeros((n_frames, n_components), dtype=float)
    stars = np.empty((n_frames, n_components, n_stars, 3), dtype=float)
    star_velocities = np.zeros((n_frames, n_components, n_stars, 3), dtype=float)
    bath_qfi = np.zeros(n_frames, dtype=float)
    sld_stars = np.zeros((n_frames, sld_k, n_stars, 3), dtype=float)
    sld_eigenvalues = np.zeros((n_frames, sld_k), dtype=float)
    sld_contributions = np.zeros((n_frames, sld_k), dtype=float)

    for index, state in enumerate(states):
        rho_central = state.ptrace(0)
        bloch_vectors[index] = [
            float(np.real(qt.expect(operator, rho_central))) for operator in pauli
        ]
        central_purity[index] = float(np.real((rho_central * rho_central).tr()))

        rho_bath = _reduced_bath_matrix(state)
        bath_purity[index] = float(np.real(np.trace(rho_bath @ rho_bath)))

        # Leading eigenstates of the (generally mixed) reduced bath state.
        eigenvalues, eigenvectors, order = _bath_eigenstates(rho_bath, n_components)
        for component, eig_index in enumerate(order):
            weights[index, component] = max(float(eigenvalues[eig_index]), 0.0)
            stars[index, component] = majorana_stars(eigenvectors[:, eig_index])

        if not has_dJ:
            continue

        rho_bath_plus = _reduced_bath_matrix(states_plus[index])
        rho_bath_minus = _reduced_bath_matrix(states_minus[index])

        # (a) d(star)/dJ: track each leading eigenstate onto its J +/- dJ branch
        # by maximum overlap, then match the perturbed stars to the base ones.
        if compute_arrows:
            _, evecs_plus, _ = _bath_eigenstates(rho_bath_plus, cfg.N + 1)
            _, evecs_minus, _ = _bath_eigenstates(rho_bath_minus, cfg.N + 1)
            for component in range(arrow_k):
                if weights[index, component] < cfg.majorana_weight_threshold:
                    continue
                base_vector = eigenvectors[:, order[component]]
                stars_plus = majorana_stars(
                    _tracked_eigenvector(base_vector, evecs_plus)
                )
                stars_minus = majorana_stars(
                    _tracked_eigenvector(base_vector, evecs_minus)
                )
                reference = stars[index, component]
                stars_plus = stars_plus[_match_stars(reference, stars_plus)]
                stars_minus = stars_minus[_match_stars(reference, stars_minus)]
                star_velocities[index, component] = (
                    stars_plus - stars_minus
                ) / two_dJ

        # (b) reduced-bath QFI and the constellations of the optimal (SLD) readout.
        rho_avg = 0.5 * (rho_bath_plus + rho_bath_minus)
        drho = (rho_bath_plus - rho_bath_minus) / two_dJ
        qfi, sld = qfi_from_rho_and_drho(rho_avg, drho, tol=cfg.qfi_tol)
        bath_qfi[index] = qfi
        if compute_sld:
            (
                sld_stars[index],
                sld_eigenvalues[index],
                sld_contributions[index],
            ) = _sld_outcome_constellations(rho_avg, sld, qfi, sld_k, cfg.qfi_tol)

    arrow_display_scale = (
        _arrow_display_scale(star_velocities, cfg) if compute_arrows else 1.0
    )

    return TrajectoryData(
        bloch_vectors=bloch_vectors,
        central_purity=central_purity,
        bath_purity=bath_purity,
        weights=weights,
        stars=stars,
        n_stars=n_stars,
        star_velocities=star_velocities,
        arrow_display_scale=arrow_display_scale,
        bath_qfi=bath_qfi,
        sld_stars=sld_stars,
        sld_eigenvalues=sld_eigenvalues,
        sld_contributions=sld_contributions,
        has_dJ=has_dJ,
    )


def _arrow_display_scale(
    star_velocities: np.ndarray,
    cfg: SimulationConfig,
) -> float:
    """Return a multiplier mapping raw ``d(star)/dJ`` to a legible screen length.

    Scaling so the 90th-percentile star speed reaches ``dJ_arrow_target_len``
    (in unit-sphere radii) keeps the fastest arrows on-sphere while preserving the
    *relative* lengths that flag the most J-sensitive stars.
    """
    speeds = np.linalg.norm(star_velocities, axis=-1)
    speeds = speeds[speeds > 0.0]
    if speeds.size == 0:
        return 1.0
    reference_speed = float(np.percentile(speeds, 90.0))
    if reference_speed <= 0.0:
        reference_speed = float(np.max(speeds))
    if reference_speed <= 0.0:
        return 1.0
    return cfg.dJ_arrow_target_len / reference_speed


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def _hamiltonian_title(cfg: SimulationConfig) -> str:
    """Return a LaTeX banner describing the Hamiltonian and its parameters."""
    return (
        r"$H=\Omega\,\sigma_x+J\,\sigma_z S_z+\omega\,S_x$   "
        rf"$N={cfg.N}$, $\Omega={cfg.Omega:g}$, $\omega={cfg.omega:g}$, "
        rf"$J={cfg.J:g}$"
    )


def _sphere_backdrop(cfg: SimulationConfig) -> dict[str, np.ndarray]:
    """Precompute the unit-sphere surface and reference circles (drawn each frame)."""
    u = np.linspace(0.0, 2.0 * np.pi, cfg.sphere_mesh_u)
    v = np.linspace(0.0, np.pi, cfg.sphere_mesh_v)
    surface_x = np.outer(np.cos(u), np.sin(v))
    surface_y = np.outer(np.sin(u), np.sin(v))
    surface_z = np.outer(np.ones_like(u), np.cos(v))
    circle = np.linspace(0.0, 2.0 * np.pi, 120)
    return {
        "surface_x": surface_x,
        "surface_y": surface_y,
        "surface_z": surface_z,
        "circle_cos": np.cos(circle),
        "circle_sin": np.sin(circle),
        "circle_zero": np.zeros_like(circle),
    }


def _weight_to_alpha(weight: float) -> float:
    """Map an eigenvalue weight in [0, 1] to a legible marker opacity."""
    return 0.25 + 0.70 * float(np.clip(weight, 0.0, 1.0))


def build_animation(
    cfg: SimulationConfig,
    times: np.ndarray,
    trajectory: TrajectoryData,
) -> tuple[plt.Figure, animation.FuncAnimation, Callable[[int], object]]:
    """Assemble the two-sphere figure, its animation, and its frame renderer.

    The frame renderer is returned so callers (e.g. a poster-frame export) can
    draw an arbitrary frame without reaching into private animation state.
    """
    figure = plt.figure(figsize=(cfg.figure_width_in, cfg.figure_height_in))
    central_axis = figure.add_subplot(1, 2, 1, projection="3d")
    bath_axis = figure.add_subplot(1, 2, 2, projection="3d")

    bloch = qt.Bloch(fig=figure, axes=central_axis)
    bloch.view = [cfg.azimuth_deg, cfg.elevation_deg]
    bloch.vector_color = ["#2ca02c"]
    bloch.vector_width = 4
    bloch.point_color = ["#1f77b4"]
    bloch.point_marker = ["o"]
    bloch.point_size = [45]

    backdrop = _sphere_backdrop(cfg)
    n_components = trajectory.weights.shape[1]

    def draw_central(frame: int) -> None:
        bloch.clear()
        # Faint history line so the tip's path reads as clearly subordinate to
        # the bold current vector (both live in the Bloch [-1, 1] coordinates).
        if cfg.show_trajectory and frame >= 1:
            start = (
                0
                if cfg.trajectory_length == 0
                else max(0, frame - cfg.trajectory_length)
            )
            trail = trajectory.bloch_vectors[start : frame + 1]
            bloch.add_points(
                [trail[:, 0], trail[:, 1], trail[:, 2]],
                meth="l",
                colors=[cfg.trajectory_color],
                alpha=cfg.trajectory_alpha,
            )
        current = trajectory.bloch_vectors[frame]
        bloch.add_vectors(current.tolist())
        bloch.add_points([[current[0]], [current[1]], [current[2]]], meth="s")
        bloch.render()
        central_axis.set_title(
            f"Central spin\npurity Tr$(\\rho_c^2)={trajectory.central_purity[frame]:.3f}$",
            fontsize=12,
        )

    def draw_bath_backdrop() -> None:
        bath_axis.plot_surface(
            backdrop["surface_x"],
            backdrop["surface_y"],
            backdrop["surface_z"],
            color=cfg.sphere_color,
            alpha=cfg.sphere_alpha,
            linewidth=0.0,
            antialiased=False,
            shade=False,
            zorder=0,
        )
        cos_t = backdrop["circle_cos"]
        sin_t = backdrop["circle_sin"]
        zero = backdrop["circle_zero"]
        for xs, ys, zs in (
            (cos_t, sin_t, zero),  # equator (z = 0)
            (cos_t, zero, sin_t),  # meridian in the x-z plane
            (zero, cos_t, sin_t),  # meridian in the y-z plane
        ):
            bath_axis.plot(xs, ys, zs, color="#c4ccd4", linewidth=0.8, alpha=0.9)
        for axis_vector, label in (
            ((1.0, 0.0, 0.0), "x"),
            ((0.0, 1.0, 0.0), "y"),
            ((0.0, 0.0, 1.0), "z"),
        ):
            bath_axis.plot(
                [-axis_vector[0], axis_vector[0]],
                [-axis_vector[1], axis_vector[1]],
                [-axis_vector[2], axis_vector[2]],
                color="#8894a3",
                linewidth=0.8,
                alpha=0.7,
            )
            bath_axis.text(
                1.18 * axis_vector[0],
                1.18 * axis_vector[1],
                1.18 * axis_vector[2],
                label,
                color="#55606e",
                fontsize=11,
                ha="center",
                va="center",
            )

    def draw_bath(frame: int) -> None:
        bath_axis.clear()
        draw_bath_backdrop()

        weights = trajectory.weights[frame]
        stars = trajectory.stars[frame]
        # Draw sub-dominant components first so the dominant stars sit on top.
        for component in range(n_components - 1, -1, -1):
            weight = float(weights[component])
            if weight < cfg.majorana_weight_threshold:
                continue
            color = (
                cfg.majorana_star_color
                if component == 0
                else cfg.majorana_subdominant_color
            )
            size = cfg.majorana_star_size * (0.45 + 0.55 * min(weight, 1.0))
            points = stars[component]
            bath_axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                marker=cfg.majorana_star_marker,
                s=size,
                c=color,
                edgecolors="black",
                linewidths=0.5,
                alpha=_weight_to_alpha(weight),
                depthshade=False,
                zorder=5,
            )

        # dJ star-motion arrows: d(star)/dJ, scaled to a legible screen length.
        if cfg.show_dJ_arrows and trajectory.has_dJ:
            arrow_scale = trajectory.arrow_display_scale
            arrow_k = min(cfg.dJ_arrow_n_components, n_components)
            for component in range(arrow_k):
                if float(weights[component]) < cfg.majorana_weight_threshold:
                    continue
                base = trajectory.stars[frame, component]
                velocity = trajectory.star_velocities[frame, component] * arrow_scale
                bath_axis.quiver(
                    base[:, 0], base[:, 1], base[:, 2],
                    velocity[:, 0], velocity[:, 1], velocity[:, 2],
                    color=cfg.dJ_arrow_color,
                    linewidth=1.4,
                    arrow_length_ratio=0.35,
                    alpha=0.9,
                    zorder=6,
                )

        # Optimal-measurement (SLD eigenvector) constellations.
        if cfg.show_sld_constellation and trajectory.sld_stars.shape[1] > 0:
            sld_stars = trajectory.sld_stars[frame]
            sld_shares = trajectory.sld_contributions[frame]
            for slot in range(sld_stars.shape[0] - 1, -1, -1):
                share = float(sld_shares[slot])
                if share < cfg.sld_contribution_threshold:
                    continue
                sld_points = sld_stars[slot]
                bath_axis.scatter(
                    sld_points[:, 0], sld_points[:, 1], sld_points[:, 2],
                    marker=cfg.sld_star_marker,
                    s=cfg.sld_star_size * (0.5 + 0.5 * min(share, 1.0)),
                    c=cfg.sld_star_color,
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=_weight_to_alpha(share),
                    depthshade=False,
                    zorder=7,
                )

        bath_axis.set_box_aspect((1.0, 1.0, 1.0))
        # Match qt.Bloch's framing (limits +/-0.7 with a unit sphere) so the two
        # spheres render at the same visual size; unit-radius stars stay fully
        # visible, exactly as the radius-1 Bloch points do on the left.
        bath_axis.set_xlim(-0.7, 0.7)
        bath_axis.set_ylim(-0.7, 0.7)
        bath_axis.set_zlim(-0.7, 0.7)
        bath_axis.view_init(elev=cfg.elevation_deg, azim=cfg.azimuth_deg)
        bath_axis.set_axis_off()
        dominant_weight = float(weights[0])
        qfi_note = (
            rf", $F_Q={trajectory.bath_qfi[frame]:.3g}$"
            if trajectory.has_dJ
            else ""
        )
        bath_axis.set_title(
            "Collective bath (Majorana stars)\n"
            f"$N={cfg.N}$ ($2S={trajectory.n_stars}$ stars), "
            f"Tr$(\\rho_b^2)={trajectory.bath_purity[frame]:.3f}$, "
            f"$p_0={dominant_weight:.3f}${qfi_note}",
            fontsize=12,
        )

    banner = _hamiltonian_title(cfg)

    def update(frame: int):
        draw_central(frame)
        draw_bath(frame)
        figure.suptitle(f"{banner}\n$t={times[frame]:.2f}$", fontsize=13)
        return ()

    # A short caption explains the star coloring (opacity/size encode weight)
    # and, when present, the two J-metrology overlays.
    caption = (
        "Majorana stars of $\\rho_b=\\sum_i p_i|\\chi_i\\rangle\\langle\\chi_i|$   "
        "—   gold: dominant eigenstate   ·   "
        "blue: sub-dominant   (opacity & size $\\propto p_i$)"
    )
    if trajectory.has_dJ and cfg.show_dJ_arrows:
        caption += (
            "\nteal arrows: $\\partial\\,\\mathrm{star}/\\partial J$ "
            "(how $J$ is encoded)"
        )
    if trajectory.has_dJ and cfg.show_sld_constellation:
        caption += (
            "   ·   magenta diamonds: SLD eigenvectors "
            "(optimal $J$ measurement)"
        )
    figure.text(
        0.5,
        0.015,
        caption,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#44505e",
    )

    figure.subplots_adjust(
        left=0.0, right=1.0, top=0.86, bottom=0.06, wspace=0.02
    )
    anim = animation.FuncAnimation(
        figure,
        update,
        frames=len(times),
        interval=cfg.interval_ms,
        blit=False,
    )
    return figure, anim, update


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _format_number(value: float | int) -> str:
    """Format a value compactly and deterministically for a filename."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4g}".replace(".", "p").replace("-", "m")


def _parameter_tag(cfg: SimulationConfig) -> str:
    """Return the compact parameter stamp shared by every auto-named output."""
    return (
        f"N{cfg.N}"
        f"_Om{_format_number(cfg.Omega)}"
        f"_om{_format_number(cfg.omega)}"
        f"_J{_format_number(cfg.J)}"
        f"_t{_format_number(cfg.t_max)}_n{cfg.n_steps}"
    )


def _output_directory() -> Path:
    """Return (creating if needed) the local ``graphs`` output directory."""
    directory = Path(__file__).resolve().parent / "graphs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def animation_output_path(cfg: SimulationConfig) -> Path:
    """Return the local file path for a saved animation."""
    directory = _output_directory()
    if cfg.animation_filename:
        name = cfg.animation_filename
        if not Path(name).suffix:
            name = f"{name}.{cfg.animation_format.lower()}"
        return directory / Path(name).name
    tag = _parameter_tag(cfg)
    return directory / f"majorana_visualizer__{tag}.{cfg.animation_format.lower()}"


def star_data_output_path(cfg: SimulationConfig) -> Path:
    """Return the local file path for the exported star-trajectory JSON."""
    directory = _output_directory()
    if cfg.star_data_filename:
        name = cfg.star_data_filename
        if not Path(name).suffix:
            name = f"{name}.json"
        return directory / Path(name).name
    return directory / f"majorana_stars__{_parameter_tag(cfg)}.json"


def _rounded(array: np.ndarray, decimals: int) -> list:
    """Return ``array`` as rounded, JSON-serializable nested Python lists.

    Star coordinates live on the unit sphere and weights/purities in ``[0, 1]``,
    so a handful of decimals is visually exact while keeping the payload small.
    """
    return np.round(np.asarray(array, dtype=float), decimals).tolist()


def export_star_data(
    cfg: SimulationConfig,
    times: np.ndarray,
    trajectory: TrajectoryData,
) -> Path:
    """Write the per-frame Majorana-star trajectory to a compact JSON file.

    The schema is consumed by the companion ``majorana_viewer.html`` WebGL app:
    drag the resulting file onto that page to orbit the constellation and scrub
    through interrogation time.  Everything needed to reproduce the animation
    frames -- the star positions, their eigenstate weights, the central-spin
    Bloch trajectory, and the two purities -- is stored per frame, alongside the
    full :class:`SimulationConfig` so the export is self-describing.
    """
    path = star_data_output_path(cfg)
    payload = {
        "schema": "majorana-visualizer/1",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": str(Path(__file__).resolve()),
        "hamiltonian": "H = Omega*sigma_x + J*sigma_z*S_z + omega*S_x",
        "config": _metadata_value(cfg),
        "n_frames": int(len(times)),
        "n_stars": int(trajectory.n_stars),
        "n_components": int(trajectory.weights.shape[1]),
        # stars: (n_frames, n_components, n_stars, 3) unit vectors on the sphere.
        "times": _rounded(times, 6),
        "stars": _rounded(trajectory.stars, 5),
        "weights": _rounded(trajectory.weights, 6),
        "central_bloch_vectors": _rounded(trajectory.bloch_vectors, 6),
        "central_purity": _rounded(trajectory.central_purity, 6),
        "bath_purity": _rounded(trajectory.bath_purity, 6),
    }
    if trajectory.has_dJ:
        # d(star)/dJ arrows and the optimal-measurement (SLD) constellations; the
        # viewer scales the raw velocities by arrow_display_scale before drawing.
        payload["has_dJ"] = True
        payload["bath_qfi"] = _rounded(trajectory.bath_qfi, 6)
        if cfg.show_dJ_arrows:
            payload["star_velocities"] = _rounded(trajectory.star_velocities, 5)
            payload["arrow_display_scale"] = round(
                float(trajectory.arrow_display_scale), 6
            )
            payload["arrow_target_len"] = round(float(cfg.dJ_arrow_target_len), 6)
            payload["arrow_color"] = cfg.dJ_arrow_color
            payload["dJ_arrow_n_components"] = int(
                min(cfg.dJ_arrow_n_components, trajectory.weights.shape[1])
            )
        if cfg.show_sld_constellation and trajectory.sld_stars.shape[1] > 0:
            payload["n_sld_components"] = int(trajectory.sld_stars.shape[1])
            payload["sld_stars"] = _rounded(trajectory.sld_stars, 5)
            payload["sld_eigenvalues"] = _rounded(trajectory.sld_eigenvalues, 5)
            payload["sld_contributions"] = _rounded(
                trajectory.sld_contributions, 5
            )
            payload["sld_color"] = cfg.sld_star_color
            payload["sld_contribution_threshold"] = round(
                float(cfg.sld_contribution_threshold), 6
            )
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"))
    return path


def save_animation(
    cfg: SimulationConfig,
    anim: animation.FuncAnimation,
) -> Path:
    """Write the animation to disk, choosing a writer from the format."""
    path = animation_output_path(cfg)
    fps = max(1, round(1000.0 / cfg.interval_ms))
    if cfg.animation_format.lower() == "gif":
        writer: animation.AbstractMovieWriter = animation.PillowWriter(fps=fps)
    else:
        if not animation.FFMpegWriter.isAvailable():
            raise RuntimeError(
                "saving an mp4 requires ffmpeg on PATH; install it or use "
                "--animation-format gif"
            )
        writer = animation.FFMpegWriter(fps=fps)
    anim.save(str(path), writer=writer, dpi=cfg.figure_dpi)
    return path


def save_poster_frame(
    cfg: SimulationConfig,
    figure: plt.Figure,
    times: np.ndarray,
    trajectory: TrajectoryData,
) -> Path:
    """Save the final rendered frame via the shared metadata-aware saver."""
    filename = animation_output_path(cfg).with_suffix(".png").name
    return save_plot(
        figure,
        filename,
        metadata={
            "config": cfg,
            "time_values": times,
            "central_bloch_vectors": trajectory.bloch_vectors,
            "central_purity": trajectory.central_purity,
            "bath_purity": trajectory.bath_purity,
            "majorana_weights": trajectory.weights,
            **(
                {"bath_qfi": trajectory.bath_qfi}
                if trajectory.has_dJ
                else {}
            ),
        },
        script_path=__file__,
        dpi=cfg.figure_dpi,
    )


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def parse_config(argv: list[str] | None = None) -> SimulationConfig:
    """Build a configuration, letting the CLI override any dataclass field."""
    defaults = SimulationConfig()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for field in fields(defaults):
        default = getattr(defaults, field.name)
        flag = "--" + field.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(
                flag,
                dest=field.name,
                default=default,
                action=argparse.BooleanOptionalAction,
            )
        elif isinstance(default, int):
            parser.add_argument(flag, dest=field.name, type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(flag, dest=field.name, type=float, default=default)
        else:
            parser.add_argument(flag, dest=field.name, type=str, default=default)
    return SimulationConfig(**vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> None:
    """Run the visualizer: evolve, reduce, animate, and show or save."""
    cfg = parse_config(argv)
    validate_config(cfg)

    times = time_grid(cfg)
    states = evolve_states(cfg, times)

    # The dJ arrows and SLD constellations both need the state's J-sensitivity,
    # estimated by re-evolving at J +/- dJ_fd_step (a symmetric finite
    # difference); skip that extra work when neither overlay is requested.
    if cfg.show_dJ_arrows or cfg.show_sld_constellation:
        states_plus = evolve_states(cfg, times, J=cfg.J + cfg.dJ_fd_step)
        states_minus = evolve_states(cfg, times, J=cfg.J - cfg.dJ_fd_step)
    else:
        states_plus = states_minus = None
    trajectory = compute_trajectories(cfg, states, states_plus, states_minus)

    if cfg.save_star_data:
        data_path = export_star_data(cfg, times, trajectory)
        print(f"Saved star data to {data_path}")

    # Assembling the matplotlib animation is only worthwhile for the rendered
    # outputs; the JSON export above needs nothing but the reduced trajectory.
    needs_figure = (
        cfg.show_animation or cfg.save_animation or cfg.save_poster_frame
    )
    if needs_figure:
        figure, anim, render_frame = build_animation(cfg, times, trajectory)
        if cfg.save_animation:
            path = save_animation(cfg, anim)
            print(f"Saved animation to {path}")
        if cfg.save_poster_frame:
            render_frame(len(times) - 1)  # draw the final frame onto the figure
            poster_path = save_poster_frame(cfg, figure, times, trajectory)
            print(f"Saved poster frame to {poster_path}")
        if cfg.show_animation:
            plt.show()
        plt.close(figure)

    if not (needs_figure or cfg.save_star_data):
        print(
            "Nothing to do: enable --save-star-data, --show-animation, "
            "--save-animation, or --save-poster-frame."
        )


if __name__ == "__main__":
    main()
