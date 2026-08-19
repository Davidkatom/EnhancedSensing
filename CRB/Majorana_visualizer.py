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

Every parameter is exposed through :class:`SimulationConfig` (and mirrored on
the command line), matching the other plot scripts in this folder.  Optional
Markovian dephasing/relaxation on the central spin is available through
``gamma`` and ``beta``; leaving both at zero evolves the pure state unitarily.

Examples
--------
Show the animation with the default parameters::

    python Majorana_visualizer.py

Stronger coupling, more bath spins, and save a GIF without displaying it::

    python Majorana_visualizer.py --J 2 --N 16 --no-show-animation \
        --save-animation --animation-format gif
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from matplotlib import animation

try:
    from CRB.crb_core import (
        build_hamiltonian,
        build_spin_operators,
        central_spin_state,
        coherent_bath_state,
        save_plot,
    )
except ModuleNotFoundError:  # Allow: python CRB/Majorana_visualizer.py
    from crb_core import (
        build_hamiltonian,
        build_spin_operators,
        central_spin_state,
        coherent_bath_state,
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
    Omega: float = 2.5  # central-spin drive, coefficient of X = sigma_x
    omega: float = 1.0  # collective-bath drive, coefficient of S_x
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
    save_animation: bool = True
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
) -> list[qt.Qobj]:
    """Return the joint state at every sampled time.

    Uses a Schroedinger solve when there is no dissipation and a master-equation
    solve otherwise; the returned objects are kets or density matrices whose
    ``ptrace`` gives the two subsystem states the animation needs.
    """
    hamiltonian = build_hamiltonian(
        Omega_0=cfg.Omega,
        J=cfg.J,
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


def compute_trajectories(
    cfg: SimulationConfig,
    states: list[qt.Qobj],
) -> TrajectoryData:
    """Reduce each joint state into a Bloch vector and bath constellations."""
    pauli = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    n_frames = len(states)
    n_stars = cfg.N  # 2S
    n_components = min(cfg.majorana_n_components, cfg.N + 1)

    bloch_vectors = np.empty((n_frames, 3), dtype=float)
    central_purity = np.empty(n_frames, dtype=float)
    bath_purity = np.empty(n_frames, dtype=float)
    weights = np.zeros((n_frames, n_components), dtype=float)
    stars = np.empty((n_frames, n_components, n_stars, 3), dtype=float)

    for index, state in enumerate(states):
        rho_central = state.ptrace(0)
        bloch_vectors[index] = [
            float(np.real(qt.expect(operator, rho_central))) for operator in pauli
        ]
        central_purity[index] = float(np.real((rho_central * rho_central).tr()))

        rho_bath = np.asarray(state.ptrace(1).full(), dtype=complex)
        rho_bath = 0.5 * (rho_bath + rho_bath.conj().T)
        bath_purity[index] = float(np.real(np.trace(rho_bath @ rho_bath)))

        # Leading eigenstates of the (generally mixed) reduced bath state.
        eigenvalues, eigenvectors = np.linalg.eigh(rho_bath)
        order = np.argsort(eigenvalues)[::-1][:n_components]
        for component, eig_index in enumerate(order):
            weights[index, component] = max(float(eigenvalues[eig_index]), 0.0)
            stars[index, component] = majorana_stars(eigenvectors[:, eig_index])

    return TrajectoryData(
        bloch_vectors=bloch_vectors,
        central_purity=central_purity,
        bath_purity=bath_purity,
        weights=weights,
        stars=stars,
        n_stars=n_stars,
    )


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
        bath_axis.set_title(
            "Collective bath (Majorana stars)\n"
            f"$N={cfg.N}$ ($2S={trajectory.n_stars}$ stars), "
            f"Tr$(\\rho_b^2)={trajectory.bath_purity[frame]:.3f}$, "
            f"$p_0={dominant_weight:.3f}$",
            fontsize=12,
        )

    banner = _hamiltonian_title(cfg)

    def update(frame: int):
        draw_central(frame)
        draw_bath(frame)
        figure.suptitle(f"{banner}\n$t={times[frame]:.2f}$", fontsize=13)
        return ()

    # A short caption explains the star coloring (opacity/size encode weight).
    figure.text(
        0.5,
        0.015,
        "Majorana stars of $\\rho_b=\\sum_i p_i|\\chi_i\\rangle\\langle\\chi_i|$   "
        "—   gold: dominant eigenstate   ·   "
        "blue: sub-dominant   (opacity & size $\\propto p_i$)",
        ha="center",
        va="bottom",
        fontsize=10,
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


def animation_output_path(cfg: SimulationConfig) -> Path:
    """Return the local file path for a saved animation."""
    directory = Path(__file__).resolve().parent / "graphs"
    directory.mkdir(parents=True, exist_ok=True)
    if cfg.animation_filename:
        name = cfg.animation_filename
        if not Path(name).suffix:
            name = f"{name}.{cfg.animation_format.lower()}"
        return directory / Path(name).name
    tag = (
        f"N{cfg.N}"
        f"_Om{_format_number(cfg.Omega)}"
        f"_om{_format_number(cfg.omega)}"
        f"_J{_format_number(cfg.J)}"
        f"_t{_format_number(cfg.t_max)}_n{cfg.n_steps}"
    )
    return directory / f"majorana_visualizer__{tag}.{cfg.animation_format.lower()}"


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
    trajectory = compute_trajectories(cfg, states)
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
    if not (cfg.show_animation or cfg.save_animation or cfg.save_poster_frame):
        print(
            "Nothing to do: enable --show-animation, --save-animation, or "
            "--save-poster-frame."
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
