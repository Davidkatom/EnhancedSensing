"""Animate the driven central-spin model as two spheres side by side.

The model is the same driven central-spin Hamiltonian used throughout this
project (see :func:`CRB.crb_core.build_hamiltonian`):

    H = Omega * sigma_x  +  J * sigma_z * S_z  +  omega * S_x
      = Omega * X        +  J * Z * S_z        +  omega * S_x ,

with central-spin Pauli operators ``X = sigma_x``, ``Z = sigma_z`` and
collective-bath operators ``S_i = 2 * jmat(N / 2, i)``.  In the notation of the
request this reads ``H = omega * S_x + Omega * X + J * S_z * Z``, i.e.

    * ``Omega`` drives the central spin      (coefficient of ``X = sigma_x``),
    * ``omega`` drives the collective bath   (coefficient of ``S_x``),
    * ``J``     couples them                  (coefficient of ``Z * S_z``).

The initial state is ``|psi(0)> = |0>_central (x) |+>^N_bath`` -- the central
spin in ``|0>`` and every bath spin in ``|+>`` (the ``+x`` spin-coherent state).

The animation shows, at each interrogation time ``t``:

    * left  -- the Bloch sphere of the reduced central spin
               ``rho_c(t) = Tr_bath |psi(t)><psi(t)|``.  Because the central
               spin entangles with the bath, its Bloch vector shrinks *inside*
               the sphere (the length is set by the state purity), which is the
               dephasing this model exploits.
    * right -- the Husimi-Q quasiprobability of the reduced collective bath
               ``rho_b(t) = Tr_central |psi(t)><psi(t)|`` painted on the spin
               sphere.

Every parameter is exposed through :class:`SimulationConfig` (and mirrored on
the command line), matching the other plot scripts in this folder.  Optional
Markovian dephasing/relaxation on the central spin is available through
``gamma`` and ``beta``; leaving both at zero evolves the pure state unitarily.

Examples
--------
Show the animation with the default parameters::

    python state_visualizer.py

Stronger coupling, more bath spins, and save a GIF without displaying it::

    python state_visualizer.py --J 2 --N 16 --no-show-animation \
        --save-animation --animation-format gif
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from matplotlib import animation, cm
from matplotlib.colors import Normalize

try:
    from CRB.crb_core import (
        build_hamiltonian,
        build_spin_operators,
        central_spin_state,
        coherent_bath_state,
        save_plot,
    )
except ModuleNotFoundError:  # Allow: python CRB/state_visualizer.py
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
    J: float = 1.0  # central-bath coupling, coefficient of Z * S_z

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

    # --- Husimi-Q sphere sampling and coloring ---
    husimi_n_theta: int = 40
    husimi_n_phi: int = 80
    husimi_cmap: str = "viridis"
    # False -> one color scale shared by every frame (peaks that grow/shrink
    # stay comparable); True -> rescale each frame to its own maximum.
    husimi_normalize_per_frame: bool = False

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
    save_animation: bool = True  # write the animation to graphs/ on every run
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
    if cfg.husimi_n_theta < 2 or cfg.husimi_n_phi < 2:
        raise ValueError("husimi_n_theta and husimi_n_phi must be at least 2")
    if cfg.husimi_cmap not in matplotlib.colormaps:
        raise ValueError(f"unknown colormap: {cfg.husimi_cmap!r}")
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
# Reduced-state trajectories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryData:
    """Precomputed per-frame quantities that drive the animation."""

    bloch_vectors: np.ndarray  # (n_frames, 3): <sigma_x>, <sigma_y>, <sigma_z>
    central_purity: np.ndarray  # (n_frames,): Tr(rho_c^2)
    husimi_frames: np.ndarray  # (n_frames, n_phi, n_theta): Husimi Q values
    sphere_x: np.ndarray  # (n_phi, n_theta) unit-sphere coordinates
    sphere_y: np.ndarray
    sphere_z: np.ndarray
    husimi_vmax: float  # global Husimi peak (for a shared color scale)


def compute_trajectories(
    cfg: SimulationConfig,
    states: list[qt.Qobj],
) -> TrajectoryData:
    """Reduce each joint state into a Bloch vector and a Husimi-Q grid."""
    theta = np.linspace(0.0, np.pi, cfg.husimi_n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, cfg.husimi_n_phi)

    pauli = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    n_frames = len(states)
    bloch_vectors = np.empty((n_frames, 3), dtype=float)
    central_purity = np.empty(n_frames, dtype=float)
    husimi_frames = np.empty(
        (n_frames, cfg.husimi_n_phi, cfg.husimi_n_theta),
        dtype=float,
    )
    sphere_theta = None
    sphere_phi = None

    for index, state in enumerate(states):
        rho_central = state.ptrace(0)
        bloch_vectors[index] = [
            float(np.real(qt.expect(operator, rho_central))) for operator in pauli
        ]
        central_purity[index] = float(np.real((rho_central * rho_central).tr()))

        rho_bath = state.ptrace(1)
        husimi, sphere_theta, sphere_phi = qt.spin_q_function(rho_bath, theta, phi)
        husimi_frames[index] = np.real(husimi)

    sphere_x = np.sin(sphere_theta) * np.cos(sphere_phi)
    sphere_y = np.sin(sphere_theta) * np.sin(sphere_phi)
    sphere_z = np.cos(sphere_theta)
    husimi_vmax = float(husimi_frames.max())
    return TrajectoryData(
        bloch_vectors=bloch_vectors,
        central_purity=central_purity,
        husimi_frames=husimi_frames,
        sphere_x=sphere_x,
        sphere_y=sphere_y,
        sphere_z=sphere_z,
        husimi_vmax=husimi_vmax,
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

    cmap = matplotlib.colormaps[cfg.husimi_cmap]
    shared_norm = Normalize(vmin=0.0, vmax=max(trajectory.husimi_vmax, 1e-12))

    # A dedicated slim colorbar axis keeps the Husimi sphere the same visual
    # size as the Bloch sphere instead of stealing width from it.
    colorbar_axis = figure.add_axes([0.9, 0.28, 0.015, 0.44])
    figure.colorbar(
        cm.ScalarMappable(norm=shared_norm, cmap=cmap),
        cax=colorbar_axis,
    ).set_label("Husimi $Q(\\theta,\\phi)$")

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

    def husimi_facecolors(frame: int) -> np.ndarray:
        husimi = trajectory.husimi_frames[frame]
        if cfg.husimi_normalize_per_frame:
            norm = Normalize(vmin=0.0, vmax=max(float(husimi.max()), 1e-12))
        else:
            norm = shared_norm
        return cmap(norm(husimi))

    # Build the Husimi sphere once and, per frame, only recolor its faces.
    # Rebuilding the surface geometry every frame is ~5x slower and dominates
    # playback, so the persistent-surface recolor keeps the animation smooth.
    bath_surface = bath_axis.plot_surface(
        trajectory.sphere_x,
        trajectory.sphere_y,
        trajectory.sphere_z,
        rcount=cfg.husimi_n_phi,
        ccount=cfg.husimi_n_theta,
        facecolors=husimi_facecolors(0),
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )
    bath_axis.set_box_aspect((1.0, 1.0, 1.0))
    bath_axis.set_xlim(-0.7, 0.7)
    bath_axis.set_ylim(-0.7, 0.7)
    bath_axis.set_zlim(-0.7, 0.7)
    bath_axis.view_init(elev=cfg.elevation_deg, azim=cfg.azimuth_deg)
    bath_axis.set_axis_off()
    bath_axis.set_title(
        f"Collective bath (Husimi $Q$)\n$N={cfg.N}$ spins, $S={cfg.N / 2:g}$",
        fontsize=12,
    )

    def draw_bath(frame: int) -> None:
        # facecolors has shape (n_phi, n_theta, 4); the surface has one face per
        # interior grid cell, i.e. the (n_phi-1) x (n_theta-1) leading block.
        colors = husimi_facecolors(frame)
        bath_surface.set_facecolors(colors[:-1, :-1].reshape(-1, 4))

    banner = _hamiltonian_title(cfg)

    def update(frame: int):
        draw_central(frame)
        draw_bath(frame)
        figure.suptitle(f"{banner}\n$t={times[frame]:.2f}$", fontsize=13)
        return ()

    figure.subplots_adjust(left=0.0, right=0.89, top=0.86, bottom=0.02, wspace=0.02)
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
    return directory / f"state_visualizer__{tag}.{cfg.animation_format.lower()}"


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

    total = int(cfg.n_steps)
    print(f"Writing {total} frames to {path} ...", flush=True)

    def report(current: int, _total: int) -> None:
        # Throttle to ~20 updates so the write is visibly progressing.
        step = max(1, total // 20)
        if current % step == 0 or current == total - 1:
            print(f"  frame {current + 1}/{total}", end="\r", flush=True)

    anim.save(str(path), writer=writer, dpi=cfg.figure_dpi, progress_callback=report)
    print()  # end the in-place progress line
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
            "husimi_peak": trajectory.husimi_vmax,
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
