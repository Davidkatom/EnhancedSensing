"""Bath-only QCRB vs N for several initial-probe angles beta (arXiv:0710.0285).

The bath is initialized in the spin-``N/2`` coherent state at polar angle
``beta`` from the ``+z`` axis -- the product-state probe of Boixo et al.
(arXiv:0710.0285), where ``beta`` is the rotation about ``y`` that tilts the
north-pole state ``|m = N/2>``.  The full central-spin model

    H = Omega_0 sigma_x + J sigma_z S_z

is solved exactly with the block solver from ``crb_core`` (noiseless).  For a
fixed time slice ``t = t_slice`` this plots the bath-only QCRB for ``J``,
``delta J = 1/sqrt(F_Q)``, versus ``N``, one curve per ``beta``, with one panel
per drive strength in ``omega_panels``.

Two regimes are contrasted:

* ``Omega_0 = 0``: the generator of ``J`` is ``sigma_z S_z``, *linear* in
  ``S_z``.  Coherent (product) probes give the standard quantum limit
  ``delta J ~ N^-1/2`` at every angle.
* ``Omega_0 >> J N`` (dispersive): adiabatic elimination gives the *quadratic*
  ``H_eff ~ (J^2 / 2 Omega_0) S_z^2``.  This is the paper's ``k = 2`` case, where
  the optimal product angle is ``sin beta = 1/sqrt(2)`` (``beta = 45 deg``) and
  the coherent probe reaches ``delta J ~ N^-3/2``.  The entangled reference
  ``(|m=N/2> + |m=0>)/sqrt(2)`` reaches ``N^-1`` (linear regime) and ``N^-2``
  (dispersive regime).
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

try:
    from CRB.crb_core import (
        SimulationConfig as BaseSimulationConfig,
        coherent_bath_state,
        compute_bath_qfi_trajectory,
        compute_qcrb_matrices,
        fit_power_law,
        get_bath_density_matrices,
        optimal_sz2_bath_state,
    )
except ModuleNotFoundError:  # Allow: python CRB/qcrb_initial_angle_sweep.py
    from crb_core import (
        SimulationConfig as BaseSimulationConfig,
        coherent_bath_state,
        compute_bath_qfi_trajectory,
        compute_qcrb_matrices,
        fit_power_law,
        get_bath_density_matrices,
        optimal_sz2_bath_state,
    )


@dataclass(frozen=True)
class SimulationConfig(BaseSimulationConfig):
    """Sweep and output configuration for the probe-angle QCRB-vs-N analysis."""

    # No noise; only the initial-probe angle beta (and the drive panel) vary.
    gamma: float = 0.3
    beta: float = 0.0  # central-spin dephasing rate, OFF (not the probe angle)

    # One panel per drive.  0 is the linear-generator (SQL) regime; a value
    # >> J * max(N_values) is the dispersive S_z^2 regime of the paper.
    omega_panels: tuple[float, ...] = (0.0, 800.0)

    beta_deg_values: tuple[float, ...] = (90.0, 60.0, 45.0, 30.0, 15.0)
    show_optimal_reference: bool = True

    N_values: tuple[int, ...] = (4, 6, 8, 10, 14, 18, 24, 32, 48, 64)
    t_slice: float = 10.0

    output_figure: str = "qcrb_initial_angle_sweep.png"


def qcrb_vs_N(cfg: SimulationConfig, omega: float, make_state) -> np.ndarray:
    """Return the QCRB ``1/sqrt(F_Q)`` over ``cfg.N_values`` at ``t = t_slice``.

    ``make_state(N)`` supplies the length-``N + 1`` initial bath state vector.
    """
    tslice = np.array([cfg.t_slice], dtype=float)
    qfi_matrix = np.zeros((len(cfg.N_values), 1))

    for index, N in enumerate(cfg.N_values):
        probe = make_state(N)
        bath_rhos_plus = get_bath_density_matrices(
            Omega_0=omega,
            J=cfg.J_nominal + cfg.dJ,
            tlist=tslice,
            N=N,
            gamma=cfg.gamma,
            beta=cfg.beta,
            bath_state=probe,
        )
        bath_rhos_minus = get_bath_density_matrices(
            Omega_0=omega,
            J=cfg.J_nominal - cfg.dJ,
            tlist=tslice,
            N=N,
            gamma=cfg.gamma,
            beta=cfg.beta,
            bath_state=probe,
        )
        qfi_t, _, _, _ = compute_bath_qfi_trajectory(
            bath_rhos_plus=bath_rhos_plus,
            bath_rhos_minus=bath_rhos_minus,
            dJ=cfg.dJ,
            tol=cfg.qfi_tol,
        )
        qfi_matrix[index, 0] = qfi_t[0]

    _, qcrb_unnormalized, _ = compute_qcrb_matrices(
        qfi_matrix=qfi_matrix,
        tlist=tslice,
        t_overhead=cfg.t_overhead,
        qcrb_eps=cfg.qcrb_eps,
    )
    return qcrb_unnormalized[:, 0]


def compute_panel(
    cfg: SimulationConfig, omega: float
) -> tuple[dict[float, np.ndarray], np.ndarray | None]:
    """QCRB-vs-N for every probe angle (and the optimal reference) at one drive."""
    qcrb_by_beta: dict[float, np.ndarray] = {}
    for beta_deg in cfg.beta_deg_values:
        beta_rad = np.deg2rad(beta_deg)
        qcrb_by_beta[beta_deg] = qcrb_vs_N(
            cfg, omega, lambda N, b=beta_rad: coherent_bath_state(N, b)
        )
    reference = None
    if cfg.show_optimal_reference:
        reference = qcrb_vs_N(cfg, omega, optimal_sz2_bath_state)
    return qcrb_by_beta, reference


def _add_slope_guides(
    axis: plt.Axes, N_array: np.ndarray, slopes: tuple[float, ...], y_anchor: float
) -> None:
    """Fan reference power laws ``N^-p`` down from ``(N[0], y_anchor)``."""
    for slope in slopes:
        guide = y_anchor * (N_array[0] / N_array) ** slope
        axis.loglog(N_array, guide, color="0.6", linewidth=0.9, linestyle=(0, (3, 3)))
        axis.text(
            N_array[-1],
            guide[-1],
            rf"$N^{{-{slope:g}}}$",
            color="0.35",
            fontsize=8,
            ha="left",
            va="center",
        )


def plot_results(
    cfg: SimulationConfig,
    panels: dict[float, tuple[dict[float, np.ndarray], np.ndarray | None]],
) -> None:
    N_array = np.asarray(cfg.N_values, dtype=float)
    omega_list = list(cfg.omega_panels)
    fig, axes = plt.subplots(
        1, len(omega_list), figsize=(7.4 * len(omega_list), 6.3), squeeze=False
    )

    for column, omega in enumerate(omega_list):
        axis = axes[0, column]
        qcrb_by_beta, reference = panels[omega]

        for index, beta_deg in enumerate(cfg.beta_deg_values):
            qcrb = qcrb_by_beta[beta_deg]
            exponent = fit_power_law(N_array, qcrb)
            note = " (paper $k{=}2$ opt.)" if np.isclose(beta_deg, 45.0) else ""
            axis.loglog(
                N_array,
                qcrb,
                marker="o",
                linestyle="-",
                color=f"C{index}",
                label=rf"$\beta={beta_deg:g}^\circ$: $N^{{{exponent:.2f}}}$" + note,
            )

        if reference is not None:
            exponent = fit_power_law(N_array, reference)
            axis.loglog(
                N_array,
                reference,
                marker="D",
                linestyle="-",
                color="black",
                linewidth=1.8,
                label=rf"optimal $(|N/2\rangle+|0\rangle)/\sqrt{{2}}$: $N^{{{exponent:.2f}}}$",
            )

        dispersive = omega >= cfg.J_nominal * max(cfg.N_values)
        slopes = (1.0, 1.5, 2.0) if dispersive else (0.5, 1.0)
        stack = list(qcrb_by_beta.values()) + (
            [reference] if reference is not None else []
        )
        y_anchor = 1.7 * max(values[0] for values in stack)
        _add_slope_guides(axis, N_array, slopes, y_anchor)

        regime = (
            rf"$\Omega/J={omega:g}\gg JN$: dispersive $S_z^2$ generator"
            if dispersive
            else rf"$\Omega/J={omega:g}$: linear $\sigma_z S_z$ generator"
        )
        axis.set_title(regime)
        axis.set_xlabel(r"$N$")
        axis.set_ylabel(r"QCRB  $\delta J = 1/\sqrt{F_Q}$")
        axis.grid(True, which="both", linestyle=":", alpha=0.55)
        axis.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        r"Bath-only QCRB vs $N$ for the spin-coherent probe at polar angle "
        r"$\beta$ (arXiv:0710.0285)"
        "\n"
        rf"$H=\Omega_0\sigma_x + J\sigma_z S_z$, $t={cfg.t_slice:g}$, "
        rf"no noise, $J={cfg.J_nominal:g}$",
        y=1.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(cfg.output_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {cfg.output_figure}")


def main() -> None:
    cfg = SimulationConfig()
    N_array = np.asarray(cfg.N_values, dtype=float)

    panels: dict[float, tuple[dict[float, np.ndarray], np.ndarray | None]] = {}
    for omega in cfg.omega_panels:
        print(
            f"Computing Omega/J={omega:g} for N in {cfg.N_values} "
            f"at t={cfg.t_slice:g} ..."
        )
        qcrb_by_beta, reference = compute_panel(cfg, omega)
        panels[omega] = (qcrb_by_beta, reference)

        print(f"  QCRB (delta J = 1/sqrt(F_Q)) at t={cfg.t_slice:g}:")
        for beta_deg in cfg.beta_deg_values:
            exponent = fit_power_law(N_array, qcrb_by_beta[beta_deg])
            print(
                f"    beta={beta_deg:5.1f} deg: fit N^{exponent:+.3f}  "
                f"QCRB(N={cfg.N_values[0]})={qcrb_by_beta[beta_deg][0]:.4e}  "
                f"QCRB(N={cfg.N_values[-1]})={qcrb_by_beta[beta_deg][-1]:.4e}"
            )
        if reference is not None:
            print(
                f"    optimal ref  : fit N^{fit_power_law(N_array, reference):+.3f}  "
                f"QCRB(N={cfg.N_values[0]})={reference[0]:.4e}  "
                f"QCRB(N={cfg.N_values[-1]})={reference[-1]:.4e}"
            )

    plot_results(cfg, panels)


if __name__ == "__main__":
    main()
