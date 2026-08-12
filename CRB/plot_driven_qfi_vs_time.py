"""Plot bath quantum Fisher information versus time for the driven protocol.

The model and initial state are

    H = Omega * sigma_x + J * sigma_z * S_z + omega * S_x,
    Omega = omega = 3,
    |psi(0)> = |+>_central |0>_bath^N,
    N = 20.

The QFI estimates the coupling ``J`` from the reduced bath state.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from CRB.crb_core import (
        coherent_bath_state,
        compute_bath_qfi_trajectory,
        get_bath_density_matrices,
        save_plot,
    )
except ModuleNotFoundError:  # Allow: python CRB/plot_driven_qfi_vs_time.py
    from crb_core import (
        coherent_bath_state,
        compute_bath_qfi_trajectory,
        get_bath_density_matrices,
        save_plot,
    )


@dataclass(frozen=True)
class SimulationConfig:
    N: int = 15
    Omega: float = 3.0
    omega: float = 1

    J_nominal: float = 1.0
    dJ: float = 1e-3

    t_min: float = 0.0
    t_max: float = 40.0
    n_steps: int = 401

    gamma: float = 0.0
    beta: float = 0.0
    qfi_tol: float = 1e-12

    output_figure: str = "driven_qfi_vs_time_N20.png"


def compute_qfi(cfg: SimulationConfig, tlist: np.ndarray) -> np.ndarray:
    """Return the bath QFI trajectory for estimating J."""
    # theta=0 is the symmetric-subspace representation of |0>^N.
    bath_state = coherent_bath_state(cfg.N, theta=0.0)

    bath_rhos_plus = get_bath_density_matrices(
        Omega_0=cfg.Omega,
        omega=cfg.omega,
        J=cfg.J_nominal + cfg.dJ,
        tlist=tlist,
        N=cfg.N,
        gamma=cfg.gamma,
        beta=cfg.beta,
        bath_state=bath_state,
    )
    bath_rhos_minus = get_bath_density_matrices(
        Omega_0=cfg.Omega,
        omega=cfg.omega,
        J=cfg.J_nominal - cfg.dJ,
        tlist=tlist,
        N=cfg.N,
        gamma=cfg.gamma,
        beta=cfg.beta,
        bath_state=bath_state,
    )
    qfi, _, _, _ = compute_bath_qfi_trajectory(
        bath_rhos_plus=bath_rhos_plus,
        bath_rhos_minus=bath_rhos_minus,
        dJ=cfg.dJ,
        tol=cfg.qfi_tol,
    )
    return qfi


def plot_qfi(
    tlist: np.ndarray,
    qfi: np.ndarray,
    cfg: SimulationConfig,
) -> Path:
    """Plot and save the QFI trajectory."""
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(tlist, qfi, color="tab:blue", linewidth=2.0)
    axis.set_xlabel(r"Interrogation time $t$")
    axis.set_ylabel(r"Bath quantum Fisher information $F_Q(t)$")
    axis.set_title(
        rf"$N={cfg.N}$, $\Omega=\omega={cfg.omega:g}$, "
        r"$|+\rangle|0\rangle^{\otimes N}$"
    )
    axis.grid(True, linestyle=":", alpha=0.8)
    figure.tight_layout()

    maximum_index = int(np.argmax(qfi))
    output_path = save_plot(
        figure,
        cfg.output_figure,
        metadata={
            "config": cfg,
            "time_values": tlist,
            "bath_qfi": qfi,
            "maximum_qfi": qfi[maximum_index],
            "maximum_qfi_time": tlist[maximum_index],
        },
        script_path=__file__,
        dpi=200,
        bbox_inches="tight",
    )
    # plt.close(figure)
    plt.show()
    return output_path


def main() -> None:
    cfg = SimulationConfig()
    tlist = np.linspace(cfg.t_min, cfg.t_max, cfg.n_steps)
    qfi = compute_qfi(cfg, tlist)

    maximum_index = int(np.argmax(qfi))
    print(
        f"Maximum bath QFI: F_Q={qfi[maximum_index]:.6e} "
        f"at t={tlist[maximum_index]:.3f}"
    )

    output_path = plot_qfi(tlist, qfi, cfg)
    print(f"Saved QFI plot to {output_path}")


if __name__ == "__main__":
    main()
