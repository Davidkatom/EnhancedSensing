"""Plot central <X>, <Y>, <Z> for the HP Hamiltonian with its number term omitted.

Default: |+> times oscillator vacuum, N=15, Omega=2.2, J=omega=1, t=0..2.
The vacuum distribution of P=i(a^dagger-a) is integrated by Gaussian
quadrature. A larger quadrature is required to agree before the plot is saved.
All plots and complete numerical records are saved through CRB.smart_save,
then copied to graphs/plot_linearized_hp_spin/ in the repository.

Run: python CRB/HolsteinPrimakoff/plot_linearized_hp_spin.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
import re
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CRB.HolsteinPrimakoff.linearized_momentum import MomentumConfig, simulate_momentum
from CRB.smart_save import save_plot


@dataclass(frozen=True, slots=True)
class PlotConfig(MomentumConfig):
    """Physics/numerics inherited from MomentumConfig; dimensions are inches."""

    width: float = 10.0
    height: float = 5.2
    dpi: int = 160
    check_nodes: int = 96
    check_tol: float = 1e-7

    def __post_init__(self) -> None:
        super(PlotConfig, self).__post_init__()
        for name in ("width", "height", "check_tol"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not isinstance(self.dpi, int) or self.dpi < 1:
            raise ValueError("dpi must be a positive integer.")
        if not isinstance(self.check_nodes, int) or self.check_nodes <= self.p_nodes:
            raise ValueError("check_nodes must be an integer above p_nodes.")


def filename(cfg: PlotConfig) -> str:
    def tag(value: object) -> str:
        if isinstance(value, float):
            if value == np.pi / 2:
                return "pi_over_2"
            text = str(int(value)) if value.is_integer() else repr(value)
            return re.sub(r"e([+-])0+(\d+)", r"e\1\2", text)
        return str(value)
    return "spin__" + "__".join(f"{f.name}={tag(getattr(cfg, f.name))}" for f in fields(cfg))


def main(argv: list[str] | None = None) -> None:
    defaults = PlotConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"), dest=field.name,
                            type=type(value), default=value)
    cfg = PlotConfig(**vars(parser.parse_args(argv)))
    result = simulate_momentum(cfg)
    check = simulate_momentum(replace(cfg, p_nodes=cfg.check_nodes, check_nodes=cfg.check_nodes + 32))
    difference = float(np.max(np.abs(result.bloch - check.bloch)))
    if difference > cfg.check_tol:
        raise RuntimeError(f"Quadrature error {difference:.3e} exceeds {cfg.check_tol:g}; increase --p-nodes and --check-nodes.")

    figure, axis = plt.subplots(figsize=(cfg.width, cfg.height), layout="constrained")
    colors = ("#2463ac", "#e07820", "#32946b")
    for index, name in enumerate("XYZ"):
        axis.plot(result.time, result.bloch[:, index], color=colors[index], lw=1.7,
                  label=rf"$\langle {name}\rangle$")
    axis.axhline(0, color="0.7", lw=0.7, zorder=0)
    if cfg.omega > 0:
        crossing = np.pi / (4.0 * cfg.omega)
        if crossing <= cfg.tmax:
            axis.axvline(crossing, color="0.4", ls="--", lw=1.1,
                         label=rf"$p=0$ crossing: $t_c={crossing:.3f}$")
    axis.set(xlim=(0, cfg.tmax), ylim=(-1.06, 1.06), xlabel=r"Time $t$",
             ylabel="Central-spin expectation value")
    axis.set_yticks(np.linspace(-1, 1, 5))
    axis.grid(True, alpha=0.18)
    axis.legend(loc="upper right", ncol=2, fontsize=10)
    axis.set_title(
        "Linearized HP dynamics — oscillator vacuum, number term omitted\n"
        rf"$H_{{\rm lin}}=\Omega X+JZ[N\cos(2\omega t)+\sqrt{{N}}P\sin(2\omega t)]$"
        "\n" + rf"$N={cfg.N},\quad \Omega={cfg.Omega:g},\quad J={cfg.J:g},\quad \omega={cfg.omega:g}$"
        + rf"; central state: $\theta={cfg.theta / np.pi:g}\pi,\ \phi={cfg.phi / np.pi:g}\pi$",
        fontsize=11,
    )
    try:
        record = save_plot(
            figure, system=Path(__file__).stem, plot_type="spin_vs_time",
            params=asdict(cfg), data={f"<{name}>": (result.time, result.bloch[:, i]) for i, name in enumerate("XYZ")},
            name=filename(cfg), xlabel="Time t", ylabel="Central-spin expectation value",
            xunit="inverse reference frequency", yunit="dimensionless", script_path=__file__,
            notes="HP model with -2 a^dagger a cos(2 omega t) omitted. P=i(a^dagger-a), vacuum variance 1. Pauli convention, hbar=1. The full vacuum momentum distribution is traced for central-spin observables.",
            metadata={"norm_error": result.norm_error, "quadrature_max_difference": difference},
            dpi=cfg.dpi, bbox_inches="tight",
        )
    finally:
        plt.close(figure)
    local_dir = REPO_ROOT / "graphs" / Path(__file__).stem
    local_dir.mkdir(parents=True, exist_ok=True)
    for source in (record.image_path, record.json_path):
        if source is not None:
            shutil.copy2(source, local_dir / source.name)
    print(f"Norm error: {result.norm_error:.3e}")
    print(f"Quadrature convergence ({cfg.p_nodes} vs {cfg.check_nodes} nodes): {difference:.3e}")
    for value in (0.0, np.pi / (4 * cfg.omega) if cfg.omega > 0 else 0.0, 1.0, cfg.tmax):
        if 0 <= value <= cfg.tmax:
            index = int(np.argmin(np.abs(result.time - value)))
            print(f"t={result.time[index]:.4f}: X,Y,Z={result.bloch[index].tolist()}")
    print(f"SMART_SAVE_IMAGE={record.image_path}")
    print(f"LOCAL_IMAGE={local_dir / record.image_path.name}")
    print(f"LOCAL_DATA={local_dir / record.json_path.name}")


if __name__ == "__main__":
    main()
