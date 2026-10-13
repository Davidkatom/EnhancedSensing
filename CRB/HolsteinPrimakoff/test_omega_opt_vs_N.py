"""Checks for the Omega_opt-versus-N sweep.

Run: python -m unittest CRB.HolsteinPrimakoff.test_omega_opt_vs_N -v
"""

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from CRB import smart_save
from CRB.HolsteinPrimakoff.omega_opt_vs_N import (
    SweepConfig,
    exact_bath_qfi,
    hp_bath_qfi,
    maximize,
    omega_lz,
    plot_sweep,
    run_sweep,
)
from CRB.plot_optimized_qfi_vs_N import OptimizedQFIConfig, optimize_Omega


class OmegaOptTests(unittest.TestCase):
    def test_lz_rule_gives_an_even_split(self) -> None:
        for N, J, omega in ((15, 1.0, 1.0), (30, 0.5, 2.0)):
            Omega = omega_lz(N, J, omega)
            self.assertAlmostEqual(np.exp(-np.pi * Omega**2 / (2 * J * N * omega)), 0.5)
        self.assertAlmostEqual(omega_lz(15, 1.0, 1.0), 2.5728, places=4)

    def test_exact_objective_matches_existing_optimizer(self) -> None:
        cfg = SweepConfig(show=False)
        N = 6
        grid = np.linspace(cfg.c_min, cfg.c_max, cfg.c_points) * np.sqrt(N)
        ours, _ = maximize(lambda Omega: exact_bath_qfi(cfg, N, Omega), grid, cfg.Omega_tol)
        existing_cfg = OptimizedQFIConfig(t_max=cfg.t, J_nominal=cfg.J, omega=cfg.omega)
        theirs, _ = optimize_Omega(N, existing_cfg, np.linspace(0.1, 5.0, 101))
        self.assertAlmostEqual(ours, theirs, delta=5e-3)

    def test_hp_tracks_exact_after_the_first_crossing(self) -> None:
        cfg = SweepConfig(show=False)
        N = 8
        grid = np.linspace(cfg.c_min, cfg.c_max, cfg.c_points) * np.sqrt(N)
        Omega_exact, F_exact = maximize(lambda Omega: exact_bath_qfi(cfg, N, Omega), grid, cfg.Omega_tol)
        Omega_hp, F_hp = maximize(lambda Omega: hp_bath_qfi(cfg, N, Omega), grid, cfg.Omega_tol)
        self.assertLess(abs(Omega_hp / Omega_exact - 1), 0.03)
        self.assertLess(abs(F_hp / F_exact - 1), 0.05)

    def test_export_pairs_figure_with_complete_numerical_record(self) -> None:
        cfg = SweepConfig(N_values=(4, 6), c_points=7, dpi=50, show=False)
        result = run_sweep(cfg)
        self.assertTrue(np.all(result.F_exact_at_lz <= result.F_exact_max + 1e-9))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            def save_in_temporary_directory(figure, **kwargs):
                return smart_save.save_plot(figure, output_dir=root, **kwargs)

            with patch.object(smart_save, "GOOGLE_DRIVE_GRAPHS_DIRECTORY", root), patch(
                "CRB.HolsteinPrimakoff.omega_opt_vs_N.save_plot",
                side_effect=save_in_temporary_directory,
            ) as exporter:
                record = plot_sweep(cfg, result)

            self.assertEqual(exporter.call_count, 1)
            self.assertEqual(list(root.rglob("*.plot.json")), [record.json_path])
            self.assertEqual(list(root.rglob("*.png")), [record.image_path])
            payload = json.loads(record.json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["parameters"], json.loads(json.dumps(asdict(cfg))))
            series = {curve["name"]: curve for curve in payload["series"]}
            np.testing.assert_array_equal(series["exact: Omega_opt"]["y"], result.Omega_exact)
            np.testing.assert_array_equal(series["HP: Omega_opt"]["y"], result.Omega_hp)
            np.testing.assert_array_equal(series["LZ 50:50 Omega"]["y"], result.Omega_lz)
            for curve in series.values():
                np.testing.assert_array_equal(curve["x"], result.N)


if __name__ == "__main__":
    unittest.main()
