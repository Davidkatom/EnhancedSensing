"""Analytic-limit and sensitivity checks for the driven single qubit.

Run: python -m unittest CRB.LandauZener.test_driven_qubit -v
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from CRB import smart_save
from CRB.LandauZener.driven_qubit import SimulationConfig, plot_trajectories, simulate


class DrivenQubitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = SimulationConfig(points=151, t_pi=2.0, show=False)

    def test_constant_x_rotation_and_coupling_qfi(self) -> None:
        cfg = replace(self.cfg, N=0.0, Omega=0.7, qfi="Omega")
        zero = simulate(cfg, 0.0)
        plus = simulate(cfg, np.pi / 2)
        phase = 2 * cfg.Omega * zero.time
        expected = np.column_stack((np.zeros_like(phase), -np.sin(phase), np.cos(phase)))
        np.testing.assert_allclose(zero.bloch, expected, atol=2e-8)
        np.testing.assert_allclose(zero.qfi, 4 * zero.time**2, atol=2e-7)
        np.testing.assert_allclose(plus.bloch, np.tile([1, 0, 0], (cfg.points, 1)), atol=2e-8)
        np.testing.assert_allclose(plus.qfi, 0.0, atol=1e-10)

    def test_commuting_z_drive_and_amplitude_qfi(self) -> None:
        cfg = replace(self.cfg, Omega=0.0, omega=1.7, J=0.6, qfi="N")
        zero = simulate(cfg, 0.0)
        plus = simulate(cfg, np.pi / 2)
        integrated_modulation = np.sin(cfg.omega * plus.time) / cfg.omega
        phase = 2 * cfg.N * cfg.J * integrated_modulation
        expected = np.column_stack((np.cos(phase), np.sin(phase), np.zeros_like(phase)))
        np.testing.assert_allclose(plus.bloch, expected, atol=5e-8)
        np.testing.assert_allclose(plus.qfi, 4 * cfg.J**2 * integrated_modulation**2, atol=5e-8)
        np.testing.assert_allclose(zero.bloch, np.tile([0, 0, 1], (cfg.points, 1)), atol=1e-12)
        np.testing.assert_allclose(zero.qfi, 0.0, atol=1e-10)

    def test_time_qfi_including_initial_time(self) -> None:
        cfg = replace(self.cfg, Omega=0.0, omega=1.7, J=0.6, qfi="t")
        plus = simulate(cfg, np.pi / 2)
        np.testing.assert_allclose(plus.qfi, 4 * (cfg.N * cfg.J)**2 * np.cos(cfg.omega * plus.time)**2, atol=2e-8)
        self.assertAlmostEqual(plus.qfi[0], 4 * (cfg.N * cfg.J)**2)

    def test_longitudinal_coupling_qfi_has_n_squared_scaling(self) -> None:
        cfg = replace(self.cfg, Omega=0.0, omega=0.7, J=0.6)
        self.assertEqual(cfg.qfi, "J")
        zero = simulate(cfg, 0.0)
        plus = simulate(cfg, np.pi / 2)
        np.testing.assert_allclose(plus.qfi, 4 * cfg.N**2 * (np.sin(cfg.omega * plus.time) / cfg.omega)**2, atol=2e-6)
        np.testing.assert_allclose(zero.qfi, 0.0, atol=1e-10)

    def test_noncommuting_parameter_sensitivities(self) -> None:
        delta = 1e-5
        for parameter in ("J", "Omega", "N"):
            cfg = replace(self.cfg, Omega=0.7, omega=1.7, qfi=parameter,
                          t_pi=1.0, rtol=1e-11, atol=1e-13)
            for theta in (0.0, np.pi / 2):
                with self.subTest(parameter=parameter, theta=theta):
                    result = simulate(cfg, theta)
                    plus = simulate(replace(cfg, **{parameter: getattr(cfg, parameter) + delta}), theta)
                    minus = simulate(replace(cfg, **{parameter: getattr(cfg, parameter) - delta}), theta)
                    finite_difference = (plus.states - minus.states) / (2 * delta)
                    np.testing.assert_allclose(result.tangents, finite_difference, atol=2e-8, rtol=2e-7)
                    overlap = np.einsum("ti,ti->t", result.states.conj(), finite_difference)
                    pure_qfi = 4 * (np.sum(np.abs(finite_difference)**2, axis=1) - np.abs(overlap)**2)
                    np.testing.assert_allclose(result.qfi, pure_qfi, atol=2e-7, rtol=2e-7)
                    self.assertAlmostEqual(result.qfi[0], 0.0)

    def test_default_norm_and_tolerance_convergence(self) -> None:
        cfg = replace(self.cfg, t_pi=4.0)
        for theta in (0.0, np.pi / 2):
            result = simulate(cfg, theta)
            reference = simulate(replace(cfg, rtol=1e-11, atol=1e-13), theta)
            self.assertLess(result.norm_error, 1e-7)
            np.testing.assert_allclose(result.bloch, reference.bloch, atol=2e-7)
            np.testing.assert_allclose(result.qfi, reference.qfi, atol=2e-7, rtol=2e-7)

    def test_pauli_cfi_for_commuting_z_drive(self) -> None:
        cfg = replace(self.cfg, Omega=0.0, J=0.6)
        plus = simulate(cfg, np.pi / 2)
        expected = 4 * cfg.N**2 * np.sin(plus.time)**2
        for component in (0, 1):
            # At deterministic outcomes the helper uses the support-only
            # convention; compare the analytic regular expression elsewhere.
            regular = 1 - plus.bloch[:, component]**2 > 1e-6
            np.testing.assert_allclose(plus.cfi[regular, component], expected[regular],
                                       atol=2e-6, rtol=2e-7)
        np.testing.assert_allclose(plus.cfi[:, 2], 0.0, atol=1e-10)
        np.testing.assert_allclose(plus.cfi[0], 0.0, atol=1e-10)

    def test_cfi_from_probability_finite_differences_and_qfi_bound(self) -> None:
        cfg = replace(self.cfg, Omega=0.7, omega=1.7, rtol=1e-11, atol=1e-13)
        delta = 1e-5
        for theta in (0.0, np.pi / 2):
            result = simulate(cfg, theta)
            plus = simulate(replace(cfg, J=cfg.J + delta), theta)
            minus = simulate(replace(cfg, J=cfg.J - delta), theta)
            mean_derivative = (plus.bloch - minus.bloch) / (2 * delta)
            variance = 1 - result.bloch**2
            regular = variance > 1e-5
            expected = mean_derivative[regular]**2 / variance[regular]
            np.testing.assert_allclose(result.cfi[regular], expected, atol=2e-5, rtol=2e-6)
            self.assertTrue(np.all(np.isfinite(result.cfi)))
            self.assertTrue(np.all(result.cfi >= 0.0))
            self.assertTrue(np.all(result.cfi <= result.qfi[:, None] + 1e-7))
            np.testing.assert_allclose(result.cfi[0], 0.0, atol=1e-10)

    def test_export_pairs_full_figure_with_complete_numerical_record(self) -> None:
        cfg = replace(self.cfg, omega=0.7, points=11, t_pi=0.1, dpi=50)
        trajectories = (simulate(cfg, 0.0), simulate(cfg, np.pi / 2))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            def save_in_temporary_directory(figure, **kwargs):
                return smart_save.save_plot(figure, output_dir=root, **kwargs)

            with patch.object(smart_save, "GOOGLE_DRIVE_GRAPHS_DIRECTORY", root), patch(
                "CRB.LandauZener.driven_qubit.save_plot",
                side_effect=save_in_temporary_directory,
            ) as exporter:
                record = plot_trajectories(cfg, trajectories)

            self.assertEqual(exporter.call_count, 1)
            self.assertEqual(len(exporter.call_args.args[0].axes), 4)
            self.assertEqual(list(root.rglob("*.plot.json")), [record.json_path])
            self.assertEqual(list(root.rglob("*.png")), [record.image_path])
            self.assertEqual(record.image_path.with_suffix(".plot.json"), record.json_path)
            payload = json.loads(record.json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["parameters"], json.loads(json.dumps(asdict(cfg))))
            series = {curve["name"]: curve for curve in payload["series"]}
            self.assertEqual(len(series), 14)
            for label, result in zip(("|0>", "|+>"), trajectories):
                np.testing.assert_array_equal(series[f"{label}: QFI(J)"]["y"], result.qfi)
                for component, name in enumerate(("X", "Y", "Z")):
                    np.testing.assert_array_equal(series[f"{label}: <{name}>"]["y"], result.bloch[:, component])
                    np.testing.assert_array_equal(series[f"{label}: CFI(J) <{name}>"]["y"], result.cfi[:, component])
            for curve in series.values():
                np.testing.assert_array_equal(curve["x"], trajectories[0].time)
            with Image.open(record.image_path) as image:
                self.assertEqual(json.loads(image.info["Description"])["record"], record.json_path.name)

    def test_zero_modulation_frequency_is_a_static_z_drive(self) -> None:
        cfg = replace(self.cfg, Omega=0.0, omega=0.0, J=0.6)
        plus = simulate(cfg, np.pi / 2)
        phase = 2 * cfg.N * cfg.J * plus.time
        expected = np.column_stack((np.cos(phase), np.sin(phase), np.zeros_like(phase)))
        np.testing.assert_allclose(plus.bloch, expected, atol=5e-8)
        np.testing.assert_allclose(plus.qfi, 4 * cfg.N**2 * plus.time**2,
                                   atol=2e-5, rtol=2e-7)


if __name__ == "__main__":
    unittest.main()
