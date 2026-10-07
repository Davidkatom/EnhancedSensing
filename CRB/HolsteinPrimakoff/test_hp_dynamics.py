"""Exact-limit, sensitivity, and export checks for the Holstein-Primakoff dynamics.

Run: python -m unittest CRB.HolsteinPrimakoff.test_hp_dynamics -v
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.linalg import expm

from CRB import smart_save
from CRB.crb_core import (
    build_hamiltonian,
    central_spin_state,
    coherent_bath_state,
    evolve_bath_density_matrix_noiseless,
    qfi_from_rho_and_drho,
)
from CRB.HolsteinPrimakoff.hp_dynamics import (
    MODELS,
    PAULI,
    SimulationConfig,
    bath_operators,
    crossing_times,
    plot_trajectories,
    simulate,
)


class HolsteinPrimakoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = SimulationConfig(N=6, Omega=0.9, omega=0.7, J=0.8, points=41, t_pi=0.8, show=False)

    def test_exact_rotating_frame_matches_lab_frame_solver(self) -> None:
        cfg = self.cfg
        result = simulate(cfg, "exact")
        hamiltonian = build_hamiltonian(cfg.Omega, cfg.J, cfg.N, omega=cfg.omega).full()
        bath = coherent_bath_state(cfg.N, 0.0)
        initial = np.kron(central_spin_state(np.pi / 2).full().ravel(), bath)
        lab_sz = cfg.N - 2.0 * np.arange(cfg.N + 1)
        delta = 1e-5
        for index in (10, 25, 40):
            with self.subTest(index=index):
                t = result.time[index]
                amplitudes = (expm(-1j * hamiltonian * t) @ initial).reshape(2, cfg.N + 1)
                rho_c = amplitudes @ amplitudes.conj().T
                np.testing.assert_allclose(result.bloch[index], np.einsum("ij,aji->a", rho_c, PAULI).real,
                                           atol=1e-7)
                self.assertAlmostEqual(result.purity[index], np.trace(rho_c @ rho_c).real, places=7)
                # h_z = J B(t) is the lab-frame J S_z seen from the rotating frame.
                weights = np.sum(np.abs(amplitudes)**2, axis=0)
                mean = weights @ lab_sz
                self.assertAlmostEqual(result.field_mean[index], cfg.J * mean, places=6)
                self.assertAlmostEqual(result.field_std[index], cfg.J * np.sqrt(weights @ lab_sz**2 - mean**2),
                                       places=6)
                plus, minus = (evolve_bath_density_matrix_noiseless(cfg.Omega, J, t, cfg.N, bath, omega=cfg.omega)
                               for J in (cfg.J + delta, cfg.J - delta))
                expected, _ = qfi_from_rho_and_drho(0.5 * (plus + minus), (plus - minus) / (2 * delta), tol=1e-10)
                self.assertAlmostEqual(result.qfi_bath[index], expected, delta=1e-6 * max(1.0, expected))

    def test_static_central_spin_branch_overlap(self) -> None:
        # With Omega = 0, Z is conserved and each branch is a product of spins under
        # omega X +- J Z, so the central coherence is the overlap g^N of the two branches.
        cfg = replace(self.cfg, Omega=0.0)
        result = simulate(cfg, "exact")
        x, z = PAULI[0], PAULI[2]
        zero = np.array([1.0, 0.0], dtype=complex)
        g = np.array([zero.conj() @ expm(1j * (cfg.omega * x - cfg.J * z) * t)
                      @ expm(-1j * (cfg.omega * x + cfg.J * z) * t) @ zero for t in result.time])
        overlap = g**cfg.N
        np.testing.assert_allclose(result.bloch[:, 0], overlap.real, atol=1e-8)
        np.testing.assert_allclose(result.bloch[:, 1], -overlap.imag, atol=1e-8)
        np.testing.assert_allclose(result.bloch[:, 2], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.purity, 0.5 * (1 + np.abs(overlap)**2), atol=1e-8)

    def test_hp_coupling_is_the_linearized_exact_coupling(self) -> None:
        cfg = self.cfg
        exact, hp = bath_operators(cfg, "exact"), bath_operators(cfg, "HP")
        self.assertEqual(hp.B_c.size, 4 * cfg.N + 41)
        np.testing.assert_allclose(hp.B_c[:cfg.N + 1], exact.B_c)
        for operator in (exact.B_s, hp.B_s):
            np.testing.assert_allclose(operator, operator.conj().T, atol=1e-14)
            self.assertEqual(np.count_nonzero(np.abs(np.triu(operator, 2)) > 1e-14), 0)
        # <n+1|S_y|n> = i sqrt((N - n)(n + 1)) exactly and i sqrt(N (n + 1)) after linearizing.
        n = np.arange(cfg.N)
        np.testing.assert_allclose(np.diag(exact.B_s, -1), 1j * np.sqrt((cfg.N - n) * (n + 1)), atol=1e-12)
        np.testing.assert_allclose(np.diag(hp.B_s, -1)[:cfg.N] * np.sqrt(1 - n / cfg.N), np.diag(exact.B_s, -1),
                                   atol=1e-12)

    def test_hp_matches_exact_before_the_bath_is_excited(self) -> None:
        cfg = SimulationConfig(N=15, Omega=2.0, omega=1.0, J=1.0, points=31, t_pi=0.1, show=False)
        hp, exact = simulate(cfg, "HP"), simulate(cfg, "exact")
        self.assertLess(exact.excitation.max(), 0.01)
        np.testing.assert_allclose(hp.bloch, exact.bloch, atol=5e-3)
        np.testing.assert_allclose(hp.qfi_bath, exact.qfi_bath, rtol=5e-3, atol=1e-8)
        np.testing.assert_allclose(hp.excitation, exact.excitation, rtol=5e-2, atol=1e-6)

    def test_tangents_match_finite_differences(self) -> None:
        cfg = replace(self.cfg, points=21, rtol=1e-11, atol=1e-13)
        delta = 1e-5
        for model in MODELS:
            with self.subTest(model=model):
                result = simulate(cfg, model)
                plus = simulate(replace(cfg, J=cfg.J + delta), model)
                minus = simulate(replace(cfg, J=cfg.J - delta), model)
                finite_difference = (plus.states - minus.states) / (2 * delta)
                np.testing.assert_allclose(result.tangents, finite_difference, atol=2e-8, rtol=2e-7)
                overlap = np.einsum("tik,tik->t", result.states.conj(), finite_difference)
                pure_qfi = 4 * (np.sum(np.abs(finite_difference)**2, axis=(1, 2)) - np.abs(overlap)**2)
                np.testing.assert_allclose(result.qfi_global, pure_qfi, atol=2e-7, rtol=2e-7)

    def test_initial_values_and_information_ordering(self) -> None:
        for model in MODELS:
            with self.subTest(model=model):
                result = simulate(self.cfg, model)
                self.assertLess(result.norm_error, 1e-7)
                np.testing.assert_allclose(result.bloch[0], [1.0, 0.0, 0.0], atol=1e-12)
                for qfi in (result.qfi_bath, result.qfi_central, result.qfi_global):
                    self.assertAlmostEqual(qfi[0], 0.0)
                    self.assertTrue(np.all(qfi <= result.qfi_global + 1e-7))
                self.assertAlmostEqual(result.excitation[0], 0.0)
                self.assertAlmostEqual(result.field_mean[0], self.cfg.J * self.cfg.N)
                self.assertAlmostEqual(result.field_std[0], 0.0)
                self.assertTrue(np.all((result.purity > 0.5 - 1e-9) & (result.purity < 1 + 1e-9)))
                np.testing.assert_allclose(result.populations.sum(axis=1), 1.0, atol=1e-12)

    def test_truncation_guard(self) -> None:
        cfg = SimulationConfig(N=15, Omega=2.0, omega=1.0, J=1.0, points=11, t_pi=0.5, n_max=8, show=False)
        with self.assertRaisesRegex(RuntimeError, "increase --n-max"):
            simulate(cfg, "HP")

    def test_crossing_times(self) -> None:
        np.testing.assert_allclose(crossing_times(SimulationConfig(omega=1.0, t_pi=1.0)),
                                   [np.pi / 4, 3 * np.pi / 4])
        self.assertEqual(crossing_times(SimulationConfig(omega=1.0, t_pi=0.2)).size, 0)
        self.assertEqual(crossing_times(SimulationConfig(omega=0.0)).size, 0)

    def test_export_pairs_full_figure_with_complete_numerical_record(self) -> None:
        cfg = replace(self.cfg, points=11, t_pi=0.2, dpi=50)
        trajectories = {model: simulate(cfg, model) for model in MODELS}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            def save_in_temporary_directory(figure, **kwargs):
                return smart_save.save_plot(figure, output_dir=root, **kwargs)

            with patch.object(smart_save, "GOOGLE_DRIVE_GRAPHS_DIRECTORY", root), patch(
                "CRB.HolsteinPrimakoff.hp_dynamics.save_plot",
                side_effect=save_in_temporary_directory,
            ) as exporter:
                record = plot_trajectories(cfg, trajectories)

            self.assertEqual(exporter.call_count, 1)
            self.assertEqual(len(exporter.call_args.args[0].axes), 7)  # six panels and a colorbar
            self.assertIn("initial=plus", exporter.call_args.kwargs["name"])
            self.assertEqual(list(root.rglob("*.plot.json")), [record.json_path])
            self.assertEqual(list(root.rglob("*.png")), [record.image_path])
            payload = json.loads(record.json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["parameters"], json.loads(json.dumps(asdict(cfg))))
            series = {curve["name"]: curve for curve in payload["series"]}
            for model, result in trajectories.items():
                np.testing.assert_array_equal(series[f"{model}: QFI(J) bath"]["y"], result.qfi_bath)
                np.testing.assert_array_equal(series[f"{model}: <a^dag a>/N"]["y"], result.excitation)
                for component, name in enumerate("XYZ"):
                    np.testing.assert_array_equal(series[f"{model}: <{name}>"]["y"], result.bloch[:, component])
                np.testing.assert_array_equal(series[f"{model}: p_n=0"]["y"], result.populations[:, 0])
            self.assertIn(f"exact: p_n={cfg.N}", series)
            self.assertNotIn(f"exact: p_n={cfg.N + 1}", series)
            for curve in series.values():
                np.testing.assert_array_equal(curve["x"], trajectories["HP"].time)
            with Image.open(record.image_path) as image:
                self.assertEqual(json.loads(image.info["Description"])["record"], record.json_path.name)


if __name__ == "__main__":
    unittest.main()
