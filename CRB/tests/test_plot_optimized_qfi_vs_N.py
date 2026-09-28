"""Numerical checks: python -m unittest CRB.tests.test_plot_optimized_qfi_vs_N -v."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import qutip as qt
from scipy.linalg import expm_frechet

from CRB import plot_optimized_qfi_vs_N as sweep
from CRB.crb_core import (
    build_hamiltonian, build_spin_operators, central_spin_state,
    coherent_bath_state, qfi_from_rho_and_drho,
)


class OptimizedQFITests(unittest.TestCase):
    def test_subsystem_qfi_matches_exact_matrix_exponential_derivative(self):
        """An independent J derivative catches changed drives/time/initial state."""
        cfg = sweep.OptimizedQFIConfig(dJ=1e-4)
        for N, Omega in ((1, 0.37), (3, 1.6), (5, 3.1)):
            with self.subTest(N=N):
                hamiltonian = build_hamiltonian(Omega, cfg.J_nominal, N, cfg.omega).full()
                operators = build_spin_operators(N)
                dH = (operators["sz_s"] * operators["Sz_op"]).full()
                U, dU = expm_frechet(-1j * cfg.t_max * hamiltonian, -1j * cfg.t_max * dH)
                initial = np.kron(
                    central_spin_state(cfg.central_theta_rad).full().ravel(),
                    coherent_bath_state(N, cfg.bath_theta_rad),
                )
                state, derivative = U @ initial, dU @ initial
                exact_global = 4 * (np.vdot(derivative, derivative).real
                                    - abs(np.vdot(state, derivative)) ** 2)
                rho = np.outer(state, state.conj())
                drho = np.outer(derivative, state.conj()) + np.outer(state, derivative.conj())
                dims = [[2, N + 1], [2, N + 1]]
                bath_rho = qt.Qobj(rho, dims=dims).ptrace(1).full()
                bath_drho = qt.Qobj(drho, dims=dims).ptrace(1).full()
                exact_bath, _ = qfi_from_rho_and_drho(bath_rho, bath_drho, cfg.qfi_tol)
                bath, global_qfi = sweep.qfi_at_optimum(N, Omega, cfg)
                np.testing.assert_allclose([bath, global_qfi], [exact_bath, exact_global],
                                           rtol=2e-5, atol=1e-8)
                self.assertLessEqual(bath, global_qfi + 1e-8)

    def test_refines_all_resolved_peaks_including_lower_coarse_peak(self):
        cfg = sweep.OptimizedQFIConfig(Omega_min=0.1, Omega_max=1, Omega_points=10,
                                      Omega_tol=1e-7)
        grid = np.linspace(cfg.Omega_min, cfg.Omega_max, cfg.Omega_points)

        def two_peaks(Omega, omega, map_cfg, bath):
            return np.exp(-((Omega - 0.225) / 0.08) ** 2) + 1.2 * np.exp(-((Omega - 0.735) / 0.05) ** 2)

        with patch.object(sweep, "qfi_at_drive_pair", side_effect=two_peaks):
            optimum, scan = sweep.optimize_Omega(1, cfg, grid)
        self.assertLess(grid[np.argmax(scan)], 0.5)
        self.assertAlmostEqual(optimum, 0.735, places=5)

    def test_exact_search_endpoints_are_candidates(self):
        cfg = sweep.OptimizedQFIConfig(Omega_points=5)
        grid = np.linspace(cfg.Omega_min, cfg.Omega_max, cfg.Omega_points)
        for sign, expected in ((1, cfg.Omega_max), (-1, cfg.Omega_min)):
            with patch.object(sweep, "qfi_at_drive_pair",
                              side_effect=lambda Omega, *args: 10 + sign * Omega):
                optimum, _ = sweep.optimize_Omega(1, cfg, grid)
            self.assertEqual(optimum, expected)

    def test_small_sweep_uses_same_optimum_for_bath_and_global(self):
        cfg = sweep.OptimizedQFIConfig(N_max=3, Omega_points=21)
        with redirect_stdout(io.StringIO()):
            result = sweep.run_sweep(cfg)
        np.testing.assert_array_equal(result.N_values, [1, 2, 3])
        self.assertTrue(np.all(result.qfi_bath <= result.qfi_global + 1e-8))
        self.assertTrue(np.all(result.qfi_bath >= result.bath_qfi_scan.max(axis=1) - 1e-7))
        for i, N in enumerate(result.N_values):
            np.testing.assert_allclose(
                [result.qfi_bath[i], result.qfi_global[i]],
                sweep.qfi_at_optimum(int(N), result.Omega_opt[i], cfg),
            )
        np.testing.assert_allclose(result.ratio, result.qfi_bath / result.qfi_global)

    def test_zero_information_ratio_is_undefined(self):
        cfg = replace(sweep.OptimizedQFIConfig(), N_max=1, t_max=0, Omega_points=3)
        with redirect_stdout(io.StringIO()):
            result = sweep.run_sweep(cfg)
        np.testing.assert_allclose(result.qfi_bath, 0, atol=1e-15)
        np.testing.assert_allclose(result.qfi_global, 0, atol=1e-15)
        self.assertTrue(np.isnan(result.ratio[0]))


if __name__ == "__main__":
    unittest.main()
