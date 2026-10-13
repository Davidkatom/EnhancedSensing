"""Checks of the p-picture reduction and its adiabatic-impulse closed form.

Run: python -m unittest CRB.HolsteinPrimakoff.test_p_picture -v
"""

import unittest

import numpy as np

from CRB.HolsteinPrimakoff.p_picture import (
    aim_bath_qfi,
    ensemble_amplitudes,
    ensemble_bath_qfi,
    fock_bath_qfi,
    leading_bath_qfi,
    p_grid,
    sweep,
)

LZ = np.sqrt(2 * np.log(2) / np.pi)


class PPictureTests(unittest.TestCase):
    def test_each_p_sees_the_same_sweep_shifted_in_time(self) -> None:
        N, J, omega = 15, 0.8, 1.3
        t = np.linspace(0, 2, 101)
        for p in (-2.0, 0.5, 1.7):
            theta = np.arctan(p / np.sqrt(N))
            np.testing.assert_allclose(sweep(p, t, N, J, omega),
                                       J * np.sqrt(N**2 + N * p**2) * np.cos(2 * omega * t - theta), atol=1e-12)

    def test_p_reduction_matches_the_fock_basis_model(self) -> None:
        for N, Omega, t in ((8, 1.9, 1.0), (15, 2.6, 1.2)):
            with self.subTest(N=N):
                self.assertAlmostEqual(ensemble_bath_qfi(N, Omega, t) / fock_bath_qfi(N, Omega, t), 1.0, places=5)

    def test_bath_P_distribution_is_conserved(self) -> None:
        p, weight = p_grid()
        c = ensemble_amplitudes(15, 2.6, 1.2, 1.0, 1.0, p)
        np.testing.assert_allclose(np.sum(np.abs(weight * c) ** 2, axis=0), weight**2, atol=1e-10)

    def test_adiabatic_impulse_tracks_the_ensemble(self) -> None:
        for N in (15, 30):
            Omega = LZ * np.sqrt(N)
            with self.subTest(N=N):
                self.assertLess(abs(aim_bath_qfi(N, Omega, 1.0) / ensemble_bath_qfi(N, Omega, 1.0) - 1), 0.03)

    def test_leading_order_is_the_large_N_limit(self) -> None:
        N = 100
        Omega = LZ * np.sqrt(N)
        self.assertAlmostEqual(leading_bath_qfi(N, Omega) / N**2, 1.0, places=12)
        self.assertLess(abs(ensemble_bath_qfi(N, Omega, 1.0) / leading_bath_qfi(N, Omega) - 1), 0.02)

    def test_leading_order_peaks_at_the_lz_drive(self) -> None:
        N, J, omega = 20, 0.7, 1.4
        grid = np.linspace(0.5, 1.0, 501) * np.sqrt(J * N * omega)
        best = grid[np.argmax([leading_bath_qfi(N, Omega, J, omega) for Omega in grid])]
        self.assertAlmostEqual(best / np.sqrt(J * N * omega), LZ, delta=2e-3)


if __name__ == "__main__":
    unittest.main()
