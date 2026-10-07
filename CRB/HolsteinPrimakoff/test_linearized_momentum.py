"""Independent checks of vacuum momentum integration for the reduced HP model."""

import unittest

import numpy as np
from scipy.integrate import solve_ivp

from CRB.HolsteinPrimakoff.hp_dynamics import PAULI, SimulationConfig, bath_operators
from CRB.HolsteinPrimakoff.linearized_momentum import MomentumConfig, simulate_momentum
from CRB.crb_core import central_spin_state


class MomentumTests(unittest.TestCase):
    def test_zero_drive_matches_analytic_vacuum_dephasing(self):
        cfg = MomentumConfig(N=4, Omega=0.0, J=0.8, omega=0.7, tmax=1.2, points=61, p_nodes=48)
        result = simulate_momentum(cfg)
        angle = 2 * cfg.omega * result.time
        phase = cfg.N * np.sin(angle) / (2 * cfg.omega)
        displacement = np.sqrt(cfg.N) * (1 - np.cos(angle)) / (2 * cfg.omega)
        envelope = np.exp(-2 * (cfg.J * displacement) ** 2)
        expected = np.column_stack((envelope * np.cos(2 * cfg.J * phase),
                                    envelope * np.sin(2 * cfg.J * phase), np.zeros(cfg.points)))
        np.testing.assert_allclose(result.bloch, expected, atol=2e-8, rtol=0)
        self.assertLess(result.norm_error, 1e-7)

    def test_nonzero_drive_matches_boson_vacuum_evolution(self):
        cfg = MomentumConfig(N=3, Omega=0.9, J=0.8, omega=0.7, tmax=1.0, points=41, p_nodes=48)
        result = simulate_momentum(cfg)
        ops = bath_operators(SimulationConfig(N=cfg.N, n_max=40), "HP")
        initial = np.zeros((2, ops.B_c.size), dtype=complex)
        initial[:, 0] = central_spin_state(cfg.theta, cfg.phi).full().ravel()
        z_signs = np.array([[1], [-1]])

        def rhs(t, flat):
            state = flat.reshape(initial.shape)
            coupling = cfg.N * np.cos(2 * cfg.omega * t) * state + np.sin(2 * cfg.omega * t) * state @ ops.B_s.T
            return (-1j * (cfg.Omega * state[::-1] + cfg.J * z_signs * coupling)).ravel()

        solution = solve_ivp(rhs, (0, cfg.tmax), initial.ravel(), t_eval=result.time,
                             method="DOP853", rtol=1e-10, atol=1e-12)
        self.assertTrue(solution.success)
        states = solution.y.T.reshape(cfg.points, *initial.shape)
        rho_c = np.einsum("tik,tjk->tij", states, states.conj())
        expected = np.einsum("tij,aji->ta", rho_c, PAULI).real
        np.testing.assert_allclose(result.bloch, expected, atol=2e-8, rtol=0)


if __name__ == "__main__":
    unittest.main()
