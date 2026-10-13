"""Physical checks: conserved momentum, coherent pointers, and LZ phase gauge.

Run: python -m unittest CRB.HolsteinPrimakoff.test_linearized_cat_state -v
"""
import unittest
import numpy as np
from scipy.integrate import quad, solve_ivp

from CRB.HolsteinPrimakoff.linearized_cat_state import CatConfig, adiabatic_impulse, coherent_cat_at, fidelity, ideal_cat, ideal_parameters, lz_parameters, readout_quadrature, stokes_phase
from CRB.HolsteinPrimakoff.linearized_momentum import simulate_momentum


class CatStateTests(unittest.TestCase):
    def test_split_quadrature_agrees_with_exact_default_solver(self):
        from dataclasses import replace
        cfg=CatConfig(readout=1.,points=3,nodes=64,check=96)
        quadrature=readout_quadrature(cfg,cfg.readout,cfg.nodes)
        momentum_cfg=replace(cfg.momentum_config(),tmax=cfg.readout)
        hermite=simulate_momentum(momentum_cfg)
        split=simulate_momentum(momentum_cfg,quadrature=quadrature)
        np.testing.assert_allclose(split.bloch,hermite.bloch,atol=2e-10,rtol=0)
        np.testing.assert_allclose(sum(quadrature[1]*quadrature[0]**2),1,atol=2e-14,rtol=0)

    def test_arbitrary_readout_packet_motion_without_drive(self):
        cfg = CatConfig(N=4,Omega=0.,J=.8,omega=.7,readout=1.5,points=3,nodes=48,check=64)
        from dataclasses import replace
        result = simulate_momentum(replace(cfg.momentum_config(),tmax=cfg.readout))
        predicted = coherent_cat_at(cfg,result.momenta,cfg.readout)
        np.testing.assert_allclose(predicted,result.states[-1],atol=2e-9,rtol=0)
        half = np.pi/(2*cfg.omega)
        np.testing.assert_allclose(coherent_cat_at(cfg,result.momenta,half),ideal_cat(cfg,result.momenta,'half'),atol=2e-15,rtol=0)

    def test_zero_drive_revival_and_half_period_displacement(self):
        cfg = CatConfig(N=4, Omega=0., J=.8, omega=.7, points=41, nodes=48, check=64, rtol=1e-11, atol=1e-13)
        result = simulate_momentum(cfg.momentum_config())
        angle = 2 * cfg.omega * result.time[:, None]
        integral = cfg.J / (2 * cfg.omega) * (cfg.N * np.sin(angle) + np.sqrt(cfg.N) * result.momenta * (1-np.cos(angle)))
        expected = np.stack((np.exp(-1j*integral),np.exp(1j*integral)),axis=-1)/np.sqrt(2)
        np.testing.assert_allclose(result.states,expected,atol=3e-9,rtol=0)
        np.testing.assert_allclose(adiabatic_impulse(cfg,result.time,result.momenta),expected,atol=3e-14,rtol=0)
        for name,index in [('half',cfg.points//2),('full',-1)]:
            self.assertGreater(float(fidelity(ideal_cat(cfg,result.momenta,name),result.states[index],result.weights)),1-1e-12)

    def test_stokes_limits_and_probability_factor(self):
        self.assertAlmostEqual(float(stokes_phase(0)),np.pi/4)
        self.assertGreater(float(stokes_phase(.1)),float(stokes_phase(1)))
        self.assertLess(abs(float(stokes_phase(1000))),1e-4)
        cfg = CatConfig(N=9,Omega=2,J=.8,omega=.6)
        delta,probability,_ = lz_parameters(cfg)
        self.assertAlmostEqual(float(delta),cfg.Omega**2/(4*cfg.J*cfg.N*cfg.omega))
        self.assertAlmostEqual(float(probability),np.exp(-np.pi*cfg.Omega**2/(2*cfg.J*cfg.N*cfg.omega)))

    def test_phase_sign_against_finite_time_linear_lz_sweep(self):
        # Independent direct Schrodinger evolution fixes the signs and gauge.
        coupling,v,limit = 1.,1.,20.
        probability = np.exp(-np.pi*coupling**2/v)
        phi = float(stokes_phase(coupling**2/(2*v)))
        def rotation(t):
            theta = np.arctan2(coupling,-v*t)/2
            c,s = np.cos(theta),np.sin(theta)
            return np.array([[c,-s],[s,c]])
        def rhs(t,y):
            h = np.array([[-v*t,coupling],[coupling,v*t]])
            return (-1j*h@y.reshape(2,2)).ravel()
        solution = solve_ivp(rhs,(-limit,limit),np.eye(2,dtype=complex).ravel(),method='DOP853',rtol=1e-10,atol=1e-12)
        self.assertTrue(solution.success)
        zeta = quad(lambda t: np.hypot(coupling,v*t),0,limit)[0]
        d = np.diag(np.exp(np.array([-1j,1j])*zeta))
        a,b = np.sqrt(probability),np.sqrt(1-probability)
        jump = np.array([[b*np.exp(-1j*phi),a],[-a,b*np.exp(1j*phi)]])
        approx = rotation(limit)@d@jump@d@rotation(-limit).T
        np.testing.assert_allclose(solution.y[:,-1].reshape(2,2),approx,atol=8e-5,rtol=0)

    def test_ideal_cat_norm_and_coherent_pointer_overlap(self):
        cfg = CatConfig()
        result = simulate_momentum(cfg.momentum_config())
        d = ideal_parameters(cfg)['d']
        self.assertAlmostEqual(float(np.sum(result.weights*np.exp(-1j*d*result.momenta)).real),np.exp(-d*d/2),places=13)
        for passage in ['half','full']:
            state = ideal_cat(cfg,result.momenta,passage)
            self.assertAlmostEqual(float(np.sum(result.weights[:,None]*np.abs(state)**2)),1.,places=13)
            self.assertAlmostEqual(float(fidelity(state,np.exp(.3j)*state,result.weights)),1.,places=13)

    def test_full_momentum_aia_is_unitary(self):
        cfg = CatConfig(N=6,Omega=1.7)
        p = np.linspace(-5,5,51)
        t = np.linspace(0,np.pi/cfg.omega,101)
        aia = adiabatic_impulse(cfg,t,p)
        np.testing.assert_allclose(np.sum(np.abs(aia)**2,axis=-1),1,atol=2e-15,rtol=0)
        np.testing.assert_allclose(aia[0],1/np.sqrt(2),atol=2e-16,rtol=0)


if __name__ == '__main__':
    unittest.main()
