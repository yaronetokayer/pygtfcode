"""Velocity-dependent implicit transport and diagnostic regression tests."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from pygtfcode import Config, State
from pygtfcode.evolve import transport as tr
from pygtfcode.evolve.integrator import integrate_time_step, allocate_work_arrays
from pygtfcode.util.calc_kp import conductivity, factors
from pygtfcode.util.calc_runtime import calc_kappa_cell, calc_kappa_edge
from pygtfcode.util.interpolate import interp_linear_to_interfaces


class VelocityTransportTests(unittest.TestCase):
    def mesh(self):
        r = np.r_[0., np.geomspace(.04, 3., 12)]
        return r, r**3 / 3, np.linspace(.8, 1.2, 12), 1 + .2*np.cos(np.arange(12))

    def test_builder_jacobian(self):
        r, m, rho, temperature = self.mesh()
        rho_int = interp_linear_to_interfaces(r, rho)
        dm = np.diff(m)
        dt = .003
        for order in (1, 2):
            for w in (.03, 1., np.inf):
                for alph in (.5, 1., 2.):
                    a, b, c, d = [np.empty(12) for _ in range(4)]
                    tr.build_tridiag_system(r, m, rho_int, temperature, dt,
                                           2.256758, 1.3847, .75, alph, 3.6, w, order, a, b, c, d)
                    def rate(T):
                        flux = np.zeros(13)
                        for i in range(11):
                            k, _ = conductivity(.5*(T[i]+T[i+1]), rho_int[i], 3.6,
                                                w, alph, 2.256758, 1.3847, .75, order)
                            flux[i+1] = 2*r[i+1]**2/(r[i+2]-r[i])*k*(T[i+1]-T[i])
                        return np.diff(flux)/dm
                    jac = np.empty((12, 12))
                    for j in range(12):
                        plus, minus = temperature.copy(), temperature.copy()
                        h = 1e-5*temperature[j]
                        plus[j] += h
                        minus[j] -= h
                        jac[:, j] = (rate(plus)-rate(minus))/(2*h)
                    matrix = np.diag(b)+np.diag(a[1:], -1)+np.diag(c[:-1], 1)
                    actual = matrix/(np.sqrt(2)*dm[:, None])+np.eye(12)/dt
                    self.assertLess(np.max(abs(actual-jac))/np.max(abs(jac)), 1e-8)
                    np.testing.assert_allclose(d, -np.sqrt(2)*dm*rate(temperature), atol=1e-14)

    def test_all_wrappers_conserve_and_agree(self):
        r, m, rho, initial = self.mesh()
        for w in (.2, np.inf):
            for order in (1, 2):
                results = []
                for suffix in ('nolim', 'tcool_nolim', 'Theta_nolim', 'dulim', 'tcool_dulim', 'Theta_dulim'):
                    T = initial.copy()
                    dv = np.empty_like(T)
                    args = [T, rho, r, m, dv]
                    if 'tcool' in suffix or 'Theta' in suffix:
                        args.append(np.empty_like(T))
                    args += [1e-6, 2.256758, 1.3847, .75, 3.6, w, order, 1.]
                    if suffix.endswith('dulim'):
                        args += [.01, 10]
                    _, _, retries = getattr(tr, 'conduct_implicit_'+suffix)(*args)
                    self.assertGreaterEqual(retries, 0)
                    self.assertTrue(np.all(T > 0))
                    self.assertLess(abs(np.dot(np.diff(m), T-initial)), 1e-14)
                    results.append(T)
                for T in results[1:]:
                    np.testing.assert_allclose(T, results[0], rtol=1e-14)

    def test_kappa_outputs_and_constant_limit(self):
        r, m, rho, T = self.mesh()
        rm = .5*(r[1:]+r[:-1])
        for w in (.2, np.inf):
            for order in (1, 2):
                for func, radii, te, de, re in (
                    (calc_kappa_cell, rm, T, rho, rm),
                    (calc_kappa_edge, r, .5*(T[:-1]+T[1:]), interp_linear_to_interfaces(r, rho), r[1:-1]),
                ):
                    kl, ks, k = func(T, rho, radii, 2.256758, 1.3847, .75, 3.6, 1., w, order)
                    for i in range(len(te)):
                        fl, fs, _, _ = factors(te[i], w, order)
                        self.assertAlmostEqual(kl[i], 1.5*re[i]**2*np.sqrt(te[i])*.75*de[i]*te[i]*fl)
                        self.assertAlmostEqual(ks[i], 1.5*re[i]**2*np.sqrt(te[i])*1.3847/(2.256758*3.6**2*fs))
                        coeff, _ = conductivity(te[i], de[i], 3.6, w, 1., 2.256758, 1.3847, .75, order)
                        self.assertAlmostEqual(k[i], 1.5*re[i]**2*coeff)
                    if func is calc_kappa_edge:
                        self.assertTrue(all(np.isnan(x[-1]) for x in (kl, ks, k)))

    def test_constant_builder_matches_legacy(self):
        r, m, rho, T = self.mesh()
        ri = interp_linear_to_interfaces(r, rho)
        new, old = [np.empty((4, 12)) for _ in range(2)]
        tr.build_tridiag_system(r, m, ri, T, .001, 2.256758, 1.3847, .75, 1., 3.6, np.inf, 2, *new)
        tr.build_tridiag_system_ALPH1(r, m, ri, T, 2.256758*3.6**2/1.3847, 1/.75, .001, *old)
        np.testing.assert_allclose(new, old, rtol=2e-13, atol=1e-13)

    def test_saturated_rejection_and_no_mutation(self):
        r = np.arange(7, dtype=float)
        m = r**3/3
        rho = np.ones(6)
        initial = np.ones(6)
        initial[2] = 1.01
        for suffix in ('dulim', 'tcool_dulim', 'Theta_dulim'):
            for maxiter in (1, 10):
                T = initial.copy()
                args = [T, rho, r, m, np.empty(6)]
                if suffix != 'dulim':args.append(np.empty(6))
                args += [1000., 2.256758, 1.38, .75, 1., np.inf, 2, 1., .008, maxiter]
                du, dt, retries = getattr(tr, 'conduct_implicit_'+suffix)(*args)
                if maxiter == 1:
                    self.assertEqual(retries, -1)
                    np.testing.assert_array_equal(T, initial)
                else:
                    self.assertGreaterEqual(retries, 0)
                    self.assertLessEqual(du, .008)
                self.assertLessEqual(dt, 500.)

    def test_state_time_units_and_step(self):
        for w in (50., np.inf):
            s = State(Config(sim=dict(sigma_m_0=300., w=w),
                             prec=dict(du_boost=1.), io=dict(chatter=False, profiles=False, t_evol=False)))
            s.reset()
            factor = s.config.sim.a*s.char.sigma_m_0_char
            np.testing.assert_allclose(s.t_dyn, factor/np.sqrt(s.rho), rtol=1e-14)
            integrate_time_step(s, s.config, 1e-6, 1e-4, 1, *allocate_work_arrays(s.n)[:-1])
            np.testing.assert_allclose(s.t_dyn, factor/np.sqrt(s.rho), rtol=1e-14)
            self.assertTrue(np.all(s.v2 > 0))
            self.assertTrue(np.all(np.diff(s.r) > 0))

    def test_output_and_full_driver(self):
        # Keep all test output within the checkout; remove only this temporary run.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            cfg = Config(sim=dict(sigma_m_0=300., w=50.), prec=dict(du_boost=1.),
                         io=dict(base_dir=folder, model_no=1, chatter=False))
            state = State.from_config(cfg)
            state.run(steps=3)
            self.assertTrue(np.isfinite(state.t) and state.t > 0)
            self.assertTrue(list(Path(folder).rglob('profile_*.dat')))
            np.testing.assert_allclose(state.t_dyn,
                cfg.sim.a*state.char.sigma_m_0_char/np.sqrt(state.rho), rtol=1e-14)


if __name__ == '__main__':
    unittest.main()
