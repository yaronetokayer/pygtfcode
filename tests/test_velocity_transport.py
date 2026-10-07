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

    def check_diagnostics(self, state):
        from pygtfcode.util.calc_core import calc_core_r, calc_logmean_within_r
        sigma = state.char.sigma_m_0_char
        what = state.char.w_char
        moments = np.array([factors(t, what, state.config.sim.smfp_order) for t in state.v2])
        scale = np.sqrt(moments[:, 0])*np.sqrt(moments[:, 1])
        np.testing.assert_allclose(state.kn, 1/(sigma*np.sqrt(state.rho*state.v2)))
        np.testing.assert_allclose(state.mfp, 1/(sigma*state.rho))
        np.testing.assert_allclose(state.kn_cond, state.kn/scale)
        np.testing.assert_allclose(state.mfp_cond, state.mfp/scale)
        np.testing.assert_allclose(state.mfp_cond, np.sqrt(state.v2/state.rho)*state.kn_cond)
        rc = calc_core_r(state.r, state.rmid, state.rho)
        self.assertAlmostEqual(state.kn_c, calc_logmean_within_r(state.r, state.m, state.kn, rc))
        self.assertAlmostEqual(state.kn_cond_c, calc_logmean_within_r(state.r, state.m, state.kn_cond, rc))
        kl, ks, _ = calc_kappa_cell(state.v2, state.rho, state.rmid,
            state.config.sim.a, state.config.sim.b, state.config.sim.c, sigma,
            state.config.sim.alph, what, state.config.sim.smfp_order)
        np.testing.assert_allclose(ks/kl,
            state.config.sim.b/(state.config.sim.a*state.config.sim.c)*state.kn_cond**2)

    def test_transport_diagnostics_lifecycle(self):
        from pygtfcode.evolve.split import split_grid, merge_grid
        for w in (50., np.inf):
            for order in (1, 2):
                s = State(Config(sim=dict(w=w, smfp_order=order), io=dict(chatter=False)))
                s.reset()
                self.check_diagnostics(s)
                if np.isinf(w):
                    np.testing.assert_array_equal(s.kn_cond, s.kn)
                    np.testing.assert_array_equal(s.mfp_cond, s.mfp)
                mask = np.zeros(s.n, dtype=np.int64)
                mask[20] = 1
                split_grid(s, mask)
                s.resize_state_arrays()
                self.check_diagnostics(s)
                mask = np.zeros(s.n, dtype=np.int64)
                mask[20] = 1
                merge_grid(s, mask)
                s.resize_state_arrays()
                self.check_diagnostics(s)
                integrate_time_step(s, s.config, 1e-6, 1e-4, 1, *allocate_work_arrays(s.n)[:-1])
                self.check_diagnostics(s)

    def test_default_w_and_metadata(self):
        from pygtfcode.io.write import write_metadata
        from pygtfcode.io.read import import_metadata
        self.assertTrue(np.isposinf(Config().sim.w))
        self.assertEqual(Config().prec.du_boost, 100.)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            for w in (50., np.inf):
                cfg = Config(sim=dict(w=w), io=dict(base_dir=folder, model_no=1, chatter=False))
                state = State.from_config(cfg)
                write_metadata(state)
                rebuilt = Config.from_dict(import_metadata(Path(folder)/cfg.io.model_dir))
                self.assertEqual(rebuilt.sim.w, w)
                self.assertEqual(rebuilt.sim.smfp_order, 2)

    def test_output_schema_and_plotting(self):
        import re
        import matplotlib.pyplot as plt
        from pygtfcode.io.read import extract_time_evolution_data, extract_snapshot_data
        from pygtfcode.io.write import write_log_entry
        from pygtfcode.plot.time_evolution import plot_time_evolution
        from pygtfcode.plot.snapshot import plot_profile
        for w in (50., np.inf):
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
                cfg = Config(sim=dict(w=w), grid=dict(drfrac_init=.15),
                             prec=dict(du_boost=1.),
                             io=dict(base_dir=folder, model_no=1, chatter=False))
                state = State.from_config(cfg)
                state.run(steps=3)
                model = Path(folder)/cfg.io.model_dir
                data = extract_time_evolution_data(model/'time_evolution.txt')
                self.assertNotIn('te', data)
                for suffix in ('c', 'm2'):
                    x = np.sqrt(data['v2_'+suffix])/state.char.w_char
                    np.testing.assert_allclose(data['x_'+suffix], x, rtol=2e-6)
                    for j, T in enumerate(data['v2_'+suffix]):
                        kl, ks, _, _ = factors(T, state.char.w_char, cfg.sim.smfp_order)
                        self.assertAlmostEqual(data['K_L_'+suffix][j]/kl, 1., places=5)
                        self.assertAlmostEqual(data['K_S_'+suffix][j]/ks, 1., places=5)
                expected = data['r_c']/np.sqrt(data['v2_c'])*cfg.sim.a*state.char.sigma_m_0_char
                np.testing.assert_allclose(data['tsc_c'], expected, rtol=2e-6)
                snap = extract_snapshot_data(model/'profile_0.dat')
                np.testing.assert_allclose(snap['mfp_cond_ltemp'], snap['mfp_cond']/snap['ltemp'], rtol=2e-6)
                for key in ('x', 'K_L', 'K_S', 'kn_cond', 'mfp_cond', 'mfp_cond_ltemp', 'krat_e', 'k_se'):
                    fig, ax = plt.subplots()
                    plot_profile(ax, key, [snap])
                    if key in ('krat_e', 'k_se'):
                        np.testing.assert_allclose(ax.lines[0].get_xdata(), 10**snap['log_r'])
                    fig.canvas.draw()
                    plt.close(fig)
                for key in ('x_c', 'x_m2', 'K_L_c', 'K_S_m2', 'n', 'tsc_c'):
                    fig, ax = plot_time_evolution(cfg, quantity=key, show=False)
                    fig.canvas.draw()
                    if np.isinf(w) and key.startswith('x_'):
                        self.assertEqual(ax.get_yscale(), 'linear')
                    plt.close(fig)
                # Verify interval means use the actual per-step tolerance.
                state.du_limit_cum = .2 + .8
                state.du_max_cum = 999.  # Must not be used in <du lim>.
                state.log_steps = 2
                state.n_split, state.n_merge = 3, 2
                write_log_entry(state, 0)
                lines = (model/'logfile.txt').read_text().splitlines()
                keys = re.split(r' {2,}', lines[0].strip())
                values = lines[-1].split()
                row = dict(zip(keys, values))
                self.assertAlmostEqual(float(row['<du lim>']), .5)
                self.assertEqual(int(row['n_split']), 3)
                self.assertEqual(int(row['n_merge']), 2)
                self.assertIn('<n_retry_du>', row)
                self.assertEqual(state.log_steps, 0)

    def test_output_and_full_driver(self):
        # Keep all test output within the checkout; remove only this temporary run.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            cfg = Config(sim=dict(sigma_m_0=300., w=50.), prec=dict(du_boost=1.),
                         io=dict(base_dir=folder, model_no=1, chatter=False))
            state = State.from_config(cfg)
            state.run(steps=3)
            self.assertTrue(np.isfinite(state.t) and state.t > 0)
            profiles = list(Path(folder).rglob('profile_*.dat'))
            self.assertTrue(profiles)
            header = profiles[0].read_text().splitlines()[0].split()
            self.assertIn('kn_cond', header)
            self.assertIn('mfp_cond', header)
            for profile in profiles:
                rows = np.loadtxt(profile, skiprows=1)
                self.assertEqual(rows.shape[1], len(header))
            history = (Path(folder)/cfg.io.model_dir/'time_evolution.txt').read_text()
            self.assertIn('kn_cond_c', history.splitlines()[0])
            self.check_diagnostics(state)
            np.testing.assert_allclose(state.t_dyn,
                cfg.sim.a*state.char.sigma_m_0_char/np.sqrt(state.rho), rtol=1e-14)


if __name__ == '__main__':
    unittest.main()
