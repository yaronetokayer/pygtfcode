"""Regression tests for package API, output, and diagnostic consistency."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from pygtfcode import Config, State
from pygtfcode.io.read import extract_snapshot_data, extract_time_evolution_data, load_snapshot_bundle
from pygtfcode.plot.snapshot import plot_snapshots, plot_profile, plot_plummer


class PackageConsistencyTests(unittest.TestCase):
    def tempdir(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])

    def config(self, folder, **kwargs):
        return Config(grid=dict(drfrac_init=.2), prec=dict(du_boost=1),
                      io=dict(base_dir=folder, model_no=1, chatter=False), **kwargs)

    def test_util_is_a_regular_package(self):
        import pkgutil
        import pygtfcode
        import pygtfcode.util
        self.assertEqual(Path(pygtfcode.util.__file__).name, '__init__.py')
        packages = {item.name: item.ispkg for item in pkgutil.iter_modules(pygtfcode.__path__)}
        self.assertTrue(packages['util'])

    def test_boost_uses_conductivity_core_in_driver_and_log(self):
        import re
        from pygtfcode.evolve import integrator
        from pygtfcode.util.calc_runtime import low_kn_boost
        for w in (50., np.inf):
            with self.subTest(w=w), self.tempdir() as folder, contextlib.redirect_stdout(io.StringIO()):
                cfg = self.config(folder, sim=dict(w=w))
                cfg.prec.du_boost = 100.
                state = State.from_config(cfg)
                prec = cfg.prec
                def effective(kn):
                    return prec.eps_du * low_kn_boost(kn, prec.kn_threshold,
                                                     prec.du_boost, prec.kn_width)
                expected = effective(state.kn_cond_c)
                if np.isfinite(w):
                    self.assertLess(expected, effective(state.kn_c))
                else:
                    self.assertEqual(expected, effective(state.kn_c))
                with patch.object(integrator, 'integrate_time_step',
                                  wraps=integrator.integrate_time_step) as step:
                    state.run(steps=1)
                    self.assertEqual(step.call_args.args[3], expected)
                lines = (Path(folder)/cfg.io.model_dir/'logfile.txt').read_text().splitlines()
                keys = re.split(r' {2,}', lines[0].strip())
                row = dict(zip(keys, lines[-1].split()))
                self.assertAlmostEqual(float(row['eps_du_eff'])/effective(state.kn_cond_c),
                                       1., places=6)

    def test_parameter_validation(self):
        from pygtfcode.parameters import SimParams, GridParams, PrecisionParams, IOParams
        for key in ('sigma_m_0', 'a', 'b', 'c', 'alph'):
            for invalid in (0., -1., np.nan, np.inf, True):
                with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                    SimParams(**{key: invalid})
        for invalid in (0., -1., np.nan, -np.inf, True):
            with self.assertRaises(ValueError):
                SimParams(w=invalid)
        self.assertTrue(np.isinf(SimParams(w=np.inf).w))
        self.assertTrue(np.isinf(SimParams(t_halt=np.inf).t_halt))
        with self.assertRaises(ValueError):
            SimParams(smfp_order=True)
        with self.assertRaises(ValueError):
            PrecisionParams(max_iter_du=0)
        with self.assertRaises(ValueError):
            GridParams(drfrac_min=.2, drfrac_max=.1)
        for key in ('nlog', 'nupdate'):
            with self.assertRaises(ValueError):
                IOParams(**{key: 0})
        self.assertIn('Model00001', repr(IOParams(model_no=1)))

    def test_profile_scalar_array_contracts(self):
        from pygtfcode.profiles.nfw import sigr_nfw
        from pygtfcode.profiles.abg import menc_abg, sigr_abg
        for function, config in ((sigr_nfw, Config()),
                                 (menc_abg, Config(init=('abg', dict(alpha=2., beta=5., gamma=0.)))),
                                 (sigr_abg, Config(init=('abg', dict(alpha=2., beta=5., gamma=0.))))):
            scalar = function(1., config)
            single = function(np.array([1.]), config)
            matrix = function(np.ones((1, 2)), config)
            self.assertIsInstance(scalar, float)
            self.assertEqual(single.shape, (1,))
            self.assertEqual(matrix.shape, (1, 2))
            np.testing.assert_allclose(single, scalar)
            np.testing.assert_allclose(matrix, scalar)
        # Analytic Plummer values in this package's mass/velocity units.
        cfg = Config(init=('abg', dict(alpha=2., beta=5., gamma=0.)))
        self.assertAlmostEqual(menc_abg(1., cfg), 1/(3*2**1.5))
        self.assertAlmostEqual(sigr_abg(1., cfg), 1/(18*np.sqrt(2)))

    def test_repeat_run_preserves_history_and_bundle(self):
        with self.tempdir() as folder, contextlib.redirect_stdout(io.StringIO()):
            state = State.from_config(self.config(folder))
            state.run(steps=2)
            model = Path(folder)/state.config.io.model_dir
            before = extract_time_evolution_data(model/'time_evolution.txt')
            old_step = state.step_count
            state.run(steps=2)
            after = extract_time_evolution_data(model/'time_evolution.txt')
            np.testing.assert_array_equal(after['step'][:len(before['step'])], before['step'])
            self.assertEqual(after['step'][-1], old_step+2)
            for name in ('dlnmc_dlnvc', 'dlnrhocdlnvc', 'zeta_balb'):
                self.assertEqual(len(after[name]), len(after['step']))
            first = load_snapshot_bundle(model, 0)
            latest = load_snapshot_bundle(model)
            self.assertEqual(first['step_count'], 0)
            self.assertEqual(latest['step_count'], state.step_count)
            self.assertAlmostEqual(latest['time']/state.t, 1., places=5)
            self.assertTrue(all(np.isfinite(v) for v in state.get_phys().values()))
            with self.assertRaises(ValueError):
                load_snapshot_bundle(model, 999)
            state.config.io.overwrite = False
            original = (model/'model_metadata.txt').read_bytes()
            with self.assertRaises(FileExistsError):
                State.from_config(state.config)
            self.assertEqual((model/'model_metadata.txt').read_bytes(), original)

    def test_ic_grid_and_initialized_diagnostics(self):
        with self.tempdir() as folder:
            cfg = self.config(folder)
            original = State.from_config(cfg)
            path = Path(folder)/cfg.io.model_dir/'profile_0.dat'
            newcfg = self.config(folder)
            newcfg.io.model_no = 2
            newcfg.grid.drfrac_init = .1
            with self.assertWarns(RuntimeWarning):
                loaded = State.from_config(newcfg, ic_filepath=path)
            self.assertEqual(loaded.n, original.n)
            self.assertEqual(loaded.r.size, original.r.size)
            self.assertEqual(loaded.t_cool.size, loaded.n)
            self.assertTrue(np.isinf(loaded.t_cool).all())
            expected = np.diff(loaded.r[1:])/np.sqrt(loaded.r[1:-1]*loaded.r[2:])
            np.testing.assert_allclose(loaded.drfrac[1:], expected)
            for name in ('kn_cond', 'mfp_cond', 't_dyn'):
                self.assertTrue(np.isfinite(getattr(loaded, name)).all())

    def test_zero_gradients_and_small_diagnostics(self):
        from pygtfcode.util.calc_runtime import calc_ltemp
        from pygtfcode.util.calc_slopes import calc_dlnrho_dlnr, calc_dlnv_dlnr, calc_dlogrho_dlogp
        T = np.ones(5); r = np.arange(1., 6.); output = np.empty(5)
        calc_ltemp(output, T, r)
        self.assertTrue(np.isnan(output[:2]).all())
        self.assertTrue(np.isinf(output[2:]).all())
        self.assertTrue(np.isnan(calc_dlogrho_dlogp(T, T)).all())
        for n in (0, 1, 2):
            for function in (calc_dlnrho_dlnr, calc_dlnv_dlnr):
                result = function(T[:n], r[:n])
                self.assertEqual(result.size, n)
                if n == 2:
                    self.assertEqual(result[-1], 0.)

    def test_plot_api_and_plummer_normalization(self):
        with self.tempdir() as folder:
            state = State.from_config(self.config(folder))
            snapshots = [-1]
            with patch('matplotlib.pyplot.show') as show:
                fig, axs = plot_snapshots(state, snapshots=snapshots,
                                          profiles=('rho', 'kn_cond'), xaxis='m', show=False)
                self.assertEqual(snapshots, [-1])
                show.assert_not_called()
                fig.canvas.draw(); plt.close(fig)
            data = extract_snapshot_data(Path(folder)/state.config.io.model_dir/'profile_0.dat')
            for key in ('dttcool', 'tdyntcool'):
                fig, ax = plt.subplots()
                plot_profile(ax, key, [data])
                fig.canvas.draw(); plt.close(fig)
            fig, ax = plt.subplots()
            r = np.geomspace(.1, 10., 101)
            rho = (1+r*r)**-2.5
            data = dict(log_rmid=np.log10(r), rho=rho)
            plot_plummer(ax, 'v2', data, 1.)
            np.testing.assert_allclose(ax.lines[0].get_ydata(), 1/(18*np.sqrt(1+r*r)), rtol=1e-12)
            plt.close(fig)

    def test_nonfinite_implicit_solution_is_rejected(self):
        from pygtfcode.evolve import transport as tr
        r = np.r_[0., np.geomspace(.1, 2., 8)]
        m = r**3/3; rho = np.ones(8); T = np.ones(8)
        def invalid_solve(a,b,c,d,out):
            out[:] = np.nan
        with patch.object(tr, 'solve_tridiagonal_thomas', invalid_solve):
            for name in ('conduct_implicit_dulim', 'conduct_implicit_tcool_dulim', 'conduct_implicit_Theta_dulim'):
                temperature = T.copy(); diagnostic = np.full(8, -7.)
                args = [temperature, rho, r, m, np.empty(8)]
                if name != 'conduct_implicit_dulim':args.append(diagnostic)
                result = getattr(tr, name).py_func(*args, .01, 2., 1., .75, 3., np.inf, 2, 1., 1e-4, 2)
                self.assertEqual(result[-1], -1)
                self.assertTrue(np.isinf(result[0]))
                self.assertGreater(result[1], 0.)
                np.testing.assert_array_equal(temperature, T)
                np.testing.assert_array_equal(diagnostic, -7.)

    def test_implicit_theta_uses_amplitude_time_units(self):
        from pygtfcode.evolve import transport as tr
        r = np.r_[0., np.geomspace(.1, 2., 8)]
        m = r**3/3; rho = np.ones(8); T = np.linspace(1., 1.1, 8)
        old = T.copy(); dv = np.empty(8); theta = np.empty(8)
        _, dt, status = tr.conduct_implicit_Theta_dulim(T, rho, r, m, dv, theta, 1e-4, 2., 1., .75, 3., np.inf, 2, 1., .01, 10)
        self.assertEqual(status, 0)
        expected = old*dt/abs(dv)/(2*3*np.diff(r)/np.sqrt(old))
        np.testing.assert_allclose(theta, expected)


if __name__ == '__main__':
    unittest.main()
