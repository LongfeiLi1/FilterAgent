"""Self-authored, deterministic synthetic tests. No ngspice or real circuits."""
import json
from pathlib import Path
import unittest
import numpy as np
from newbench.metrics import profile_for, ideal_response, fit_response, band_metrics, evaluate_response


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.f = np.geomspace(1., 1e8, 801)
        self.p = profile_for('T2', 'L1')
        self.h = ideal_response(self.f, self.p)

    def evaluate(self, h=None, p=None, f=None):
        return evaluate_response(self.f if f is None else f, self.h if h is None else h,
                                 self.p if p is None else p, fit_max_component_error=.05)

    def test_all_nine_synthetic_profiles_not_full_case_acceptance(self):
        for tid in ('T1', 'T2', 'T3'):
            for lid in ('L1', 'L2', 'L3'):
                with self.subTest(type=tid, level=lid):
                    p = profile_for(tid, lid)
                    r = self.evaluate(ideal_response(self.f, p), p)
                    self.assertEqual(r['response_status'], 'pass', r)
                    self.assertFalse(r['runnable'])
                    self.assertFalse(r['full_circuit_acceptance'])
                    self.assertAlmostEqual(r['f0_hz'], p.f0_hz, places=6)
                    self.assertAlmostEqual(r['q'], p.q, places=10)

    def test_hp_not_dc_normalized(self):
        r = self.evaluate()
        self.assertGreater(r['reference_gain'], .99)
        self.assertLess(abs(self.h[0]), 1e-8)
        self.assertEqual(r['reference_hz'], 40000.)
        self.assertGreater(r['stopband_min_attenuation_db'], 20.)

    def test_hp_finite_upper_band_ignores_out_of_band_collapse(self):
        h = self.h.copy()
        h[self.f > 1e6] *= 1e-9
        a, b = band_metrics(self.f, self.h, self.p), band_metrics(self.f, h, self.p)
        self.assertEqual(a['passband_max_loss_db'], b['passband_max_loss_db'])
        self.assertEqual(a['stopband_min_attenuation_db'], b['stopband_min_attenuation_db'])

    def test_gain_is_independent_gate(self):
        normal = self.evaluate()
        low = self.evaluate(.8*self.h)
        self.assertAlmostEqual(normal['passband_max_loss_db'], low['passband_max_loss_db'])
        self.assertAlmostEqual(normal['stopband_min_attenuation_db'], low['stopband_min_attenuation_db'])
        self.assertIn('reference_gain', low['failed_constraints'])

    def test_t3_signed_complex_fit_rejects_wrong_polarity(self):
        p = profile_for('T3', 'L1')
        h = ideal_response(self.f, p)
        good = self.evaluate(h, p)
        bad = self.evaluate(-h, p)
        self.assertAlmostEqual(good['fitted_signed_gain'], -1.)
        self.assertIn('polarity', bad['failed_constraints'])
        self.assertGreater(bad['fit_max_component_error'], .05)

    def test_hp_complex_fit_rejects_lowpass_shape(self):
        wrong = ideal_response(self.f, profile_for('T1', 'L1'))
        r = self.evaluate(wrong)
        self.assertIn('fit_max_component_error', r['failed_constraints'])

    def test_fit_recovers_non_nominal_f0_q_gain(self):
        h = ideal_response(self.f, self.p, gain=.98, f0_hz=self.p.f0_hz*1.03, q=.8)
        r = self.evaluate(h)
        self.assertAlmostEqual(r['f0_hz'], self.p.f0_hz*1.03, places=6)
        self.assertAlmostEqual(r['q'], .8, places=9)
        self.assertIn('f0_hz', r['failed_constraints'])
        self.assertIn('q', r['failed_constraints'])

    def test_log_log_edge_interpolation(self):
        # Exact interpolation for any power law; all boundaries deliberately off grid.
        f = np.geomspace(1., 1e8, 601)
        h = f.astype(complex)**.25
        r = band_metrics(f, h, self.p)
        self.assertAlmostEqual(r['reference_gain'], 40000.**.25, places=10)
        self.assertAlmostEqual(r['passband_max_loss_db'], -5*np.log10(20000./40000.), places=10)
        self.assertAlmostEqual(r['stopband_min_attenuation_db'], -5*np.log10(3500./40000.), places=10)

    def test_lowpass_single_sided_droop(self):
        p = profile_for('T1', 'L1')
        # Peak is not a ripple penalty. This test addresses bands only, not shape fitting.
        h = np.ones_like(self.f, dtype=complex)
        h[(self.f > 100) & (self.f < 1000)] = 1.5
        r = band_metrics(self.f, h, p)
        self.assertEqual(r['passband_max_loss_db'], 0.)

    def test_nonfinite_zero_and_bad_grid_fail_closed(self):
        for value in (0., complex(float('nan')), complex(float('inf')), complex(1.7e308, 1.7e308)):
            with self.subTest(value=value):
                h = self.h.copy(); h[10] = value
                with self.assertRaises(ValueError): self.evaluate(h)
        for f in (self.f[::-1], np.r_[0., self.f[1:]], np.r_[self.f[0], self.f[:-1]], np.r_[np.nan, self.f[1:]]):
            with self.assertRaises(ValueError): self.evaluate(f=f)

    def test_missing_boundaries_and_reference_fail_closed(self):
        for keep in (self.f > 1., self.f < 1e4, (self.f >= 1e3) & (self.f <= 1e6)):
            with self.assertRaises(ValueError): self.evaluate(self.h[keep], f=self.f[keep])

    def test_insufficient_fitting_points_fail_closed(self):
        f = np.geomspace(1, 1e8, 20)
        with self.assertRaisesRegex(ValueError, '20 points'):
            fit_response(f, ideal_response(f, self.p), self.p)

    def test_mismatched_or_multidimensional_arrays_fail_closed(self):
        for f, h in ((self.f, self.h[:-1]), (self.f[:, None], self.h[:, None]), (self.f[:2], self.h[:2])):
            with self.assertRaises(ValueError): self.evaluate(h, f=f)

    def test_fit_gate_requires_explicit_positive_limit(self):
        for lim in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError): evaluate_response(self.f, self.h, self.p, fit_max_component_error=lim)

    def test_missing_band_samples_fail_closed(self):
        f = np.r_[1., np.geomspace(10000., 1e8, 100)]
        with self.assertRaisesRegex(ValueError, 'samples'):
            band_metrics(f, ideal_response(f, self.p), self.p)

    def test_legacy_t1_band_consistency_readonly(self):
        # Frozen legacy code is only imported/called with synthetic arrays and scalar fixtures.
        from rc_agent.baseline import task_metrics
        from rc_agent.physical import fit_lowpass
        config = json.loads((Path(__file__).resolve().parents[2]/'configs/t1_baseline_full.json').read_text())
        for level in ('L1', 'L3'):
            p = profile_for('T1', level)
            task=dict(f0_hz=p.f0_hz,q=p.q,f0_relative_tolerance=p.f0_relative_tolerance,q_relative_tolerance=p.q_relative_tolerance,passband_edge_hz=p.passband_hz[1],stopband_edge_hz=p.stopband_hz[0],passband_max_loss_db=p.passband_max_loss_db,stopband_min_attenuation_db=p.stopband_min_attenuation_db,power_limit_w=2e-4,phase_margin_min_deg=config['phase_margin_min_deg'])
            h = ideal_response(self.f, p, gain=.985)
            m = fit_lowpass(self.f, h, p.f0_hz)
            m.update(power_w=1e-4, phase_margin_deg=70., dc_gain=.985, dc_out=1.2,
                     minimum_saturation_margin_v=.2, step_final_error_v=0., step_tail_peak_to_peak_v=0.,
                     stable=True, full_stability_status='pass_numerical_local')
            legacy, _, _, _ = task_metrics(self.f, h, m, task, config['common_acceptance'])
            new = band_metrics(self.f, h, p)
            for key in ('passband_max_loss_db', 'stopband_min_attenuation_db'):
                self.assertAlmostEqual(new[key], legacy[key], places=12)
            fitted = fit_response(self.f, h, p)
            for key in ('f0_hz', 'q', 'fitted_gain'):
                self.assertAlmostEqual(fitted[key], m[key], places=6)


if __name__ == '__main__':
    unittest.main()
