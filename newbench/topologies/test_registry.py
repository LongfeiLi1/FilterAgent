"""No simulator tests; symbolic nodal elimination is independent of registry formulas."""
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import sympy as sp
from newbench.topologies.registry import (
    TOPOLOGIES, FREQUENCIES, Q, analytic_seed, build_manifest,
    get_topology, ideal_coefficients, ideal_parameters, ideal_response,
    real_netlist_prefix, validate_passives,
)


def symbolic_nodal_transfer(topology_id):
    t = get_topology(topology_id)
    s = sp.Symbol('s')
    values = {k: sp.Symbol(k, positive=True) for k in t.passive_names}
    v = {'in': sp.Integer(1), 'vref': sp.Integer(0), 'out': sp.Symbol('out')}
    nodes = sorted({node for e in t.elements for node in e[1:3]} - {'in', 'vref', 'out'})
    v.update({node: sp.Symbol(node) for node in nodes})
    eqs = []
    # Infinite input impedance: zero current into both OTA input pins.
    for node in nodes:
        current = 0
        for component, a, b, key in t.elements:
            admittance = 1 / values[key] if component.startswith('R') else s * values[key]
            if node == a:
                current += admittance * (v[a] - v[b])
            elif node == b:
                current += admittance * (v[b] - v[a])
        eqs.append(current)
    eqs.append(v[t.ota_plus] - v[t.ota_minus])
    solved = sp.solve(eqs, [v[n] for n in nodes] + [v['out']], dict=True)
    if len(solved) != 1:
        raise AssertionError('Nodal equations do not have unique ideal solution')
    return s, values, sp.factor(solved[0][v['out']])


class TopologyTests(unittest.TestCase):
    def test_symbolic_nodal_elimination(self):
        for tid in TOPOLOGIES:
            with self.subTest(topology=tid):
                s, p, h = symbolic_nodal_transfer(tid)
                if tid == 'sallen_key_lowpass':
                    expected = 1 / (1 + s*(p['r1']+p['r2'])*p['cg'] + s*s*p['r1']*p['r2']*p['cg']*p['cf'])
                elif tid == 'sallen_key_highpass':
                    a = p['c1']*p['c2']*p['rf']*p['rg']
                    expected = a*s*s/(1+s*p['rf']*(p['c1']+p['c2'])+a*s*s)
                else:
                    expected = (-p['rfb']/p['rin'])/(1+s*p['cf']*(p['rlink']+p['rfb']+p['rfb']*p['rlink']/p['rin'])+s*s*p['rfb']*p['rlink']*p['cg']*p['cf'])
                self.assertEqual(sp.simplify(h - expected), 0)

    def test_numerical_back_substitution_random_passives(self):
        rng = np.random.default_rng(20261010)
        for tid, t in TOPOLOGIES.items():
            s, symbols, h = symbolic_nodal_transfer(tid)
            for _ in range(8):
                p = {k: float(np.exp(rng.uniform(np.log(t.physical_bounds[k][0]), np.log(t.physical_bounds[k][1])))) for k in t.passive_names}
                fs = np.geomspace(1, 1e8, 15)
                values = {symbols[k]: v for k, v in p.items()}
                reference = np.array([complex(h.subs(values | {s: 2j*np.pi*f})) for f in fs])
                np.testing.assert_allclose(ideal_response(tid, p, fs), reference, rtol=2e-13, atol=2e-14)

    def test_all_analytic_seeds_match_exact_targets_and_domain(self):
        for tid in TOPOLOGIES:
            for f0 in FREQUENCIES:
                p = analytic_seed(tid, f0)
                result = ideal_parameters(tid, p)
                self.assertAlmostEqual(result['f0_hz']/f0, 1, places=14)
                self.assertAlmostEqual(result['q'], Q, places=14)
                validate_passives(tid, p, enforce_bounds=True)

    def test_dc_and_high_frequency_behavior(self):
        for tid in TOPOLOGIES:
            h = ideal_response(tid, analytic_seed(tid, FREQUENCIES[0]), [0., 1e12])
            if tid == 'sallen_key_highpass':
                self.assertEqual(h[0], 0)
                self.assertAlmostEqual(abs(h[1]), 1, places=12)
            else:
                self.assertEqual(h[0], -1 if tid == 'mfb_lowpass_proposal' else 1)
                self.assertLess(abs(h[1]), 1e-12)

    def test_manifest_hash_domain_and_gate_invariants(self):
        m = build_manifest()
        self.assertEqual(len(m['cases']), 9)
        self.assertFalse(m['runnable'])
        for tid in ('T1', 'T2', 'T3'):
            cases = [c for c in m['cases'] if c['type_id'] == tid]
            self.assertEqual(len({c['structure_hash'] for c in cases}), 1)
            self.assertEqual(len({json.dumps(c['physical_bounds'], sort_keys=True) for c in cases}), 1)
            self.assertEqual([c['f0_hz'] for c in cases], list(FREQUENCIES))
            for c in cases:
                self.assertFalse(c['runnable'])
                self.assertFalse(c['simulation_authorized'])
                self.assertEqual(c['gate_status'], 'unmeasured')
        self.assertEqual(len({c['structure_hash'] for c in m['cases']}), 3)
        self.assertEqual([c['case_id'] for c in m['cases'] if c['role']=='development'], ['T1L1', 'T1L3'])
        self.assertTrue(all(c['topology_id']=='mfb_lowpass_proposal' for c in m['cases'] if c['type_id']=='T3'))

    def test_finite_highpass_profile_and_lowpass_legacy_edges(self):
        for c in build_manifest()['cases']:
            level = c['level']
            if c['type_id'] == 'T2':
                self.assertEqual(c['gain_reference_hz'], 40000*level)
                self.assertEqual(c['passband_hz'], [20000*level, 100000*level])
                self.assertEqual(c['stopband_hz'], [1, 3500*level])
            else:
                self.assertEqual(c['gain_reference_hz'], 1)
                self.assertEqual(c['passband_hz'], [1, 8000*level])
                self.assertEqual(c['stopband_hz'], [45000*level, 1e8])

    def test_explicit_invalid_inputs_fail(self):
        tid = 'sallen_key_highpass'
        p = analytic_seed(tid, FREQUENCIES[0])
        for bad in (p | {'unexpected': 1}, p | {'c1': 0}, p | {'rf': math.nan}):
            with self.assertRaises(ValueError):
                ideal_coefficients(tid, bad)
        with self.assertRaises(ValueError):
            get_topology('mfb_bandpass')

    def test_closed_dc_prefix_equivalent_to_zero_loop_source(self):
        # Header stub isolates topology construction without PDK or simulator access.
        with patch('rc_agent.physical.header', return_value='* existing header\n'):
            for tid in ('sallen_key_lowpass', 'sallen_key_highpass'):
                p = analytic_seed(tid, FREQUENCIES[0])
                instrumented = real_netlist_prefix(tid, p, object(), 5e-12)
                closed = real_netlist_prefix(tid, p, object(), 5e-12, closed_dc_only=True)
                normalized = '\n'.join(line.replace('ampout', 'out').replace('Vin in 0 DC 1.2 AC 1', 'Vin in 0 1.2').replace('Vref vref 0 DC 1.2 AC 0', 'Vref vref 0 1.2')
                                       for line in instrumented.splitlines() if not line.startswith(('Vloop ', 'Iport ')))+'\n'
                self.assertEqual(closed, normalized)
                self.assertIn('Vref vref 0 1.2', closed)
                self.assertNotIn('ampout', closed)
                with self.assertRaises(ValueError):
                    real_netlist_prefix(tid, p, object(), 5e-12, input_spec='DC 1.2 AC 1', closed_dc_only=True)

    def test_proposal_lowpass_cannot_construct_execution_deck(self):
        with self.assertRaisesRegex(ValueError, 'analytics only'):
            real_netlist_prefix('mfb_lowpass_proposal', analytic_seed('mfb_lowpass_proposal', FREQUENCIES[0]), object(), 5e-12)



if __name__ == '__main__':
    unittest.main(verbosity=2)
