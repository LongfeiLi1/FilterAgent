"""Synthetic tests only. Importing/running this module invokes no simulator."""
import unittest
import numpy as np
from topology_stability.core import ContractError, ac_response, local_modes, recover_charge
from .contract import explicit_hp_contract, expected_hp_variables, build_hp_matrix_deck
from .evidence import inspect_hp_linear_evidence, highpass_step_metrics


def fixture(unstable=False):
    # Deliberately reverse order: node-set order cannot label matrix rows.
    names = tuple(reversed(expected_hp_variables()))
    layout, ports, anchors = explicit_hp_contract(names, 20e-12, 30e-12)
    index = layout.validate(); n = len(names)
    g, c = np.zeros((n, n)), np.zeros((n, n))
    for node in layout.dynamic_nodes:
        i = index[node]; g[i, i] = 1e-3; c[i, i] = 1e-12
    c[index['x'], index['x']] = 0
    for a, b, value in [('in', 'x', 20e-12), ('x', 'y', 30e-12)]:
        i, j = index[a], index[b]
        c[i, i] += value; c[j, j] += value
        c[i, j] -= value; c[j, i] -= value
    g[index['out'], index['y']] = -1e-4
    for source in layout.sources:
        i, j = index[source.branch], index[source.node]
        g[i, j] = g[j, i] = 1
    if unstable:
        g[index[layout.dynamic_nodes[-1]], index[layout.dynamic_nodes[-1]]] = -1e-3
    return layout, ports, anchors, g, c


def prefix():
    return '''* Synthetic textual fixture; NOT a real OTA model
Vdd vdd 0 1.8
Vin in 0 DC 1.2
Vref vref 0 DC 1.2
XB vdd 0 vbias bias
XU y out out vdd 0 vbias ota
C1 in x 2e-11
C2 x y 3e-11
Rf x out 100000
Rg y vref 200000
CL out 0 5e-12
'''


class HPValidationTests(unittest.TestCase):
    def test_explicit_38_state_32_dynamic_coverage(self):
        layout, _, _ = explicit_hp_contract(expected_hp_variables(), 2e-11, 3e-11)
        self.assertEqual(len(layout.names), 38)
        self.assertEqual(len(layout.dynamic_nodes), 32)
        self.assertEqual({s.branch for s in layout.sources}, {'vin#branch', 'vdd#branch', 'vref#branch'})

    def test_missing_body_node_rejected(self):
        with self.assertRaises(ContractError):
            explicit_hp_contract(expected_hp_variables()[1:], 2e-11, 3e-11)

    def test_vref_branch_not_dropped(self):
        with self.assertRaises(ContractError):
            explicit_hp_contract(tuple(n for n in expected_hp_variables() if n != 'vref#branch'), 2e-11, 3e-11)

    def test_nonpositive_anchor_rejected(self):
        for value in (0, -1, float('nan')):
            with self.assertRaises(ContractError): explicit_hp_contract(expected_hp_variables(), value, 3e-11)

    def test_all_synthetic_stable_modes_retained(self):
        layout, _, _, g, c = fixture()
        r = local_modes(g, c, layout, layout.names)
        self.assertEqual(r['status'], 'stable')
        self.assertEqual(r['finite_mode_count'], 32)
        self.assertEqual(r['discarded_dynamic_modes'], 0)
        self.assertFalse(r['full_circuit_qualified'])

    def test_unobserved_internal_rhp_not_discarded(self):
        layout, _, _, g, c = fixture(True)
        r = local_modes(g, c, layout, layout.names)
        self.assertEqual(r['status'], 'unstable')
        self.assertEqual(r['finite_mode_count'], 32)
        self.assertTrue(any(p[0] > 0 for p in r['poles_rad_s']))

    def test_two_step_three_anchor_charge_recovery(self):
        layout, _, anchors, g, c = fixture()
        got, r = recover_charge(g, [g+1e9*c, g+1e12*c], layout, layout.names, anchors)
        np.testing.assert_allclose(got, c, rtol=1e-12, atol=1e-25)
        self.assertLess(r['charge_relative_difference'], 1e-5)

    def test_anchor_mismatch_rejected(self):
        layout, _, anchors, g, c = fixture()
        bad = c.copy(); bad[layout.names.index('x'), layout.names.index('in')] *= 2
        with self.assertRaises(ContractError):
            recover_charge(g, [g+1e9*bad, g+1e12*bad], layout, layout.names, anchors)

    def test_wrong_observed_order_rejected(self):
        layout, _, _, g, c = fixture()
        with self.assertRaises(ContractError): local_modes(g, c, layout, sorted(layout.names))

    def test_singular_charge_does_not_drop_mode(self):
        layout, _, _, g, c = fixture()
        idx = layout.names.index(layout.dynamic_nodes[-1]); c[idx, :] = c[:, idx] = 0
        with self.assertRaises(ContractError): local_modes(g, c, layout, layout.names)

    def test_prepared_matrix_deck_not_runnable(self):
        r = build_hp_matrix_deck(prefix(), 2e-11, 3e-11)
        self.assertFalse(r['runnable']); self.assertFalse(r['simulation_authorized'])
        self.assertEqual(r['matrix_mapping_status'], 'unresolved')
        self.assertIn('alter Vin acmag=1', r['deck'])

    def test_wrong_outer_wiring_rejected(self):
        with self.assertRaises(ContractError):
            build_hp_matrix_deck(prefix().replace('C2 x y', 'C2 x out'), 2e-11, 3e-11)

    def test_open_output_probe_rejected(self):
        with self.assertRaises(ContractError):
            build_hp_matrix_deck(prefix() + 'Vloop ampout out 0\n', 2e-11, 3e-11)

    def test_unaccounted_x_charge_rejected(self):
        with self.assertRaises(ContractError):
            build_hp_matrix_deck(prefix() + 'Cextra x out 1e-12\n', 2e-11, 3e-11)

    def test_missing_load_rejected(self):
        with self.assertRaisesRegex(ContractError, 'missing required top-level element: cl'):
            build_hp_matrix_deck(prefix().replace('CL out 0 5e-12\n', ''), 2e-11, 3e-11)

    def test_missing_bias_instance_rejected(self):
        with self.assertRaisesRegex(ContractError, 'missing required top-level element: xb'):
            build_hp_matrix_deck(prefix().replace('XB vdd 0 vbias bias\n', ''), 2e-11, 3e-11)

    def test_duplicate_ota_instance_rejected(self):
        with self.assertRaisesRegex(ContractError, 'duplicate top-level element: xu'):
            build_hp_matrix_deck(prefix() + 'XU y out out vdd 0 vbias ota\n', 2e-11, 3e-11)

    def test_invalid_load_rejected(self):
        for value in ('NaN', 'inf', '-1e-12', '0', 'bad'):
            with self.subTest(value=value), self.assertRaises(ContractError):
                build_hp_matrix_deck(prefix().replace('CL out 0 5e-12', 'CL out 0 ' + value), 2e-11, 3e-11)

    def test_exact_component_set_required(self):
        for line in ('Vdd vdd 0 1.8', 'Vin in 0 DC 1.2', 'Vref vref 0 DC 1.2',
                     'XB vdd 0 vbias bias', 'XU y out out vdd 0 vbias ota',
                     'C1 in x 2e-11', 'C2 x y 3e-11', 'Rf x out 100000',
                     'Rg y vref 200000', 'CL out 0 5e-12'):
            with self.subTest(line=line, mutation='missing'), self.assertRaises(ContractError):
                build_hp_matrix_deck(prefix().replace(line+'\n', ''), 2e-11, 3e-11)
            with self.subTest(line=line, mutation='duplicate'), self.assertRaises(ContractError):
                build_hp_matrix_deck(prefix()+line+'\n', 2e-11, 3e-11)

    def test_registered_real_prefix_static_integration(self):
        from newbench.topologies.registry import analytic_seed, real_netlist_prefix
        from rc_agent.physical import Devices
        p = analytic_seed('sallen_key_highpass', 12648.820123726657)
        netlist = real_netlist_prefix('sallen_key_highpass', p, Devices(), 5e-12, closed_dc_only=True)
        r = build_hp_matrix_deck(netlist, p['c1'], p['c2'])
        self.assertIn('Vref vref 0 1.2', r['deck'])
        self.assertEqual(r['expected_mna_dimension'], 38)
        self.assertFalse(r['runnable'])

    def evidence(self, mapping):
        layout, ports, _, g, c = fixture()
        f = np.geomspace(1, 1e8, 20)
        op = {n: 1.2 for n in layout.names if '#branch' not in n}
        return inspect_hp_linear_evidence(mapping_provenance=mapping,
                    matrix_names_in_exporter_order=layout.names, observed_names=layout.names,
                    c1_f=2e-11, c2_f=3e-11, g=g, jacobians=[g+1e9*c,g+1e12*c],
                    op_voltages=op, dc_trajectories=[{n:[v,v] for n,v in op.items()}]*2,
                    frequencies_hz=f, measured_ac=ac_response(g,c,layout,layout.names,ports,f))

    def test_absent_mapping_stays_unknown(self):
        r = self.evidence(None)
        self.assertEqual(r['status'], 'unknown'); self.assertFalse(r['runnable'])

    def test_sorted_op_is_not_mapping(self):
        r = self.evidence({'method':'sorted_op'})
        self.assertEqual(r['status'], 'unknown')
        self.assertIn('not mapping evidence',r['reason'])

    def test_synthetic_evidence_not_case_qualification(self):
        r = self.evidence(dict(method='independent_mapping_probe',evidence_path='synthetic-fixture-only',
                      netlist_sha256='synthetic',simulator_sha256='no-simulator',review_record='unit-test'))
        self.assertEqual(r['status'], 'stable')
        self.assertFalse(r['full_circuit_qualified']); self.assertFalse(r['runnable'])

    def step(self, offset=0):
        t=np.linspace(0,1,1001); v=1.2+offset+np.where(t>=.1, .001*np.exp(-30*(t-.1)), 0)
        return highpass_step_metrics(t,v,dc_out_v=1.2,input_edge_time_s=.1,tail_start_s=.8,
                                     final_error_limit_v=1e-5,tail_p2p_limit_v=1e-5)

    def test_hp_step_returns_to_original_bias(self):
        r=self.step(); self.assertEqual(r['step_status'],'pass'); self.assertEqual(r['expected_final_v'],1.2)

    def test_hp_step_offset_fails(self):
        self.assertEqual(self.step(.001)['step_status'],'fail')


if __name__ == '__main__': unittest.main()
