import dataclasses
import unittest

import numpy as np

from .core import (ChargeAnchor, ContractError, GroundedSource, Layout, Ports,
                   ac_response, check_linear_evidence, local_modes, recover_charge)
from .decks import ExtractionPorts, build_matrix_deck
from .sk_reference import DYNAMIC_NODES, explicit_sk_contract


def hidden_fixture(hidden_g=.0005):
    layout = Layout(('in', 'out', 'hidden', 'vin#branch'), ('out', 'hidden'),
                    (GroundedSource('vin#branch', 'in'),))
    g = np.array([[.001, -.001, 0, 1], [-.001, .001, 0, 0],
                  [0, 0, hidden_g, 0], [1, 0, 0, 0]], dtype=float)
    c = np.diag([0., 1e-6, 1e-6, 0.])
    return g, c, layout, Ports({'vin#branch': 1.}, {'out': 1.})


class MatrixContractTests(unittest.TestCase):
    def setUp(self):
        self.g, self.c, self.layout, self.ports = hidden_fixture()

    def modes(self, **changes):
        args = dict(g=self.g, c=self.c, layout=self.layout, observed_names=self.layout.names)
        args.update(changes)
        return local_modes(**args)

    def test_stable_fixture_all_modes_retained(self):
        result = self.modes()
        self.assertEqual(result['status'], 'stable')
        self.assertEqual(result['finite_mode_count'], 2)
        self.assertEqual(result['discarded_dynamic_modes'], 0)
        self.assertFalse(result['full_circuit_qualified'])
        np.testing.assert_allclose(sorted(p[0] for p in result['poles_rad_s']), [-1000, -500])

    def test_unobservable_unstable_mode_retained_despite_identical_transfer(self):
        g, c, layout, ports = hidden_fixture(-.0005)
        f = np.logspace(0, 5, 20)
        np.testing.assert_allclose(ac_response(g, c, layout, layout.names, ports, f),
                                   ac_response(self.g, self.c, layout, layout.names, ports, f))
        result = self.modes(g=g)
        self.assertEqual(result['status'], 'unstable')
        self.assertEqual(result['finite_mode_count'], 2)
        np.testing.assert_allclose(sorted(p[0] for p in result['poles_rad_s']), [-1000., 500.], rtol=1e-12)

    def test_zero_mode_is_unknown(self):
        g, _, _, _ = hidden_fixture(0.)
        self.assertEqual(self.modes(g=g)['status'], 'unknown')

    def test_repeated_eigenvalues_preserve_multiplicity(self):
        g, _, _, _ = hidden_fixture(.001)
        self.assertEqual(self.modes(g=g)['finite_mode_count'], 2)

    def test_matrix_permutation_with_explicit_mapping(self):
        order = [3, 2, 0, 1]
        names = tuple(self.layout.names[i] for i in order)
        layout = dataclasses.replace(self.layout, names=names)
        g, c = self.g[np.ix_(order, order)], self.c[np.ix_(order, order)]
        self.assertEqual(self.modes(g=g, c=c, layout=layout, observed_names=names)['status'], 'stable')
        f = [10., 100., 1000.]
        np.testing.assert_allclose(ac_response(g, c, layout, names, self.ports, f),
                                   ac_response(self.g, self.c, self.layout, self.layout.names, self.ports, f))

    def test_wrong_export_order_rejected(self):
        with self.assertRaisesRegex(ContractError, 'name/order'):
            self.modes(observed_names=self.layout.names[::-1])

    def test_missing_node_rejected(self):
        layout = dataclasses.replace(self.layout, names=('in', 'out', 'vin#branch'))
        with self.assertRaisesRegex(ContractError, 'coverage'):
            self.modes(layout=layout)

    def test_undeclared_state_rejected(self):
        layout = dataclasses.replace(self.layout, dynamic_nodes=('out',))
        with self.assertRaisesRegex(ContractError, 'coverage'):
            self.modes(layout=layout)

    def test_duplicate_names_rejected(self):
        layout = dataclasses.replace(self.layout, names=('in', 'out', 'out', 'vin#branch'))
        with self.assertRaisesRegex(ContractError, 'duplicate'):
            self.modes(layout=layout)

    def test_dimension_mismatch_rejected(self):
        with self.assertRaisesRegex(ContractError, 'dimension'):
            self.modes(g=self.g[:3, :3])

    def test_nonfinite_or_complex_matrix_rejected(self):
        g = self.g.copy()
        g[1, 1] = np.nan
        with self.assertRaises(ContractError):
            self.modes(g=g)
        with self.assertRaises(ContractError):
            self.modes(g=self.g.astype(complex))

    def test_algebraic_dynamic_state_not_silently_deleted(self):
        c = self.c.copy()
        c[2, 2] = 0
        with self.assertRaisesRegex(ContractError, 'rank.*no state removed'):
            self.modes(c=c)

    def test_source_charge_rejected(self):
        c = self.c.copy()
        c[3, 3] = 1e-15
        with self.assertRaisesRegex(ContractError, 'source branch has charge'):
            self.modes(c=c)

    def test_fake_source_incidence_rejected(self):
        g = self.g.copy()
        g[1, 3] = .1
        with self.assertRaisesRegex(ContractError, 'pure grounded incidence'):
            self.modes(g=g)

    def test_non_grounded_source_is_explicitly_unsupported(self):
        g = self.g.copy()
        g[3, 2] = g[2, 3] = -1.
        with self.assertRaisesRegex(ContractError, 'pure grounded incidence'):
            self.modes(g=g)

    def test_missing_port_mapping_rejected(self):
        for ports in (Ports({'missing': 1.}, {'out': 1.}), Ports({'vin#branch': 1.}, {'missing': 1.})):
            with self.assertRaisesRegex(ContractError, 'missing port'):
                ac_response(self.g, self.c, self.layout, self.layout.names, ports, [1., 10., 100.])

    def test_inverting_output_mapping(self):
        ports = Ports({'vin#branch': 1.}, {'out': -1.})
        actual = ac_response(self.g, self.c, self.layout, self.layout.names, ports, [100.])
        np.testing.assert_allclose(actual, [-1/(1+2j*np.pi*100/1000)])

    def test_highpass_capacitive_input_has_no_dc_normalization(self):
        layout = Layout(('drive', 'sense', 'vexc#branch'), ('sense',), (GroundedSource('vexc#branch', 'drive'),))
        g = np.array([[0., 0., 1.], [0., .001, 0.], [1., 0., 0.]])
        c = np.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., 0.]])*1e-6
        f = np.logspace(-3, 6, 50)
        response = ac_response(g, c, layout, layout.names, Ports({'vexc#branch': 1.}, {'sense': 1.}), f)
        np.testing.assert_allclose(response, (2j*np.pi*f*.001)/(1+2j*np.pi*f*.001))
        self.assertEqual(local_modes(g, c, layout, layout.names)['status'], 'stable')


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.g, self.c, self.layout, self.ports = hidden_fixture()
        self.js = [self.g + a*self.c for a in (1e5, 1e8)]
        self.anchors = (ChargeAnchor('out', 'out', 1e-6), ChargeAnchor('hidden', 'hidden', 1e-6))

    def recover(self, **changes):
        args = dict(g=self.g, jacobians=self.js, layout=self.layout,
                    observed_names=self.layout.names, anchors=self.anchors)
        args.update(changes)
        return recover_charge(**args)

    def evidence_args(self):
        op = {'in': 1.2, 'out': 1.2, 'hidden': 0.}
        trajectories = [{k: [v]*3 for k, v in op.items()} for _ in self.js]
        f = np.logspace(0, 6, 30)
        return dict(g=self.g, jacobians=self.js, layout=self.layout, observed_names=self.layout.names,
                    anchors=self.anchors, ports=self.ports, op_voltages=op, dc_trajectories=trajectories,
                    frequencies_hz=f, measured_ac=1/(1+2j*np.pi*f/1000))

    def test_two_jacobians_recover_analytic_charge(self):
        c, detail = self.recover()
        np.testing.assert_allclose(c, self.c)
        np.testing.assert_allclose(detail['integration_coefficients_per_s'], [1e5, 1e8])

    def test_full_synthetic_evidence(self):
        result = check_linear_evidence(**self.evidence_args())
        self.assertEqual(result['status'], 'stable')
        self.assertLess(result['ac_reconstruction_error'], 1e-12)

    def test_missing_dc_node_rejected(self):
        args = self.evidence_args()
        del args['dc_trajectories'][0]['hidden']
        with self.assertRaisesRegex(ContractError, 'trajectory node coverage'):
            check_linear_evidence(**args)

    def test_dc_drift_rejected(self):
        args = self.evidence_args()
        args['dc_trajectories'][0]['hidden'][-1] += .01
        with self.assertRaisesRegex(ContractError, 'changed DC'):
            check_linear_evidence(**args)

    def test_wrong_output_mapping_fails_ac(self):
        args = self.evidence_args()
        args['ports'] = Ports({'vin#branch': 1.}, {'hidden': 1.})
        with self.assertRaisesRegex(ContractError, 'AC reconstruction'):
            check_linear_evidence(**args)

    def test_bad_or_missing_anchor_rejected(self):
        for anchors in ((), (ChargeAnchor('missing', 'out', 1e-6),),
                        (ChargeAnchor('out', 'out', 0.),), self.anchors+(self.anchors[0],)):
            with self.assertRaises(ContractError):
                self.recover(anchors=anchors)

    def test_disagreeing_anchors_rejected(self):
        with self.assertRaisesRegex(ContractError, 'anchors disagree'):
            self.recover(anchors=(self.anchors[0], ChargeAnchor('hidden', 'hidden', 2e-6)))

    def test_same_or_missing_jacobian_rejected(self):
        for js in (self.js[:1], [self.js[0], self.js[0]]):
            with self.assertRaises(ContractError):
                self.recover(jacobians=js)

    def test_two_step_charge_disagreement_rejected(self):
        js = [x.copy() for x in self.js]
        js[1][2, 2] *= 2
        with self.assertRaisesRegex(ContractError, 'charge matrices disagree'):
            self.recover(jacobians=js, anchors=self.anchors[:1])


class DeckTests(unittest.TestCase):
    def setUp(self):
        _, _, self.layout, _ = hidden_fixture()
        self.prefix = '* artificial fixture\nVin in 0 DC 1.2\nR1 in out 1000\nC1 out 0 1e-6\nRh hidden 0 2000\nCh hidden 0 1e-6\n'
        self.ports = ExtractionPorts('Vin', 'out')

    def test_builder_preserves_prefix_and_maps_ports(self):
        deck = build_matrix_deck(self.prefix, self.layout, self.ports)
        self.assertTrue(deck.startswith(self.prefix))
        self.assertEqual(deck.count('\nmdump\n'), 3)
        self.assertIn('alter Vin acmag=1', deck)
        self.assertIn('let hr=real(v(out))', deck)
        self.assertNotIn('Vloop', deck)

    def test_active_waveform_rejected_without_silent_termination(self):
        for suffix in (' AC 1', ' PULSE(1.2 1.201 0 1n 1n 1u 2u)'):
            with self.assertRaisesRegex(ContractError, 'DC-only'):
                build_matrix_deck(self.prefix.replace('DC 1.2', 'DC 1.2'+suffix), self.layout, self.ports)

    def test_missing_or_extra_source_rejected(self):
        for prefix in (self.prefix.replace('Vin in 0 DC 1.2\n', ''), self.prefix+'Vextra other 0 0\n'):
            with self.assertRaisesRegex(ContractError, 'source incidence'):
                build_matrix_deck(prefix, self.layout, self.ports)

    def test_analysis_in_prefix_rejected(self):
        with self.assertRaisesRegex(ContractError, 'circuit statements only'):
            build_matrix_deck(self.prefix+'.control\nop\n.endc\n', self.layout, self.ports)

    def test_output_missing_and_control_injection_rejected(self):
        for output in ('absent', 'out)\nquit', 'vin#branch'):
            with self.assertRaises(ContractError):
                build_matrix_deck(self.prefix, self.layout, ExtractionPorts('Vin', output))

    def test_sk_assumptions_only_in_explicit_adapter(self):
        names = DYNAMIC_NODES + ('in', 'vdd', 'vin#branch', 'vdd#branch')
        layout, ports, anchors = explicit_sk_contract(names, 40e-12)
        self.assertEqual(len(layout.names), 36)
        self.assertEqual(len(layout.dynamic_nodes), 32)
        self.assertEqual(ports.input_rhs, {'vin#branch': 1.})
        self.assertEqual(anchors[1], ChargeAnchor('a', 'out', -40e-12))
        with self.assertRaisesRegex(ContractError, 'coverage'):
            explicit_sk_contract(names[:-1], 40e-12)


if __name__ == '__main__':
    unittest.main()
