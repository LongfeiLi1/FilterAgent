"""Explicit outer topologies. No simulations, optimization, or gate promotion.

Transfer coefficients are independently checked from nodal equations by tests.
The transistor subcircuit and PDK header are imported unchanged, on demand.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
import numpy as np

VERSION = 'filteragent_type_level_text_v1'
SOURCE = 'authorized textual handoff 2026-10-10; damaged Library ZIP not consumed'
FREQUENCIES = (12648.820123726657, 25297.640247453313, 37946.46037117997)
Q = 0.7071067811865476
OLD_BOUNDS = {'r1': (1e4, 1e6), 'r2': (1e4, 1e6), 'cg': (5e-12, 1e-10), 'cf': (5e-12, 2e-10),
              'current_scale': (.7, 1.4), 'output_scale': (.7, 1.4), 'cc_pf': (1., 4.)}


@dataclass(frozen=True)
class Topology:
    topology_id: str
    type_id: str
    family: str
    elements: tuple  # (component, positive node, negative node, parameter)
    ota_plus: str
    ota_minus: str
    bounds_mapping: tuple
    execution_policy: str = 'requires_per_case_gates'

    @property
    def passive_names(self):
        return tuple(row[3] for row in self.elements)

    @property
    def physical_bounds(self):
        return {name: list(OLD_BOUNDS[source]) for name, source in self.bounds_mapping} | {
            name: list(OLD_BOUNDS[name]) for name in ('current_scale', 'output_scale', 'cc_pf')}

    @property
    def structure(self):
        # Target, sizing, level, bias magnitude and load magnitude never enter this hash.
        return {'order': 2, 'sections': 1, 'ota_count': 1,
                'elements': [list(e[:3]) for e in self.elements],
                'ota_ports': [self.ota_plus, self.ota_minus, 'ampout', 'vdd', '0', 'vbias'],
                'fixed_edges': [['Vloop', 'ampout', 'out'], ['Iport', '0', 'out'],
                                ['CL', 'out', '0'], ['Vref', 'vref', '0'],
                                ['Vin', 'in', '0'], ['Vdd', 'vdd', '0'],
                                ['XB', 'vdd', '0', 'vbias']]}

    @property
    def structure_hash(self):
        return hashlib.sha256(json.dumps(self.structure, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


TOPOLOGIES = MappingProxyType({t.topology_id: t for t in (
    Topology('sallen_key_lowpass', 'T1', 'lowpass',
             (('R1', 'in', 'x', 'r1'), ('R2', 'x', 'y', 'r2'), ('Cg', 'y', 'vref', 'cg'), ('Cf', 'x', 'out', 'cf')),
             'y', 'out', (('r1', 'r1'), ('r2', 'r2'), ('cg', 'cg'), ('cf', 'cf'))),
    Topology('sallen_key_highpass', 'T2', 'highpass',
             (('C1', 'in', 'x', 'c1'), ('C2', 'x', 'y', 'c2'), ('Rf', 'x', 'out', 'rf'), ('Rg', 'y', 'vref', 'rg')),
             'y', 'out', (('c1', 'cg'), ('c2', 'cf'), ('rf', 'r1'), ('rg', 'r2'))),
    Topology('mfb_lowpass_proposal', 'T3', 'inverting_lowpass',
             (('Rin', 'in', 'x', 'rin'), ('Rfb', 'x', 'out', 'rfb'), ('Rlink', 'x', 'n', 'rlink'),
              ('Cg', 'x', 'vref', 'cg'), ('Cf', 'n', 'out', 'cf')),
             'vref', 'n', (('rin', 'r1'), ('rfb', 'r2'), ('rlink', 'r1'), ('cg', 'cg'), ('cf', 'cf')), 'proposal_only'),
)})


def get_topology(topology_id):
    try:
        return TOPOLOGIES[topology_id]
    except KeyError as exc:
        raise ValueError('Unknown topology_id: ' + str(topology_id)) from exc


def validate_passives(topology_id, passives, enforce_bounds=False):
    t = get_topology(topology_id)
    if set(passives) != set(t.passive_names):
        raise ValueError('Explicit passive keys mismatch: expected ' + repr(t.passive_names))
    p = {k: float(v) for k, v in passives.items()}
    if not all(math.isfinite(v) and v > 0 for v in p.values()):
        raise ValueError('Passives must be finite positive SI values')
    if enforce_bounds:
        for k, v in p.items():
            lo, hi = t.physical_bounds[k]
            if not lo <= v <= hi:
                raise ValueError('Passive outside frozen domain: ' + k)
    return p


def ideal_coefficients(topology_id, passives):
    """Return descending numerator and denominator coefficients in physical s."""
    p = validate_passives(topology_id, passives)
    family = get_topology(topology_id).family
    if family == 'lowpass':
        a = p['r1'] * p['r2'] * p['cg'] * p['cf']
        b = (p['r1'] + p['r2']) * p['cg']
        numerator = [0., 0., 1.]
    elif family == 'highpass':
        a = p['c1'] * p['c2'] * p['rf'] * p['rg']
        b = p['rf'] * (p['c1'] + p['c2'])
        numerator = [a, 0., 0.]
    else:
        a = p['rfb'] * p['rlink'] * p['cg'] * p['cf']
        b = p['cf'] * (p['rlink'] + p['rfb'] + p['rfb'] * p['rlink'] / p['rin'])
        numerator = [0., 0., -p['rfb'] / p['rin']]
    return numerator, [a, b, 1.]


def ideal_parameters(topology_id, passives):
    _, (a, b, _) = ideal_coefficients(topology_id, passives)
    return {'f0_hz': 1 / (2 * math.pi * math.sqrt(a)), 'q': math.sqrt(a) / b}


def ideal_response(topology_id, passives, frequency_hz):
    f = np.asarray(frequency_hz, dtype=float)
    if not np.all(np.isfinite(f)) or np.any(f < 0):
        raise ValueError('Frequency must be finite and nonnegative')
    num, den = ideal_coefficients(topology_id, passives)
    s = 2j * np.pi * f
    return np.polyval(num, s) / np.polyval(den, s)


def analytic_seed(topology_id, f0_hz):
    if not math.isfinite(f0_hz) or f0_hz <= 0:
        raise ValueError('Positive finite target frequency required')
    w = 2 * math.pi * f0_hz
    family = get_topology(topology_id).family
    if family == 'lowpass':
        r = 1 / (w * math.sqrt(20e-12 * 40e-12))
        p = dict(r1=r, r2=r, cg=20e-12, cf=40e-12)
    elif family == 'highpass':
        rf = 1 / (w * 20e-12 * math.sqrt(2))
        p = dict(c1=20e-12, c2=20e-12, rf=rf, rg=2 * rf)
    else:
        r = 1 / (w * math.sqrt(90e-12 * 20e-12))
        p = dict(rin=r, rfb=r, rlink=r, cg=90e-12, cf=20e-12)
    return validate_passives(topology_id, p, enforce_bounds=True)


def real_netlist_prefix(topology_id, passives, devices, load_f, input_spec=None, closed_dc_only=False):
    """Unterminated real-OTA prefix; caller must supply and validate analyses.

    T3 is deliberately rejected until a separately authorized execution phase.
    Never treats this deck construction as measurement or gate acceptance.
    """
    t = get_topology(topology_id)
    if t.execution_policy == 'proposal_only':
        raise ValueError('mfb_lowpass_proposal is registered for analytics only')
    p = validate_passives(topology_id, passives, enforce_bounds=True)
    if not math.isfinite(load_f) or load_f <= 0:
        raise ValueError('Positive explicit load required')
    from rc_agent.physical import header
    if closed_dc_only and input_spec is not None:
        raise ValueError('DC-only construction does not accept waveform override')
    if closed_dc_only:
        input_spec = '1.2'
    elif input_spec is None:
        input_spec = 'DC 1.2 AC 1'
    if '\n' in input_spec or '\r' in input_spec:
        raise ValueError('Input source spec must occupy exactly one line')
    output_node = 'out' if closed_dc_only else 'ampout'
    reference_source = 'Vref vref 0 1.2' if closed_dc_only else 'Vref vref 0 DC 1.2 AC 0'
    lines = [header(devices), 'Vdd vdd 0 1.8', reference_source,
             'Vin in 0 ' + input_spec, 'XB vdd 0 vbias bias',
             f'XU {t.ota_plus} {t.ota_minus} {output_node} vdd 0 vbias ota']
    if not closed_dc_only:
        lines += ['Vloop ampout out DC 0 AC 0', 'Iport 0 out DC 0 AC 0']
    lines += [f'{name} {a} {b} {p[key]:.16g}' for name, a, b, key in t.elements]
    lines.append(f'CL out 0 {load_f:.16g}')
    return '\n'.join(lines) + '\n'


def build_manifest():
    cases = []
    for topology in TOPOLOGIES.values():
        for i, f0 in enumerate(FREQUENCIES):
            level = i + 1
            case_id = topology.type_id + 'L' + str(level)
            hp = topology.family == 'highpass'
            cases.append({
                'case_id': case_id, 'type_id': topology.type_id, 'level': level,
                'topology_id': topology.topology_id, 'structure_hash': topology.structure_hash,
                'order': 2, 'sections': 1, 'ota_count': 1,
                'role': 'development' if case_id in ('T1L1', 'T1L3') else 'development_candidate',
                'existing_development_alias': {'T1L1': 'A', 'T1L3': 'B'}.get(case_id),
                'existing_task_id': {'T1L1': 'reference_100ksps', 'T1L3': 'power_load_300ksps'}.get(case_id),
                'runnable': False, 'gate_status': 'unmeasured', 'execution_policy': topology.execution_policy,
                'freeze_status': 'proposal_pending_type_level_review',
                'simulation_authorized': False, 'optimization_authorized': False,
                'f0_hz': f0, 'q': Q, 'f0_relative_tolerance': .02, 'q_relative_tolerance': .05,
                'load_f': (5e-12, 7.5e-12, 1e-11)[i], 'power_limit_w': (200e-6, 175e-6, 150e-6)[i],
                'gain_reference_range': [.95, 1.05], 'required_polarity': -1 if topology.type_id == 'T3' else 1,
                'profile': 'bounded_finite_reference_highpass_v1' if hp else 'legacy_1hz_single_sided_lowpass_v1',
                'passband_hz': [20000. * level, 100000. * level] if hp else [1., 8000. * level],
                'stopband_hz': [1., 3500. * level] if hp else [45000. * level, 1e8],
                'gain_reference_hz': 40000. * level if hp else 1.,
                'passband_max_loss_db': 1., 'stopband_min_attenuation_db': 20.,
                'physical_bounds': topology.physical_bounds,
                'bounds_source_mapping': dict(topology.bounds_mapping),
                'analytic_seed': analytic_seed(topology.topology_id, f0),
                'required_gates': ['topology', 'conditions', 'metric_extraction', 'operating_point',
                                   'complete_local_stability', 'phase_margin_definition', 'power'],
                'phase_margin_min_deg': 60., 'phase_margin_status': 'requires_topology_verification',
            })
    return {'version': VERSION, 'source': SOURCE, 'zip_status': 'truncated_unresolved_not_consumed',
            'heldout': False, 'runnable': False, 'llm_enabled': False,
            'execution_status': 'paused_pending_type_level_freeze_review',
            'scope': 'registration and analytic validation, not nine implemented accepted cases',
            'mfb_note': 'T3 remains mfb_lowpass_proposal; future MFB bandpass requires a distinct topology_id and version',
            'bounds_source': 'configs/three_group_frozen.json physical_bounds',
            'environment': {'header': 'rc_agent.physical.header (unchanged)', 'corner': 'tt',
                            'vdd_v': 1.8, 'input_dc_v': 1.2, 'vref_dc_v': 1.2,
                            'power_definition': 'quiescent -I(Vdd)*1.8 including bias reference-current branch; no signal-source power'},
            'band_definition': {'loss': 'max(0,max(-20*log10(abs(H)/Gref)))',
                                'attenuation': 'min(-20*log10(abs(H)/Gref))',
                                'boundary_interpolation': 'log frequency / log magnitude',
                                'highpass': 'bounded pass and stop bands, both boundaries included; never DC normalization'},
            'cases': cases}


if __name__ == '__main__':
    print(json.dumps(build_manifest(), indent=2))
